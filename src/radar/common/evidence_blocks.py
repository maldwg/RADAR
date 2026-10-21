"""Select immutable source passages by ID without asking models to reproduce text."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

EVIDENCE_BLOCK_VERSION = 'source-blocks-v1'
MAX_BLOCK_CHARS = 1600
BLOCK_ID_PATTERN = r'^B[0-9a-f]{12}-[0-9]{5,}$'


@dataclass(frozen=True)
class EvidenceBlock:
    """Keep an exact half-open character range and available extraction metadata."""
    id: str
    start: int
    end: int
    text: str
    kind: str
    page: int | None = None
    section: str = ''


def clip_structure(structure: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    """Clip extraction metadata alongside a deliberately bounded full-text window."""
    return [{**item, 'end': min(item['end'], length)} for item in structure if item['start'] < length]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Exclude boundary whitespace while retaining original source coordinates."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split(text: str, start: int, end: int):
    """Bound long blocks at sentence/whitespace seams without rewriting any characters."""
    start, end = _trim(text, start, end)
    while end - start > MAX_BLOCK_CHARS:
        limit = start + MAX_BLOCK_CHARS
        window = text[start:limit]
        seams = [m.start() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])', window) if m.start() >= MAX_BLOCK_CHARS // 2]
        cut = start + seams[-1] if seams else max(text.rfind(' ', start, limit), text.rfind('\n', start, limit))
        if cut <= start:
            cut = limit
        left, right = _trim(text, start, cut)
        if left < right:
            yield left, right
        start, end = _trim(text, cut, end)
    if start < end:
        yield start, end


class EvidenceBlockIndex:
    """Deterministically index a frozen extraction and resolve only its own block IDs."""

    def __init__(self, text: str, structure: list[dict[str, Any]] | None = None):
        """Freeze source coordinates and bind IDs to text, boundaries and the indexing policy."""
        if not isinstance(text, str):
            raise ValueError('evidence full_text must be a string')
        self.text = text
        self.text_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
        structure = [] if structure is None else structure
        if not isinstance(structure, list):
            raise ValueError('full_text_structure must be an array')
        regions = []
        previous = 0
        for item in structure:
            if not isinstance(item, dict):
                raise ValueError('invalid full_text_structure entry')
            start, end = item.get('start'), item.get('end')
            page, section, kind = item.get('page'), item.get('section', ''), item.get('kind', 'paragraph')
            if (type(start) is not int or type(end) is not int or not previous <= start < end <= len(text)
                    or text[previous:start].strip() or page is not None and (type(page) is not int or page < 1)
                    or not isinstance(section, str) or not isinstance(kind, str)):
                raise ValueError('full_text_structure must contain ordered, complete source ranges')
            regions.append((start, end, kind, page, section))
            previous = end
        if structure and text[previous:].strip():
            raise ValueError('full_text_structure omits source text')
        if not structure:
            regions = [(0, len(text), 'text_block', None, '')]
        raw = []
        for start, end, kind, page, section in regions:
            # PDF pages and plain text use blank-line blocks; individual TEI
            # paragraphs/headings/formulae already have structural boundaries.
            boundaries = list(re.finditer(r'\n[ \t]*\n|\f', text[start:end])) if kind in {'page', 'text_block'} else []
            left = start
            for separator in boundaries:
                right = start + separator.start()
                raw.extend((a, b, kind, page, section) for a, b in _split(text, left, right))
                left = start + separator.end()
            raw.extend((a, b, kind, page, section) for a, b in _split(text, left, end))
        identity = {'version': EVIDENCE_BLOCK_VERSION, 'text_sha256': self.text_sha256,
            'max_block_chars': MAX_BLOCK_CHARS, 'ranges': raw}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.blocks = tuple(EvidenceBlock(f'B{self.fingerprint[:12]}-{i:05d}', a, b, text[a:b], kind, page, section)
            for i, (a, b, kind, page, section) in enumerate(raw, 1))
        self.by_id = {block.id: block for block in self.blocks}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EvidenceBlockIndex:
        """Build an index from a paper's text and optional extraction structure."""
        return cls(record.get('full_text', ''), record.get('full_text_structure'))

    def prompt_blocks(self) -> list[dict[str, Any]]:
        """Give the model immutable IDs and original text, never expected judgments."""
        return [asdict(block) for block in self.blocks]

    def judgment_error(self, judgment: Any, values: list[str]) -> str | None:
        """Check schema separately from references that cannot ground a definite judgment."""
        if (not isinstance(judgment, dict) or set(judgment) != {'value', 'evidence_ids'}
                or judgment.get('value') not in values or not isinstance(judgment.get('evidence_ids'), list)
                or any(not isinstance(key, str) for key in judgment['evidence_ids'])):
            return 'invalid judgment schema'
        ids = judgment['evidence_ids']
        if judgment['value'] == 'unknown':
            return 'unknown requires empty evidence_ids' if ids else None
        if not ids or len(set(ids)) != len(ids) or any(key not in self.by_id for key in ids):
            return 'evidence_ids must select distinct blocks from supplied full_text'
        return None

    def manifest(self, judgments: list[dict[str, Any]]) -> dict[str, Any]:
        """Store selected passages separately in source order with exact offsets and hashes."""
        ids = {key for judgment in judgments for key in judgment.get('evidence_ids', [])}
        if ids - self.by_id.keys():
            raise ValueError('cannot export evidence_ids from a different extraction')
        return {'version': EVIDENCE_BLOCK_VERSION, 'fingerprint': self.fingerprint,
            'text_sha256': self.text_sha256,
            'passages': [asdict(block) for block in self.blocks if block.id in ids]}
