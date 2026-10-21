"""Ground model quotations in original source spans without fuzzy acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re


QUOTE_GROUNDING = "source-spans-typography-line-wrap-v3"
HTML_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);")
WRAP_MARKER = "\U0010fffe"
LINE_WRAP_RE = re.compile(r"(?<![\w-])(?P<left>[A-Z]?[a-z]{2,})-(?P<gap>[ \t]*\r?\n[ \t]*)(?=(?P<right>[a-z]{2,})(?!\w))")
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|\u2026)+")
TYPOGRAPHY = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "“": '"', "”": '"', "\u00ad": "",
}


def _indexed_text(text: str, replacements: dict[int, tuple[int, str]] | None = None) -> tuple[str, list[int], list[int]]:
    """Normalize typography and HTML entities while retaining original character offsets."""
    characters, starts, ends = [], [], []
    offset = 0
    while offset < len(text):
        entity = HTML_ENTITY_RE.match(text, offset)
        end = entity.end() if entity else offset + 1
        decoded = unescape(entity.group()) if entity else text[offset]
        if replacements and offset in replacements:
            end, decoded = replacements[offset]
        for character in decoded:
            for normalized in TYPOGRAPHY.get(character, character):
                if normalized.isspace():
                    if not characters:
                        continue
                    if characters[-1] == " ":
                        ends[-1] = end
                        continue
                    normalized = " "
                characters.append(normalized)
                starts.append(offset)
                ends.append(end)
        offset = end
    if characters and characters[-1] == " ":
        characters.pop()
        starts.pop()
        ends.pop()
    return "".join(characters), starts, ends


def _line_wraps(text: str, vocabulary: set[str] | None = None) -> dict[int, tuple[int, str]]:
    """Join letter-word line wraps, dropping a hyphen only with corroborating prose evidence."""
    replacements = {}
    for match in LINE_WRAP_RE.finditer(text):
        start = match.start('gap') - 1
        word = match['left'] + match['right']
        surroundings = text[max(0, match.start() - 24):match.start()] + text[match.end() + len(match['right']):match.end() + len(match['right']) + 24]
        mathematical = re.search(r'[=<>≤≥±×÷−+*/_^\\{}\[\]]', surroundings)
        droppable = vocabulary is not None and len(word) >= 6 and word.casefold() in vocabulary and not mathematical
        replacements[start] = (match.end('gap'), WRAP_MARKER if droppable else '-')
    return replacements


@dataclass(frozen=True)
class SourceQuotation:
    """Identify a verbatim source passage and the policy used to locate it."""
    text: str
    start: int
    end: int
    method: str


class SourceQuoteIndex:
    """Ground copied quotations in original source spans despite harmless rendering differences."""

    def __init__(self, text: str):
        """Index one frozen full text without changing its words or scientific notation."""
        self.text = text
        self.normalized, self.starts, self.ends = _indexed_text(text)
        self.wrapped: tuple[str, list[int], list[int]] | None = None

    def _occurrences(self, needle: str, *, word_boundary: bool = False, minimum_start: int = 0) -> list[int]:
        """Locate up to two complete matches without splitting a normalized ligature."""
        matches, offset = [], 0
        while len(matches) < 2:
            start = self.normalized.find(needle, offset)
            if start < 0:
                break
            end = start + len(needle)
            left_boundary = start == 0 or self.starts[start] != self.starts[start - 1]
            right_boundary = end == len(self.ends) or self.ends[end - 1] != self.ends[end]
            initial_boundary = start == 0 or not (self.normalized[start - 1].isalnum() or self.normalized[start - 1] == "_")
            if (self.starts[start] >= minimum_start and left_boundary and right_boundary
                    and (not word_boundary or initial_boundary)):
                matches.append(start)
            offset = start + 1
        return matches

    def _wrapped_occurrences(self, quote: str, *, start: int = 0, word_boundary: bool = False) -> list[SourceQuotation]:
        """Match literal characters with optional, source-attested PDF line-wrap seams."""
        if WRAP_MARKER in self.text or WRAP_MARKER in quote:
            return []
        if self.wrapped is None:
            vocabulary = {word.casefold() for word in re.findall(r'(?<![\w-])[A-Za-z]{4,}(?![\w-])', self.normalized)}
            self.wrapped = _indexed_text(self.text, _line_wraps(self.text, vocabulary))
        text, starts, ends = self.wrapped
        needle, _, _ = _indexed_text(quote, _line_wraps(quote))
        if not needle:
            return []
        pieces, previous = [], ''
        for character in needle:
            if previous.isascii() and previous.isalpha() and character.isascii() and character.isalpha():
                pieces.append(WRAP_MARKER + '?')
            pieces.append('(?:-|' + WRAP_MARKER + ')' if character == '-' else re.escape(character))
            previous = character
        pattern = re.compile(''.join(pieces))
        matches = []
        for found in pattern.finditer(text):
            left, right = found.span()
            if starts[left] < start or (left and starts[left] == starts[left - 1]):
                continue
            if right < len(ends) and ends[right - 1] == ends[right]:
                continue
            if word_boundary and left and (text[left - 1].isalnum() or text[left - 1] == '_'):
                continue
            matches.append(SourceQuotation(self.text[starts[left]:ends[right - 1]], starts[left], ends[right - 1], 'line_wrap'))
            if len(matches) == 2:
                break
        return matches

    def match(self, quote: str, *, start: int = 0) -> SourceQuotation | None:
        """Return an actual source span, preserving scientific characters through PDF wrapping."""
        needle, _, _ = _indexed_text(quote)
        if not needle:
            return None
        positions = self._occurrences(needle, minimum_start=start)
        if positions:
            left = self.starts[positions[0]]
            right = self.ends[positions[0] + len(needle) - 1]
            passage = self.text[left:right]
            return SourceQuotation(passage, left, right, 'exact' if passage == quote else 'typography')
        wrapped = self._wrapped_occurrences(quote, start=start)
        if wrapped:
            return wrapped[0]
        word = re.match(r'[A-Za-z]+\b', needle)
        if word is None or word.group().lower() not in {'a', 'an', 'the', 'this', 'these', 'those', 'we', 'our'}:
            return None
        decoded = HTML_ENTITY_RE.sub(lambda entity: unescape(entity.group()), quote)
        _, offsets, _ = _indexed_text(decoded)
        initial = offsets[0]
        alternative = decoded[:initial] + decoded[initial].swapcase() + decoded[initial + 1:]
        alternative_needle, _, _ = _indexed_text(alternative)
        candidates = {}
        for position in self._occurrences(alternative_needle, word_boundary=True):
            left, right = self.starts[position], self.ends[position + len(alternative_needle) - 1]
            candidates[left, right] = SourceQuotation(self.text[left:right], left, right, 'initial_case')
        for match in self._wrapped_occurrences(alternative, word_boundary=True):
            candidates.setdefault((match.start, match.end), SourceQuotation(match.text, match.start, match.end, 'initial_case_line_wrap'))
        if len(candidates) != 1:
            return None
        result = next(iter(candidates.values()))
        return result if result.start >= start else None

    def match_fragments(self, quote: str) -> tuple[SourceQuotation, ...] | None:
        """Ground catalogue ellipsis fragments in order, preserving every omitted-span boundary."""
        literal = self.match(quote)
        if literal is not None:
            return (literal,)
        decoded = HTML_ENTITY_RE.sub(lambda entity: unescape(entity.group()), quote)
        parts = ELLIPSIS_RE.split(decoded)
        fragments = [fragment.strip() for fragment in parts if fragment.strip()]
        if len(parts) < 2 or not fragments:
            return None
        matches = []
        offset = 0
        for fragment in fragments:
            match = self.match(fragment, start=offset)
            if match is None:
                return None
            matches.append(match)
            offset = match.end
        return tuple(matches)
