"""Bounded PDF and structured full-text acquisition shared by scholarly providers."""
from __future__ import annotations

import gzip
import hashlib
from io import BytesIO
import os
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

from radar.common.evidence_blocks import clip_structure
from radar.common.logging import logger
from radar.common.sources import content_hash
from radar.common.retrieval_diagnostics import safe_url

LOG = logger('pdf')
MAX_CONTENT_BYTES = 30 * 1024 * 1024


class ContentKeyRequired(ValueError):
    """Report a missing content-service credential without issuing an unauthenticated download."""


def _download_bytes(url: str, timeout: float, diagnostics: dict[str, Any] | None = None) -> bytes:
    """Stream a bounded document, attaching credentials only to OpenAlex's content host."""
    params = {}
    if urlparse(url).hostname == 'content.openalex.org':
        key = os.getenv('OPENALEX_API_KEY')
        if not key:
            raise ContentKeyRequired('OpenAlex cached content requires OPENALEX_API_KEY')
        params['api_key'] = key
    with requests.get(url, params=params or None, timeout=timeout, stream=True,
                      headers={'User-Agent': 'radar/0.1 review replication'}) as response:
        if diagnostics is not None:
            status = response.status_code
            media = response.headers.get('Content-Type', '')
            diagnostics.update(full_text_final_url=safe_url(response.url) or safe_url(url),
                full_text_content_type=media[:120] if isinstance(media, str) else '')
            if type(status) is int:
                diagnostics['full_text_http_status'] = status
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > MAX_CONTENT_BYTES:
                raise ValueError('document exceeds download limit')
            chunks.append(chunk)
    return b''.join(chunks)


def _failure(exc: Exception, diagnostics: dict[str, Any], url: str) -> dict[str, Any]:
    """Keep bounded failure reasons; the evidence service adds paper/provider context at INFO."""
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    message = str(exc)
    trusted = isinstance(exc, ValueError) and message.startswith(('PDF URL returned', 'document exceeds',
        'decompressed XML', 'TEI ', 'full-text XML', 'OpenAlex cached content'))
    detail = f'HTTP {status}' if type(status) is int else message[:300] if trusted else type(exc).__name__
    reason = ('api_key_required' if isinstance(exc, ContentKeyRequired) else
        'http_error' if type(status) is int else 'timeout' if isinstance(exc, requests.Timeout) else
        'connection_error' if isinstance(exc, requests.ConnectionError) else
        'non_pdf_content' if message == 'PDF URL returned non-PDF content' else
        'download_limit' if trusted and 'limit' in message else
        'invalid_document' if isinstance(exc, (ValueError, ET.ParseError)) else 'extraction_failed')
    LOG.debug('document extraction failed url=%s reason=%s detail=%s', safe_url(url), reason, detail)
    return {'full_text_status': 'api_key_required' if isinstance(exc, ContentKeyRequired) else 'download_failed',
        'full_text_error': type(exc).__name__, 'full_text_error_detail': detail,
        'full_text_failure_reason': reason, **diagnostics}


def _extracted(text: str, data: bytes, url: str, maximum: int, kind: str, structure: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Keep the extraction format and source hash alongside the bounded evidence text."""
    return {'full_text': text[:maximum], 'full_text_url': url,
        'full_text_structure': clip_structure(structure, maximum),
        'full_text_complete': len(text) <= maximum, 'full_text_status': 'downloaded' if text.strip() else 'empty_document',
        'full_text_format': kind, 'full_text_source_sha256': hashlib.sha256(data).hexdigest(),
        'full_text_original_sha256': content_hash(text), **diagnostics}


def download_pdf_text(url: str, max_chars: int, timeout: float = 30) -> dict[str, Any]:
    """Extract an openly supplied or authenticated OpenAlex PDF within the shared bounds."""
    if not url.startswith(('https://', 'http://')):
        return {'full_text_status': 'no_pdf_url'}
    diagnostics: dict[str, Any] = {}
    try:
        data = _download_bytes(url, timeout, diagnostics)
        if not data.startswith(b'%PDF'):
            raise ValueError('PDF URL returned non-PDF content')
        from pypdf import PdfReader

        passages = [(page.extract_text() or '', {'kind': 'page', 'page': number})
            for number, page in enumerate(PdfReader(BytesIO(data)).pages, 1)]
        text, structure = _join_passages(passages)
        return _extracted(text, data, url, max_chars, 'pdf', structure, diagnostics)
    except Exception as exc:
        return _failure(exc, diagnostics, url)


def _local_name(node: ET.Element) -> str:
    """Read namespaced and historical case variants of TEI element names."""
    return node.tag.rsplit('}', 1)[-1].casefold()


def _join_passages(passages: list[tuple[str, dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]]:
    """Join extracted blocks while preserving each original paragraph or PDF page boundary."""
    parts, structure, offset = [], [], 0
    for passage, metadata in passages:
        if not passage:
            continue
        if parts:
            offset += 2
        structure.append({'start': offset, 'end': offset + len(passage), **metadata})
        parts.append(passage)
        offset += len(passage)
    return '\n\n'.join(parts), structure


def _tei_document(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Extract article paragraphs from modern and historical TEI without accepting HTML pages."""
    xml = data.decode('utf-8-sig')
    if '<!doctype' in xml.casefold() or '<!entity' in xml.casefold():
        raise ValueError('TEI entity and document-type declarations are not allowed')
    root = ET.fromstring(xml)
    tei = next((node for node in root.iter() if _local_name(node) == 'tei'), None)
    if tei is None:
        raise ValueError('full-text XML does not contain TEI')
    text = next((node for node in tei if _local_name(node) == 'text'), None)
    if text is None:
        raise ValueError('TEI has no article text')
    body = next((node for node in text if _local_name(node) == 'body'), None)
    if body is None:
        if not any(_local_name(node) == 'div' for node in text):
            raise ValueError('TEI has no article body')
        body = text
    blocks = []
    body_content = False

    def collect(node: ET.Element, section: str = '') -> None:
        """Preserve inline text and mathematics once per block without duplicating descendants."""
        nonlocal body_content
        name = _local_name(node)
        if name in {'listbibl', 'biblstruct', 'bibl'} or name == 'div' and node.attrib.get('type', '').casefold() in {'references', 'bibliography'}:
            return
        if name == 'div':
            heading = next((child for child in node if _local_name(child) == 'head'), None)
            if heading is not None:
                section = ''.join(heading.itertext()).strip()
        if name in {'head', 'p', 'formula', 'figdesc', 'table'}:
            passage = ''.join(node.itertext()).strip()
            if passage:
                blocks.append((passage, {'kind': 'paragraph' if name == 'p' else name, 'section': section}))
                body_content = body_content or name != 'head'
        else:
            for child in node:
                collect(child, section)

    collect(body)
    if body is not text:
        for child in text:
            if _local_name(child) == 'back':
                collect(child)
    if not blocks or not body_content:
        raise ValueError('TEI article body contains no extractable text')
    headers = [node for node in tei if _local_name(node) == 'teiheader']
    abstracts = [''.join(node.itertext()).strip() for header in headers for node in header.iter() if _local_name(node) == 'abstract']
    return _join_passages([(value, {'kind': 'abstract', 'section': 'Abstract'}) for value in abstracts if value] + blocks)


def _tei_text(data: bytes) -> str:
    """Return the article text when callers do not need its structural annotations."""
    return _tei_document(data)[0]


def download_tei_text(url: str, max_chars: int, timeout: float = 30) -> dict[str, Any]:
    """Use cached Grobid TEI as an alternate source when PDF delivery or extraction fails."""
    if not url.startswith(('https://', 'http://')):
        return {'full_text_status': 'no_full_text_url'}
    diagnostics: dict[str, Any] = {}
    try:
        data = _download_bytes(url, timeout, diagnostics)
        if data.startswith(b'\x1f\x8b'):
            with gzip.GzipFile(fileobj=BytesIO(data)) as compressed:
                data = compressed.read(MAX_CONTENT_BYTES + 1)
            if len(data) > MAX_CONTENT_BYTES:
                raise ValueError('decompressed XML exceeds download limit')
        text, structure = _tei_document(data)
        return _extracted(text, data, url, max_chars, 'tei_xml', structure, diagnostics)
    except Exception as exc:
        return _failure(exc, diagnostics, url)
