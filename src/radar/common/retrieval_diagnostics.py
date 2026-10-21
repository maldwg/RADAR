"""Safe request diagnostics for identity lookups and document retrieval."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def safe_url(value: Any) -> str:
    """Keep the document host/path while excluding credentials, queries and fragments."""
    if not isinstance(value, str):
        return ''
    try:
        parts = urlsplit(value)
        host = parts.hostname or ''
        if ':' in host:
            host = '[' + host + ']'
        if parts.port is not None:
            host += ':' + str(parts.port)
        return urlunsplit((parts.scheme, host, parts.path, '', ''))
    except ValueError:
        return '[invalid-url]'


def paper_context(work: dict[str, Any]) -> str:
    """Identify even DOI-less candidates and keep titles on one quoted log line."""
    identity = work.get('paper_id') or work.get('entity_id') or work.get('doi') or work.get('openalex_id') or work.get('semantic_scholar_id') or '-'
    fields = {'paper_id': identity, 'doi': work.get('doi') or '-', 'title': work.get('title') or '-'}
    return ' '.join(f'{key}={json.dumps(str(value)[:500], ensure_ascii=False)}' for key, value in fields.items())


def lookup_failure(exc: Exception) -> dict[str, Any]:
    """Unwrap provider errors without copying exception strings containing API keys."""
    chain, current = [], exc
    while current is not None and id(current) not in {id(item) for item in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    failure = next((item for item in reversed(chain) if getattr(getattr(item, 'response', None), 'status_code', None) is not None), chain[-1])
    status = getattr(getattr(failure, 'response', None), 'status_code', None)
    return {'status': 'not_found' if status == 404 else 'lookup_failed',
        'error': type(failure).__name__, 'http_status': status if isinstance(status, int) else None,
        'detail': f'HTTP {status}' if isinstance(status, int) else type(failure).__name__}


def lookup_diagnostics(attempts: list[dict[str, Any]], *, resolved: bool) -> dict[str, Any]:
    """Retain failed provider attempts even if a subsequent identity route succeeds."""
    result: dict[str, Any] = {'evidence_lookup_attempts': attempts}
    errors = [attempt for attempt in attempts if attempt['status'] == 'lookup_failed']
    if errors and not resolved:
        result.update(evidence_lookup_error=errors[-1]['error'], evidence_lookup_error_detail=errors[-1]['detail'])
    return result
