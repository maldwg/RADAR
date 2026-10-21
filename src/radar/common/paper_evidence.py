"""Shared scholarly evidence acquisition for all paper screening strategies."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

from radar.adapters.contracts import EvidenceAdapter, ProviderDescriptor
from radar.adapters.profiles import paper_identifiers
from radar.common.logging import logger
from radar.common.pdf import download_pdf_text, download_tei_text
from radar.common.retrieval_diagnostics import lookup_failure, paper_context, safe_url

LOG = logger('paper_evidence')
FULL_TEXT_AUDIT_FIELDS = ('full_text_status', 'full_text_error', 'full_text_error_detail', 'full_text_format', 'full_text_source_sha256', 'full_text_http_status', 'full_text_content_type', 'full_text_final_url', 'full_text_failure_reason', 'full_text_attempts_json')
DEFAULT_FULL_TEXT_PROVIDERS = ('openalex', 'semantic_scholar')
FULL_TEXT_RETRIEVAL_POLICY = 'structured-tei-before-pdf-v1'


class PaperEvidenceService:
    """Apply one configurable, cached full-text acquisition chain for every paper study."""
    def __init__(self, config: dict[str, Any], enricher: EvidenceAdapter | None = None):
        """Bind provider ordering and allow an explicitly injected offline or custom enricher."""
        self.config, self.enricher = config, enricher
        self.enrichers: dict[str, EvidenceAdapter] = {'injected': enricher} if enricher is not None else {}
        self.results: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
        self.document_results: dict[str, dict[str, Any]] = {}

    def _lookup(self, provider: str, work: dict[str, Any], uncertain: bool, audit: list[dict[str, Any]]) -> dict[str, Any]:
        """Report record resolution separately from document availability and API failure."""
        if provider not in self.enrichers:
            from radar.adapters.registry import create_adapter

            self.enrichers[provider] = create_adapter(provider, {
                'timeout_seconds': self.config.get('evidence_lookup_timeout_seconds', 30),
                'max_full_text_chars': 0}, Path())
        key = (provider, json.dumps(paper_identifiers(work), sort_keys=True), str(work.get('title', '')), uncertain)
        cached = key in self.results
        started = time.monotonic()
        context = paper_context(work)
        if not cached:
            LOG.info('full-text lookup start %s provider=%s', context, provider)
            try:
                self.results[key] = self.enrichers[provider].enrich(work)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                failure = lookup_failure(exc)
                self.results[key] = {'evidence_lookup_attempts': [{'method': 'provider', **failure}],
                    **({'evidence_lookup_error': failure['error'], 'evidence_lookup_error_detail': failure['detail']}
                        if failure['status'] == 'lookup_failed' else {})}
        result = self.results[key]
        attempts = result.get('evidence_lookup_attempts', [])
        for attempt in attempts:
            if attempt['status'] == 'lookup_failed':
                LOG.warning('full-text lookup request failed %s provider=%s method=%s identifier=%s http_status=%s error=%s detail=%s cached=%s',
                    context, provider, attempt.get('method', '-'), attempt.get('identifier', '-'),
                    attempt.get('http_status') or '-', attempt.get('error', '-'), attempt.get('detail', '-'), str(cached).lower())
        links = len([url for url in self._urls({}, result, True) if url])
        resolved = any(result.get(field) for field in ('evidence_native_id', 'abstract'))
        status = ('text_available' if str(result.get('full_text') or '').strip() else
            'links_found' if links else 'record_found_no_document' if resolved else
            'lookup_failed' if result.get('evidence_lookup_error') else 'no_match')
        entry = {'stage': 'lookup', 'provider': result.get('evidence_provider') or provider,
            'status': status, 'error': result.get('evidence_lookup_error', ''),
            'detail': result.get('evidence_lookup_error_detail', ''), 'links': links, 'attempts': attempts}
        audit.append(entry)
        LOG.info('full-text lookup complete %s provider=%s outcome=%s links=%d cached=%s elapsed_seconds=%.3f',
            context, provider, status, links, str(cached).lower(), time.monotonic() - started)
        return result

    def _download(self, urls: list[str], attempted: set[str], audit: list[dict[str, Any]],
                  work: dict[str, Any], provider: str) -> dict[str, Any]:
        """Identify every failed or successful document and keep HTTP/redirect diagnostics."""
        result = {}
        for url in urls:
            if not url or url in attempted:
                continue
            attempted.add(url)
            cached = url in self.document_results
            started = time.monotonic()
            if not cached:
                download = download_tei_text if urlparse(url).path.endswith('.grobid-xml') else download_pdf_text
                self.document_results[url] = download(url, self.config.get('max_full_text_chars', 120000),
                    self.config.get('evidence_lookup_timeout_seconds', 30))
            result = self.document_results[url]
            entry = {'stage': 'download', 'provider': provider, 'url': safe_url(url),
                'status': result.get('full_text_status', 'downloaded' if result.get('full_text') else 'unavailable'),
                'format': result.get('full_text_format', ''), 'error': result.get('full_text_error', ''),
                'detail': result.get('full_text_error_detail', ''), 'reason': result.get('full_text_failure_reason', ''),
                'http_status': result.get('full_text_http_status'), 'content_type': result.get('full_text_content_type', ''),
                'final_url': safe_url(result.get('full_text_final_url', ''))}
            audit.append(entry)
            LOG.info('full-text download complete %s provider=%s outcome=%s url=%s final_url=%s http_status=%s content_type=%s format=%s chars=%d reason=%s error=%s detail=%s cached=%s elapsed_seconds=%.3f',
                paper_context(work), provider, entry['status'], entry['url'], entry['final_url'] or '-',
                entry['http_status'] or '-', entry['content_type'] or '-', entry['format'] or '-',
                len(result.get('full_text', '').strip()), entry['reason'] or '-', entry['error'] or '-',
                entry['detail'] or '-', str(cached).lower(), time.monotonic() - started)
            if result.get('full_text', '').strip():
                return {**result, 'full_text_url': url, 'full_text_error': '', 'full_text_error_detail': '', 'full_text_failure_reason': ''}
        return {key: value for key, value in result.items() if key not in {'full_text', 'full_text_url', 'full_text_complete', 'full_text_original_sha256', 'full_text_source_sha256', 'full_text_format', 'full_text_structure'}}

    @staticmethod
    def _summary(work: dict[str, Any], result: dict[str, Any], audit: list[dict[str, Any]],
                 fetch: bool, started: float) -> None:
        """Emit one final paper outcome without treating missing links as failed API calls."""
        downloads = [entry for entry in audit if entry['stage'] == 'download']
        lookups = [entry for entry in audit if entry['stage'] == 'lookup']
        request_errors = sum(sum(attempt['status'] == 'lookup_failed' for attempt in entry['attempts'])
            or bool(entry['error']) for entry in lookups)
        lookup_errors = sum(entry['status'] == 'lookup_failed' for entry in lookups)
        failures = sum(entry['status'] != 'downloaded' for entry in downloads)
        text = result.get('full_text') or work.get('full_text') or ''
        if result.get('full_text', '').strip():
            outcome = 'retrieved'
        elif str(work.get('full_text') or '').strip():
            outcome = 'existing_text_retained' if audit else 'already_available'
        elif not fetch:
            outcome = 'not_requested'
        elif downloads:
            outcome = ('api_key_required' if all(d['status'] == 'api_key_required' for d in downloads) else
                'no_extractable_text' if all(d['status'] in {'empty_document', 'empty_pdf'} for d in downloads) else 'download_failed')
        else:
            outcome = 'lookup_incomplete' if lookup_errors else 'no_document_link'
        LOG.info('full-text retrieval complete %s outcome=%s providers=%s lookups=%d lookup_failures=%d lookup_request_failures=%d downloads=%d download_failures=%d chars=%d format=%s url=%s policy=%s elapsed_seconds=%.3f',
            paper_context(work), outcome, ','.join(entry['provider'] for entry in lookups) or '-', len(lookups),
            lookup_errors, request_errors, len(downloads), failures, len(text), result.get('full_text_format') or work.get('full_text_format') or '-',
            safe_url(result.get('full_text_url') or work.get('full_text_url', '')) or '-', FULL_TEXT_RETRIEVAL_POLICY, time.monotonic() - started)

    @staticmethod
    def _urls(work: dict[str, Any], discovered: dict[str, Any], derive: bool) -> list[str]:
        """Prefer declared TEI documents, then supplied and derived PDF alternatives."""
        urls = []
        for candidate in (work, discovered):
            content = candidate.get('content_urls') or {}
            if not isinstance(content, dict):
                raise ValueError('content_urls must be an object')
            urls.extend(content[kind] for kind in ('grobid_xml', 'pdf') if content.get(kind))
        for candidate in (work, discovered):
            urls.extend(str(candidate.get(key) or '') for key in ('pdf_url', 'full_text_url'))
            pdfs = candidate.get('pdf_urls') or []
            if not isinstance(pdfs, list) or any(not isinstance(url, str) for url in pdfs):
                raise ValueError('pdf_urls must be a list of URLs')
            urls.extend(pdfs)
        if derive:
            for candidate in (work, {'identifiers': discovered.get('evidence_identifiers', {})}):
                identifiers = paper_identifiers(candidate)
                if identifiers.get('arxiv'):
                    urls.append('https://arxiv.org/pdf/' + quote(identifiers['arxiv'], safe='/'))
        return sorted(dict.fromkeys(urls), key=lambda url: not urlparse(url).path.endswith('.grobid-xml'))

    def _structured_provider(self, provider: str) -> bool:
        """Discover structured-text capabilities without issuing provider requests."""
        from radar.adapters.registry import provider_descriptor

        descriptor = getattr(self.enricher, 'descriptor', None) if provider == 'injected' else provider_descriptor(provider)
        return isinstance(descriptor, ProviderDescriptor) and 'tei_xml' in descriptor.evidence_formats

    def prepare(self, work: dict[str, Any], *, uncertain: bool = False) -> dict[str, Any]:
        """Prefer structured TEI from any provider; parse PDFs when structured acquisition is exhausted."""
        started = time.monotonic()
        text = str(work.get('full_text') or '').strip()
        expand = uncertain and self.config.get('enrich_uncertain_with_full_text', False) and not work.get('full_text_complete', False)
        lookup_abstract = self.config.get('enrich_missing_abstracts', False) and not work.get('abstract') and not text
        fetch = self.config.get('fetch_full_text', False) or expand
        fallback = fetch and self.config.get('enrich_missing_full_text', False)
        result: dict[str, Any] = {}
        audit: list[dict[str, Any]] = []
        attempted: set[str] = set()
        acquire = fetch and (not text or expand)
        sources = [('candidate', self._urls(work, {}, fallback))] if acquire else []
        providers = (['injected'] if self.enricher is not None else
            list(self.config.get('full_text_lookup_providers', DEFAULT_FULL_TEXT_PROVIDERS)))
        lookup = expand or lookup_abstract or fallback and not text
        if not lookup:
            providers = []
        structured = [provider for provider in providers if self._structured_provider(provider)] if acquire else []
        if acquire:
            urls = [url for url in sources[0][1] if urlparse(url).path.endswith('.grobid-xml')]
            result.update(self._download(urls, attempted, audit, work, 'candidate'))
        # Structured-capable providers are resolved before any PDF. PDF-only
        # providers stay lazy until the supplied and already discovered PDFs fail.
        for provider in structured:
            if result.get('full_text', '').strip():
                break
            discovered = self._lookup(provider, work, uncertain, audit)
            result.update({key: value for key, value in discovered.items() if value not in (None, '', [], {})})
            urls = self._urls({}, discovered, fallback)
            sources.append((provider, urls))
            if not result.get('full_text', '').strip():
                result.update(self._download([url for url in urls if urlparse(url).path.endswith('.grobid-xml')],
                    attempted, audit, work, provider))
        if acquire and not result.get('full_text', '').strip():
            for provider, urls in sources:
                result.update(self._download([url for url in urls if not urlparse(url).path.endswith('.grobid-xml')],
                    attempted, audit, work, provider))
                if result.get('full_text', '').strip():
                    break
        for provider in (provider for provider in providers if provider not in structured):
            if result.get('full_text', '').strip():
                break
            discovered = self._lookup(provider, work, uncertain, audit)
            result.update({key: value for key, value in discovered.items() if value not in (None, '', [], {})})
            if acquire and not str(result.get('full_text') or '').strip():
                result.update(self._download(self._urls({}, discovered, fallback), attempted, audit, work, provider))
            if str(result.get('full_text') or '').strip() or not fetch and result.get('abstract'):
                break
        if str(result.get('full_text') or '').strip():
            result.update(full_text_error='', full_text_error_detail='', full_text_failure_reason='')
            result.setdefault('full_text_structure', [])
        elif text:
            for key in ('full_text_url', 'full_text_status', 'full_text_structure', 'full_text_complete', 'full_text_original_sha256', *FULL_TEXT_AUDIT_FIELDS[:-1]):
                result.pop(key, None)
        elif fetch and not attempted:
            result['full_text_status'] = 'no_pdf_url'
        if audit:
            result['full_text_attempts_json'] = json.dumps(audit, ensure_ascii=False, sort_keys=True)
        if fetch or audit:
            self._summary(work, result, audit, fetch, started)
        return {key: value for key, value in result.items() if value not in (None, '')
            or key in {'full_text_error', 'full_text_error_detail', 'full_text_failure_reason'} and result.get('full_text')}

    def enrich(self, work: dict[str, Any]) -> dict[str, Any]:
        """Expand evidence for an unresolved scientific assessment."""
        return self.prepare(work, uncertain=True)
