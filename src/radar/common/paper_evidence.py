"""Shared scholarly evidence acquisition for all paper screening strategies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from radar.adapters.contracts import EvidenceAdapter
from radar.common.pdf import download_pdf_text


class PaperEvidenceService:
    """Apply one evidence policy independently of scientific classification labels."""
    def __init__(self, config: dict[str, Any], enricher: EvidenceAdapter | None = None):
        """Bind acquisition limits and defer optional lookup client creation."""
        self.config, self.enricher = config, enricher
        self.results: dict[tuple[str, str, bool], dict[str, Any]] = {}
        self.pdf_results: dict[str, dict[str, Any]] = {}

    def prepare(self, work: dict[str, Any], *, uncertain: bool = False) -> dict[str, Any]:
        """Fetch missing abstracts or permitted PDF evidence, preserving existing text."""
        missing = not work.get("abstract") and not work.get("full_text")
        lookup = self.config.get("enrich_missing_abstracts", False) and missing
        lookup = lookup or (uncertain and self.config.get("enrich_uncertain_with_full_text", False)
            and not work.get("full_text_complete", False))
        result: dict[str, Any] = {}
        if lookup:
            if self.enricher is None:
                from radar.adapters.registry import create_adapter

                self.enricher = create_adapter("semantic_scholar", {
                    "timeout_seconds": self.config.get("evidence_lookup_timeout_seconds", 30),
                    "max_full_text_chars": self.config.get("max_full_text_chars", 30000)}, Path())
            key = (str(work.get("doi", "")), str(work.get("title", "")), uncertain)
            if key not in self.results:
                self.results[key] = self.enricher.enrich(work)
            result.update(self.results[key])
        text = result.get("full_text") or work.get("full_text")
        url = result.get("full_text_url") or work.get("pdf_url") or work.get("full_text_url")
        if self.config.get("fetch_full_text", False) and not text and url:
            if str(url) not in self.pdf_results:
                self.pdf_results[str(url)] = download_pdf_text(str(url), self.config.get("max_full_text_chars", 120000),
                    self.config.get("evidence_lookup_timeout_seconds", 30))
            result.update(self.pdf_results[str(url)])
        return {key: value for key, value in result.items() if value not in (None, "")}

    def enrich(self, work: dict[str, Any]) -> dict[str, Any]:
        """Expand evidence for an unresolved scientific assessment."""
        return self.prepare(work, uncertain=True)
