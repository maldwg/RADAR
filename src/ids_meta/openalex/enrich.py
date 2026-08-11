from __future__ import annotations

from io import BytesIO
from typing import Any

import requests

from ids_meta.common.logging import logger
from ids_meta.common.text import normalize_text

LOG = logger("openalex.enrich")


class ScholarlyEvidenceClient:
    """Resolve missing scholarly evidence by DOI or exact normalized title."""

    api_root = "https://api.semanticscholar.org/graph/v1"
    fields = "title,abstract,openAccessPdf,url,externalIds"

    def __init__(self, timeout: float = 30, api_key: str = "", max_full_text_chars: int = 30000):
        """Configure Semantic Scholar lookup and optional PDF extraction limits."""
        self.timeout = max(1.0, float(timeout))
        self.max_full_text_chars = max(0, int(max_full_text_chars))
        self.headers = {"User-Agent": "ids-meta/0.1 scholarly evidence enrichment"}
        if api_key:
            self.headers["x-api-key"] = api_key

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch one JSON object from a scholarly metadata endpoint."""
        response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, dict) else {}

    def _paper(self, title: str, doi: str) -> dict[str, Any]:
        """Resolve a paper by DOI first and exact normalized title second."""
        if doi:
            try:
                return self._get_json(f"{self.api_root}/paper/DOI:{doi}", {"fields": self.fields})
            except requests.RequestException as exc:
                LOG.info("DOI evidence lookup failed doi=%s error=%s; trying exact title", doi, exc)
        payload = self._get_json(
            f"{self.api_root}/paper/search", {"query": title, "limit": 10, "fields": self.fields},
        )
        expected = normalize_text(title)
        for paper in payload.get("data", []) if isinstance(payload.get("data"), list) else []:
            if isinstance(paper, dict) and normalize_text(paper.get("title")) == expected:
                return paper
        return {}

    def _full_text(self, url: str) -> str:
        """Best-effort extract bounded text from an open-access PDF."""
        if not url or not self.max_full_text_chars:
            return ""
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not url.lower().split("?", 1)[0].endswith(".pdf"):
                return ""
            from pypdf import PdfReader
            pages: list[str] = []
            used = 0
            for page in PdfReader(BytesIO(response.content)).pages:
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                remaining = self.max_full_text_chars - used
                if remaining <= 0:
                    break
                pages.append(text[:remaining])
                used += len(pages[-1])
            return "\n\n".join(pages)
        except Exception as exc:  # Optional PDF parsers and malformed documents fail in varied ways.
            LOG.info("optional full-text enrichment unavailable url=%s error=%s", url, exc)
            return ""

    def enrich(self, work: dict[str, Any]) -> dict[str, str]:
        """Return any abstract and optional open-access full text found for a work."""
        title = str(work.get("title", "")).strip()
        if not title:
            return {}
        try:
            paper = self._paper(title, str(work.get("doi", "")).strip())
        except (requests.RequestException, ValueError) as exc:
            LOG.warning("title evidence lookup failed title=%s error=%s", title, exc)
            return {}
        if not paper:
            return {}
        pdf = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else {}
        pdf_url = str(pdf.get("url") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        full_text = self._full_text(pdf_url)
        result = {
            "abstract": abstract, "abstract_source": "Semantic Scholar",
            "full_text": full_text, "full_text_url": pdf_url,
        }
        return {key: value for key, value in result.items() if value}
