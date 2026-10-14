"""Bounded PDF acquisition shared by all scholarly providers."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import requests

from ids_meta.common.logging import logger
from ids_meta.common.sources import content_hash

LOG = logger("pdf")


def download_pdf_text(url: str, max_chars: int, timeout: float = 30) -> dict[str, Any]:
    """Extract an openly supplied PDF with bounded download and text size."""
    if not url.startswith(("https://", "http://")):
        return {"full_text_status": "no_pdf_url"}
    try:
        with requests.get(url, timeout=timeout, stream=True,
                          headers={"User-Agent": "ids-meta/0.1 review replication"}) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 30 * 1024 * 1024:
                    raise ValueError("PDF exceeds download limit")
                chunks.append(chunk)
        data = b"".join(chunks)
        if not data.startswith(b"%PDF"):
            raise ValueError("PDF URL returned non-PDF content")
        from pypdf import PdfReader
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
        return {"full_text": text[:max_chars], "full_text_url": url,
            "full_text_complete": len(text) <= max_chars, "full_text_status": "downloaded" if text.strip() else "empty_pdf",
            "full_text_original_sha256": content_hash(text)}
    except Exception as exc:
        LOG.info("study PDF unavailable (%s)", type(exc).__name__)
        return {"full_text_status": "download_failed", "full_text_error": type(exc).__name__}

