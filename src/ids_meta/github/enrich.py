from __future__ import annotations

import os
from typing import Any

from ids_meta.github.collect import GitHubClient


class GitHubEvidenceClient:
    """Fetch a larger README/documentation evidence set for unresolved repositories."""

    def __init__(self, settings: dict[str, Any]):
        """Configure a paced GitHub client and the expanded documentation bound."""
        self.max_docs_files = max(
            int(settings.get("max_docs_files", 3)),
            int(settings.get("uncertain_max_docs_files", 10)),
        )
        self.client = GitHubClient(
            os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
            request_pause_seconds=float(settings.get("request_pause_seconds", 1.0)),
            search_pause_seconds=float(settings.get("search_pause_seconds", 2.2)),
            rate_limit_max_wait_seconds=float(settings.get("rate_limit_max_wait_seconds", 3700)),
            rate_limit_fallback_wait_seconds=float(settings.get("rate_limit_fallback_wait_seconds", 60)),
            adaptive_rate_limit_pacing=bool(settings.get("adaptive_rate_limit_pacing", True)),
        )

    def enrich(self, payload: dict[str, Any]) -> dict[str, str]:
        """Return refreshed README and a broader bounded documentation sample."""
        repo = payload.get("repository") or {}
        full_name = str(repo.get("full_name") or "").strip()
        if not full_name:
            return {}
        readme = self.client.raw_readme(full_name)
        docs = self.client.docs(full_name, self.max_docs_files)
        return {
            key: value for key, value in {
                "readme_text": readme,
                "docs_text": docs,
            }.items() if value
        }
