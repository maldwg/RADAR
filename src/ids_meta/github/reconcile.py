from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from ids_meta.common.cache import PipelineCache
from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.logging import logger
from ids_meta.common.ollama import OllamaClient, decision_profile, legacy_runtime_decision_profile
from ids_meta.common.prompts import render_prompt
from ids_meta.common.reconcile import group_by_links, margin_confidence
from ids_meta.common.run import utc_now
from ids_meta.common.text import normalize_text, stable_hash
from ids_meta.github.schema import CATALOGUE_COLUMNS

LOG = logger("github.reconcile")
PROMPT_VERSION = "github-reconcile-5"
GITHUB_LINK_RE = re.compile(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.I)

VARIANT_TOKENS = {
    "docker", "container", "containers", "kubernetes", "k8s", "helm", "ansible", "terraform",
    "rules", "ruleset", "plugin", "plugins", "integration", "integrations", "wrapper", "sdk",
    "docs", "documentation", "website", "legacy", "deprecated", "archive", "mirror", "packages",
    "deployment", "installer", "examples", "demo",
}
RELATIONSHIP_TOKENS = (
    ("deployment_variant", {"docker", "container", "containers", "kubernetes", "k8s", "helm", "ansible", "terraform", "deployment", "installer", "packages"}),
    ("rules_content", {"rules", "ruleset"}),
    ("integration", {"integration", "integrations"}),
    ("plugin", {"plugin", "plugins"}),
    ("documentation", {"docs", "documentation", "website"}),
    ("legacy_version", {"legacy", "deprecated", "archive"}),
    ("mirror", {"mirror"}),
    ("wrapper", {"wrapper", "sdk"}),
    ("demo", {"examples", "demo"}),
)
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical_repo_id": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["canonical_repo_id", "reason"],
}
class Generator(Protocol):
    model: str
    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one structured model response for the supplied prompt."""
        ...


def _tokens(name: str) -> list[str]:
    """Normalize a repository name into comparison tokens."""
    return normalize_text(name).split()


def repository_stem(name: str) -> str:
    """Remove generic packaging or derivative suffixes from a repository name."""
    tokens = _tokens(name)
    while len(tokens) > 1 and tokens[-1] in VARIANT_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _homepage(row: dict[str, str]) -> str:
    """Return a normalized HTTP homepage suitable for family linking."""
    value = row.get("homepage", "").strip().casefold().rstrip("/")
    return value if value.startswith(("http://", "https://")) else ""


def _linked_names(row: dict[str, str], cache: PipelineCache | None) -> set[str]:
    """Extract explicitly linked GitHub repository names from cached project text."""
    if cache is None:
        return set()
    payload = cache.get_record("github.repository", row.get("github_repo_id", "")) or {}
    text = f"{payload.get('readme_text', '')} {payload.get('docs_text', '')}"
    return {match.group(1).rstrip("./").casefold() for match in GITHUB_LINK_RE.finditer(text)}


def _variant_tokens(row: dict[str, str]) -> set[str]:
    """Return generic derivative-role tokens found in a repository name."""
    return set(_tokens(row.get("name") or row.get("full_name", "").rsplit("/", 1)[-1])) & VARIANT_TOKENS


def _relationship(row: dict[str, str], canonical: dict[str, str]) -> str:
    """Label how a repository relates to its selected canonical project."""
    if row.get("github_repo_id") == canonical.get("github_repo_id"):
        return "canonical"
    if row.get("fork") == "true" and row.get("independent_fork") != "true":
        return "mirror"
    tokens = _variant_tokens(row)
    for label, candidates in RELATIONSHIP_TOKENS:
        if tokens & candidates:
            return label
    role = row.get("role", "")
    if role in {"rules_content", "research_project", "demo"}:
        return role
    if role in {"support_tooling", "classifier_engine", "detection_framework", "detection_platform"}:
        return role
    return "related_repository"


def _score_parts(row: dict[str, str], inbound_links: int) -> dict[str, float]:
    """Return named structural canonical-ranking components."""
    return {
        "nonfork": 5.0 if row.get("fork") != "true" else 0.0,
        "self_description": 4.0 if row.get("direct_self_description") == "true" else 0.0,
        "nonvariant": 4.0 if not _variant_tokens(row) else 0.0,
        "inbound_links": float(min(8, inbound_links * 2)),
        "commit": 1.0 if row.get("last_commit_sha") else 0.0,
    }


def _score(row: dict[str, str], inbound_links: int) -> float:
    """Rank canonical candidates using structural evidence rather than popularity."""
    return sum(_score_parts(row, inbound_links).values())


def _score_breakdown(row: dict[str, str], inbound_links: int) -> str:
    """Format a stable human-readable structural score decomposition."""
    parts = _score_parts(row, inbound_links)
    return " + ".join(f"{name}={value:.1f}" for name, value in parts.items()) + f" = {sum(parts.values()):.1f}"


def _prompt(group: list[dict[str, str]], scores: dict[str, float], prompt_config: dict[str, Any] | None = None) -> str:
    """Construct an ambiguity-resolution prompt for one repository family."""
    items = []
    for row in group:
        items.append(
            "\n".join((
                f"repository_id: {row.get('github_repo_id', '')}", f"full_name: {row.get('full_name', '')}",
                f"role: {row.get('role', '')}", f"fork: {row.get('fork', '')}",
                f"parent: {row.get('parent_full_name', '')}", f"description: {row.get('description', '')[:500]}",
                f"deterministic_score: {scores.get(row.get('github_repo_id', ''), 0):.1f}",
            ))
        )
    return render_prompt(prompt_config, {}, "\n\n---\n\n".join(items), "reconciliation_template")


def reconcile(
    classifications_path: Path, output: Path, cache: PipelineCache | None, ollama_config: dict[str, Any],
    reconciliation_config: dict[str, Any], dry_run: bool = False, generator: Generator | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Group accepted repositories and assign order-independent canonical relationships."""
    rows = read_csv(classifications_path)
    require_columns(rows, ("github_repo_id", "full_name", "decision", "role"), classifications_path)
    accepted = [row for row in rows if row.get("decision") == "accept"]
    links = {row.get("github_repo_id", ""): _linked_names(row, cache) for row in accepted}
    for row in accepted:
        upstream = row.get("upstream_full_name", "").strip().casefold()
        if upstream:
            links[row.get("github_repo_id", "")].add(upstream)

    def should_link(left: dict[str, str], right: dict[str, str]) -> bool:
        """Decide whether two accepted repositories belong to one project family."""
        if left.get("independent_fork") == "true" or right.get("independent_fork") == "true":
            independent_pair = True
        else:
            independent_pair = False
        left_name, right_name = left.get("full_name", "").casefold(), right.get("full_name", "").casefold()
        if not independent_pair and (left.get("parent_full_name", "").casefold() == right_name or right.get("parent_full_name", "").casefold() == left_name):
            return True
        same_owner = left.get("owner", "").casefold() == right.get("owner", "").casefold() and bool(left.get("owner"))
        same_stem = repository_stem(left.get("name", "")) == repository_stem(right.get("name", "")) and len(repository_stem(left.get("name", ""))) >= 4
        explicit = right_name in links.get(left.get("github_repo_id", ""), set()) or left_name in links.get(right.get("github_repo_id", ""), set())
        shared_homepage = bool(_homepage(left)) and _homepage(left) == _homepage(right)
        derivative_context = bool(_variant_tokens(left) or _variant_tokens(right))
        return (same_owner and same_stem) or (explicit and (same_owner or same_stem or derivative_context)) or (shared_homepage and same_stem)

    groups = group_by_links(accepted, should_link)
    LOG.info("reconciling GitHub catalogue accepted=%d project_families=%d", len(accepted), len(groups))
    model_client = generator
    profile_hash = ""
    if generator is not None:
        profile_hash = decision_profile(generator.model, PROMPT_VERSION, ollama_config, {"reconciliation_template": (prompt_config or {}).get("reconciliation_template", "")})
    by_id = {row.get("github_repo_id", ""): row for row in accepted}
    reconciled_fields: dict[str, dict[str, str]] = {}
    ambiguity_margin = float(reconciliation_config.get("canonical_ambiguity_margin", 2.0))
    use_llm = bool(reconciliation_config.get("reconcile_with_llm", True)) and not dry_run
    for group_index, group in enumerate(groups, 1):
        group = sorted(group, key=lambda row: row.get("github_repo_id", ""))
        inbound: dict[str, int] = {row.get("github_repo_id", ""): 0 for row in group}
        group_names = {row.get("full_name", "").casefold(): row.get("github_repo_id", "") for row in group}
        for source_links in links.values():
            for linked in source_links:
                if linked in group_names:
                    inbound[group_names[linked]] += 1
        scores = {row.get("github_repo_id", ""): _score(row, inbound[row.get("github_repo_id", "")]) for row in group}
        base_candidates = [
            row for row in group
            if not _variant_tokens(row) and (row.get("fork") != "true" or row.get("independent_fork") == "true")
        ]
        ordered = sorted(base_candidates, key=lambda row: (-scores[row.get("github_repo_id", "")], row.get("full_name", "").casefold(), row.get("github_repo_id", "")))
        canonical = ordered[0] if ordered else None
        margin: float | None = None
        resolution = "base_repository_missing" if canonical is None else "deterministic"
        adjudicated = False
        if canonical is None:
            reason = "no eligible base repository in family; only derivative or non-independent fork repositories were found"
        elif len(group) == 1:
            resolution = "singleton_project"
            reason = f"singleton project; structural score {scores[canonical.get('github_repo_id', '')]:.1f}; margin n/a"
        elif len(ordered) == 1:
            reason = f"only eligible base repository; structural score {scores[canonical.get('github_repo_id', '')]:.1f}; margin n/a"
        else:
            margin = scores[canonical.get("github_repo_id", "")] - scores[ordered[1].get("github_repo_id", "")]
            reason = f"highest structural score {scores[canonical.get('github_repo_id', '')]:.1f}; margin {margin:.1f}"
        if canonical is not None and len(ordered) > 1 and margin is not None and margin < ambiguity_margin and use_llm:
            group_key = stable_hash(*sorted(row.get("github_repo_id", "") for row in group))
            evidence_hash = stable_hash(json.dumps(group, sort_keys=True))
            try:
                if model_client is None:
                    model_client = OllamaClient(ollama_config)
                    profile_hash = decision_profile(model_client.model, PROMPT_VERSION, ollama_config, {"reconciliation_template": (prompt_config or {}).get("reconciliation_template", "")})
                cached = cache.get_compatible_decision(
                    "github.reconciliation", group_key, profile_hash, evidence_hash,
                    (legacy_runtime_decision_profile(model_client.model, PROMPT_VERSION, ollama_config, {"reconciliation_template": (prompt_config or {}).get("reconciliation_template", "")}),),
                ) if cache else None
                raw = cached or model_client.generate(_prompt(group, scores, prompt_config), RESPONSE_SCHEMA)
                selected = str(raw.get("canonical_repo_id", ""))
                if any(row.get("github_repo_id") == selected for row in base_candidates):
                    canonical = by_id[selected]
                    reason = str(raw.get("reason", "group LLM adjudication"))[:1000]
                    resolution = "llm_group_adjudication"
                    adjudicated = True
                    if cache and not cached:
                        cache.put_decision("github.reconciliation", group_key, profile_hash, evidence_hash, raw, utc_now())
            except Exception as exc:
                resolution = "ambiguous_deterministic_fallback"
                reason += f"; group adjudication failed: {str(exc)[:240]}"
                LOG.warning("GitHub group adjudication failed group=%s: %s", group_key, exc)
        elif canonical is not None and len(ordered) > 1 and margin is not None and margin < ambiguity_margin:
            resolution = "ambiguous_deterministic_fallback"
        canonical_id = canonical.get("github_repo_id", "") if canonical else ""
        project_id = f"github-project-{canonical_id}" if canonical_id else f"github-unresolved-{stable_hash(*sorted(row.get('github_repo_id', '') for row in group), length=12)}"
        if canonical is None or len(group) == 1:
            confidence = ""
        elif len(ordered) == 1:
            confidence = ".960"
        else:
            confidence = f"{margin_confidence(float(margin), len(group), adjudicated):.3f}"
        canonical_score = scores.get(canonical_id) if canonical_id else None
        breakdown = _score_breakdown(canonical, inbound[canonical_id]) if canonical else ""
        LOG.info("reconciled project family %d/%d members=%d canonical=%s resolution=%s", group_index, len(groups), len(group), canonical.get("full_name", canonical_id) if canonical else "missing", resolution)
        for row in group:
            if canonical is None:
                relationship = "base_repository_missing"
            elif len(group) == 1:
                # A singleton is still the observed base of its project family. Keep
                # singleton_project in canonical_resolution to record that no competing
                # repository was compared.
                relationship = "canonical"
            else:
                relationship = _relationship(row, canonical)
            reconciled_fields[row.get("github_repo_id", "")] = {
                "project_id": project_id, "canonical_repo_id": canonical_id,
                "canonical_full_name": canonical.get("full_name", "") if canonical else "",
                "repository_relationship": relationship,
                "canonical_confidence": confidence, "canonical_reason": reason,
                "canonical_resolution": resolution,
                "canonical_score": f"{canonical_score:.1f}" if canonical_score is not None else "",
                "canonical_margin": f"{margin:.1f}" if margin is not None else "",
                "canonical_score_breakdown": breakdown,
            }
    output_rows: list[dict[str, str]] = []
    for row in rows:
        value = dict(row)
        value.update(reconciled_fields.get(row.get("github_repo_id", ""), {}))
        output_rows.append(value)
    atomic_write_csv(output, output_rows, CATALOGUE_COLUMNS)
    LOG.info("GitHub reconciliation complete rows=%d output=%s", len(output_rows), output)
    return output_rows
