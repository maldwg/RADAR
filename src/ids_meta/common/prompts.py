from __future__ import annotations

import re
from typing import Any


def description_block(
    config: dict[str, Any], key: str, ordered_values: tuple[str, ...],
) -> str:
    """Render a complete configured taxonomy table or fail on missing descriptions."""
    configured = config.get(key)
    if not isinstance(configured, dict):
        raise ValueError(f"prompt configuration {key!r} must be a table")
    missing = [value for value in ordered_values if not str(configured.get(value, "")).strip()]
    if missing:
        raise ValueError(f"prompt configuration {key!r} is missing descriptions for: {', '.join(missing)}")
    return "\n".join(
        f"- {value}: {str(configured[value]).strip()}"
        for value in ordered_values
    )


def render_prompt(
    config: dict[str, Any] | None, blocks: dict[str, str], evidence: str, template_key: str = "template",
) -> str:
    """Render one mandatory configured template using validated literal placeholders."""
    settings = config
    if not isinstance(settings, dict):
        raise ValueError("source prompt configuration must be a table")
    template = str(settings.get(template_key, "")).strip()
    if not template:
        raise ValueError(f"prompt configuration requires a non-empty {template_key}")
    if "${evidence}" not in template:
        raise ValueError(f"prompt configuration {template_key} must contain ${{evidence}}")
    values = {**blocks, "evidence": evidence}
    screening = settings.get("screening", {})
    if not isinstance(screening, dict):
        raise ValueError("prompt screening must be a table")
    for name, text in screening.get("eligibility", {}).items():
        values[f"eligibility_{name}"] = text
    values["eligibility"] = "\n\n".join(screening.get("eligibility", {}).values())
    unknown = sorted(set(re.findall(r"\$\{([^}]+)\}", template)) - set(values))
    if unknown:
        raise ValueError(f"unknown classifier prompt placeholders: {', '.join(unknown)}")
    # Substitute only the template, so literal placeholders in evidence stay data.
    return re.sub(r"\$\{([^}]+)\}", lambda match: values[match.group(1)], template)
