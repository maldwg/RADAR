"""Load explicit, versioned quality prompts and assessment parameters from TOML."""
from __future__ import annotations

import json
import math
from pathlib import Path
from string import Template
import tomllib
from typing import Any

from radar.common.provenance import file_hash


def _template(value: Any, placeholders: set[str], label: str) -> str:
    """Require nonempty templates with exactly their supported input placeholders."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"quality {label} must be a nonempty template")
    template = Template(value)
    if not template.is_valid() or set(template.get_identifiers()) != placeholders:
        raise ValueError(f"quality {label} requires exactly these placeholders: {', '.join(sorted(placeholders))}")
    return value


def load_quality_config(path: Path) -> dict[str, Any]:
    """Validate every configurable quality parameter without embedded experimental defaults."""
    digest = file_hash(path)
    with path.open('rb') as handle:
        raw = tomllib.load(handle)
    if raw.get('schema_version') != '1.0' or set(raw) != {'schema_version', 'evidence', 'nli', 'prompts'}:
        raise ValueError('quality configuration requires schema_version 1.0, evidence, nli and prompts')
    evidence, nli, prompts = (raw[key] for key in ('evidence', 'nli', 'prompts'))
    if not isinstance(evidence, dict) or set(evidence) != {'max_quotation_passages', 'max_block_passages'}:
        raise ValueError('quality evidence requires max_quotation_passages and max_block_passages')
    for key, minimum in (('max_quotation_passages', 1), ('max_block_passages', 0)):
        if type(evidence[key]) is not int or evidence[key] < minimum:
            raise ValueError(f'quality evidence.{key} must be an integer of at least {minimum}')
    if not isinstance(nli, dict) or set(nli) != {'entailment_threshold', 'label_margin', 'hypothesis_template', 'premise_template'}:
        raise ValueError('quality nli requires cutoffs and hypothesis/premise templates')
    for key in ('entailment_threshold', 'label_margin'):
        value = nli[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1 or key == 'entailment_threshold' and value == 0:
            raise ValueError(f'quality nli.{key} must lie in its probability range')
    _template(nli['hypothesis_template'], {'criterion'}, 'nli.hypothesis_template')
    _template(nli['premise_template'], {'source', 'text'}, 'nli.premise_template')
    if not isinstance(prompts, dict) or set(prompts) != {'instructions', 'quotations', 'blocks'}:
        raise ValueError('quality prompts requires instructions, quotations and blocks')
    if not isinstance(prompts['instructions'], str) or not prompts['instructions'].strip():
        raise ValueError('quality prompts.instructions must be nonempty')
    for key in ('quotations', 'blocks'):
        _template(prompts[key], {'instructions', 'criteria'}, 'prompts.' + key)
    if file_hash(path) != digest:
        raise ValueError('quality configuration changed while loading')
    return {**raw, '_sha256': digest}


def quality_for_study(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve the explicit quality reference of a study, including standalone review APIs."""
    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    reference = manifest.get('quality')
    if manifest.get('schema_version') != '2.0' or not isinstance(reference, str) or not reference.strip():
        raise ValueError('study manifest requires an explicit quality configuration path')
    path = (manifest_path.parent / reference).resolve()
    return path, load_quality_config(path)


def assessment_config(rules: dict[str, Any]) -> dict[str, Any]:
    """Require the shared assessment configuration when quality criteria are enabled."""
    config = rules.get('assessment')
    if not isinstance(config, dict) or not config.get('_sha256'):
        raise ValueError('quality criteria require an explicitly loaded assessment configuration')
    return config


def render_quality_prompt(criteria: dict[str, str], config: dict[str, Any], evidence_format: str) -> str:
    """Render shared assessor instructions without substituting criterion text recursively."""
    if not criteria:
        return ''
    return Template(config['prompts'][evidence_format]).substitute(
        instructions=config['prompts']['instructions'], criteria=json.dumps(criteria, ensure_ascii=False, sort_keys=True))
