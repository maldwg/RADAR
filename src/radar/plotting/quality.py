"""Offline quality outcomes, criterion audits and explicitly separate joint-inclusion exports."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from radar.common.csvio import atomic_write_csv, read_csv
from radar.common.jsonio import write_json
from radar.common.provenance import audited_bundle, file_hash
from radar.common.text import slug
from radar.plotting.common import configure_style, load_frame


@audited_bundle
def _quality_figure(input_path: Path, out_dir: Path, label: str, config: dict[str, Any]) -> list[Path]:
    """Give each classifier's figure its own input/table provenance context."""
    from radar.plotting.generic import _frequency

    configure_style(config)
    frame = load_frame(input_path, ("entity_id", "quality_status"))
    return _frequency(frame, "quality_status", out_dir, "quality_outcomes",
        label + " · content quality outcomes (not accuracy)", list(config.get("formats", ["svg", "pdf"])), int(config.get("dpi", 160)))


@audited_bundle
def build_quality_reports(classification_inputs: dict[str, Path], out_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Report assessor outputs and coverage without inventing quality ground-truth accuracy."""
    summaries, criterion_rows, outputs = [], [], []
    policies = {row["quality_profile_hash"] for path in classification_inputs.values() for row in read_csv(path) if row.get("quality_profile_hash")}
    if len(policies) > 1:
        raise ValueError("quality comparisons require identical scientific quality criteria and thresholds")
    configurations = {row.get("quality_config_hash", "") for path in classification_inputs.values()
        for row in read_csv(path) if row.get("quality_rules_json")}
    if len(configurations) > 1:
        raise ValueError("quality comparisons require identical shared assessment configurations")
    labels = [slug(label) for label in classification_inputs]
    if len(labels) != len(set(labels)):
        raise ValueError("quality report labels must have distinct filesystem slugs")
    for label, source in classification_inputs.items():
        rows = read_csv(source)
        with source.open(encoding="utf-8", newline="") as handle:
            columns = list(csv.DictReader(handle).fieldnames or [])
        if not any(row.get("quality_rules_json") for row in rows) and (rows or "quality_status" not in columns):
            continue
        directory = out_dir / slug(label)
        counts = Counter(row.get("quality_status", "not_assessed") for row in rows)
        attempted = len(rows) - counts["not_assessed"] - counts["not_configured"]
        assessed = counts["pass"] + counts["fail"] + counts["unknown"]
        summaries.append({"classifier": label, "records": len(rows), "attempted": attempted, "assessed": assessed,
            "assessment_coverage": assessed / len(rows) if rows else "",
            "definitive_quality_coverage": (counts["pass"] + counts["fail"]) / len(rows) if rows else "",
            **{key: counts[key] for key in ("pass", "fail", "unknown", "error", "not_assessed", "not_configured")},
            "quality_accuracy": "", "quality_ground_truth": "not_supplied",
            "relevance_included": sum(row.get("relevance_decision") in {"accept", "accept_adjacent"} for row in rows),
            "joint_included": sum(row.get("quality_qualified_decision") in {"accept", "accept_adjacent"} for row in rows)})
        qualified, statuses = [], []
        for index, row in enumerate(rows):
            identity = next((row[key] for key in ("entity_id", "github_repo_id", "openalex_id", "paper_id") if row.get(key)), str(index))
            statuses.append({"entity_id": identity, "quality_status": row.get("quality_status", "not_assessed")})
            judgments = json.loads(row.get("quality_assessments_json") or "{}")
            supported = set(row.get("quality_supported_criteria", "").split(";"))
            for key, judgment in judgments.items():
                criterion_rows.append({"classifier": label, "entity_id": identity, "criterion": key,
                    "value": judgment["value"], "supported": str(key in supported).lower(),
                    "backend": row.get("quality_backend", ""), "profile_hash": row.get("quality_profile_hash", ""),
                    "configuration_hash": row.get("quality_config_hash", ""),
                    "assessment_json": json.dumps(judgment, sort_keys=True)})
            qualified.append({**row, "pipeline_decision": row.get("pipeline_decision", row.get("decision", "")),
                "decision": row.get("quality_qualified_decision", "uncertain"), "decision_view": "quality_qualified"})
        qualified_path = directory / "quality_qualified.csv"
        atomic_write_csv(qualified_path, qualified, list(dict.fromkeys([*columns, "pipeline_decision", "decision_view"])))
        status_path = directory / "quality_statuses.csv"
        atomic_write_csv(status_path, statuses, ["entity_id", "quality_status"])
        outputs.extend([qualified_path, status_path])
        audit_path = directory / "quality_qualified.audit.json"
        write_json(audit_path, {"schema_version": "1.0", "decision_view": "quality_qualified",
            "input": str(source), "input_sha256": file_hash(source), "output_sha256": file_hash(qualified_path),
            "quality_profile_hashes": sorted({row["quality_profile_hash"] for row in rows if row.get("quality_profile_hash")}),
            "quality_ground_truth": "not_supplied", "quality_accuracy_estimable": False})
        outputs.append(audit_path)
        if statuses:
            outputs.extend(_quality_figure(status_path, directory, label, config))
    if summaries:
        summary_path, criteria_path = out_dir / "summary.csv", out_dir / "criteria.csv"
        atomic_write_csv(summary_path, summaries, list(summaries[0]))
        atomic_write_csv(criteria_path, criterion_rows, ["classifier", "entity_id", "criterion", "value", "supported", "backend", "profile_hash", "configuration_hash", "assessment_json"])
        outputs.extend([summary_path, criteria_path])
        readme = out_dir / "README.rst"
        readme.write_text("Content quality assessment\n==========================\n\n"
            "summary.csv reports assessment coverage and quality outcomes, separately from relevance accuracy.\n"
            "criteria.csv retains each criterion, assessor and evidence/score audit.\n"
            "Unassessed keyword criteria are N/A; missing evidence is unknown.\n"
            "Quality pass rates are model outputs, not quality accuracy: no independent quality ground truth was supplied.\n"
            "quality_qualified.csv applies the shared threshold to that classifier's own relevance and quality outputs.\n"
            "It preserves all attempted records and does not replace original predictions or reference labels.\n", encoding="utf-8")
        outputs.append(readme)
    return outputs
