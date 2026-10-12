"""Independent numerical and integrity checks for figures and reporting stores."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import tempfile
import zlib
from collections import Counter
from functools import wraps
from pathlib import Path
from statistics import NormalDist
from typing import Any

from ids_meta.common.csvio import read_csv, configure_csv_limit
from ids_meta.common.provenance import file_hash
from ids_meta.common.sources import content_hash


def guarded_verification(function: Any) -> Any:
    """Turn malformed or missing audit data into an explicit failed verification."""
    @wraps(function)
    def wrapped(path: Path) -> dict[str, Any]:
        """Preserve diagnostics without crashing on incomplete reporting bundles."""
        try:
            return function(path)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, csv.Error) as exc:
            return {'path': str(path), 'ok': False, 'errors': [f'invalid or incomplete audit: {exc}']}
    return wrapped


def _values(value: Any) -> set[str]:
    """Interpret audit values independently of the evaluator's aggregation code."""
    items = value if isinstance(value, list) else str(value or "").replace(",", ";").split(";")
    return {str(item).strip().casefold() for item in items if str(item).strip()}


def check_evaluation(metrics: dict[str, Any], instances: list[dict[str, str]], fields: list[dict[str, str]] | None = None) -> list[str]:
    """Recompute confusion counts and exact taxonomy checks from individual instances."""
    errors: list[str] = []
    counts = {name: Counter(tp=0, fp=0, tn=0, fn=0) for name in ("operational", "strict", "selective")}
    comparisons: Counter[str] = Counter()
    matches: Counter[str] = Counter()
    label_counts: dict[str, Counter] = {}
    expected_labels: dict[str, set[str]] = {}
    for row in instances:
        expected, actual = row["expected_disposition"], row["actual_disposition"]
        if expected in {"include", "exclude"}:
            positive = expected == "include"
            included = actual == "include"
            counts["operational"]["tp" if positive and included else "fn" if positive else "fp" if included else "tn"] += 1
            counts["strict"]["tp" if positive and included else "fn" if positive else "tn" if actual == "exclude" else "fp"] += 1
            if actual in {"include", "exclude"}:
                counts["selective"]["tp" if positive and included else "fn" if positive else "fp" if included else "tn"] += 1
        expected_fields, actual_fields = json.loads(row["expected_json"]), json.loads(row["actual_json"])
        checks = json.loads(row["field_results_json"])
        for field, correct in checks.items():
            computed = _values(expected_fields.get(field)) == _values(actual_fields.get(field))
            if type(correct) is not bool or correct != computed:
                errors.append(f"incorrect field check: {row['entity_id']}.{field}")
            comparisons[field] += 1
            matches[field] += int(computed)
            truth, prediction = _values(expected_fields.get(field)), _values(actual_fields.get(field))
            expected_labels.setdefault(field, set()).update(truth)
            labels = label_counts.setdefault(field, Counter())
            for label in truth | prediction:
                labels[(label, 'tp' if label in truth & prediction else 'fn' if label in truth else 'fp')] += 1
    if metrics.get("rated_entities") != len(instances):
        errors.append("rated_entities differs from the instance audit")
    if metrics.get("status_counts") != dict(Counter(row["status"] for row in instances)):
        errors.append("status_counts differs from the instance audit")
    statuses = Counter(row['status'] for row in instances)
    total = len(instances)
    positive = [r for r in instances if r['expected_disposition'] == 'include']
    present = sum(r['classification_present'] == 'true' for r in instances)
    positive_present = sum(r['classification_present'] == 'true' for r in positive)
    artifact = metrics.get('evaluation_unit') == 'artifact'
    abstention = sum(statuses[k] for k in ('uncertain', 'error', 'survey_reference_classification_uncertain', 'survey_reference_classification_error') if artifact) if artifact else sum(statuses[k] for k in ('uncertain', 'error', 'not_retrieved', 'not_classified', 'unavailable'))
    expected_rates = {
        'decision_accuracy': sum(r['expected_disposition'] == r['actual_disposition'] for r in instances) / total if total else 0,
        'classification_coverage': (positive_present / len(positive) if positive else 0) if artifact else (present / total if total else 0),
        'definitive_coverage': statuses['classified'] / total if total else 0,
        'abstention_rate': abstention / total if total else 0,
        'classification_field_accuracy': sum(matches.values()) / sum(comparisons.values()) if sum(comparisons.values()) else None,
    }
    if artifact:
        expected_rates['unresolved_artifact_rate'] = 1 - expected_rates['classification_coverage']
    elif metrics.get('retrieval_coverage') is not None:
        expected_rates['retrieval_coverage'] = 1 - statuses['not_retrieved'] / total if total else 0
    for key, value in expected_rates.items():
        saved = metrics.get(key)
        if saved is None and value is None:
            continue
        if saved is None or value is None or not math.isclose(float(saved), value, abs_tol=1e-10):
            errors.append(f'{key}: incorrect audited denominator or count')
    if metrics.get('classification_field_comparisons') != sum(comparisons.values()):
        errors.append('classification_field_comparisons differs from individual field checks')
    for field, saved in metrics.get('taxonomy_field_metrics', {}).items():
        if saved['comparisons'] != comparisons[field] or saved['exact_matches'] != matches[field]:
            errors.append(f'taxonomy_field_metrics.{field}: counts differ from instances')
        for label, result in saved.get('per_label', {}).items():
            counter = label_counts.get(field, Counter())
            tp, fp, fn = (counter[(label, key)] for key in ('tp', 'fp', 'fn'))
            for key, value in [('tp', tp), ('fp', fp), ('fn', fn),
                ('precision', tp / (tp + fp) if tp + fp else 0),
                ('recall', tp / (tp + fn) if tp + fn else 0),
                ('f1', 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)]:
                if result.get(key) is None or not math.isclose(float(result[key]), value, abs_tol=1e-10):
                    errors.append(f'taxonomy_field_metrics.{field}.{label}.{key}: incorrect score')
        if set(saved.get('per_label', {})) != expected_labels.get(field, set()):
            errors.append(f'taxonomy_field_metrics.{field}: expected label inventory differs')
        exact = matches[field] / comparisons[field] if comparisons[field] else None
        macro = sum(r['f1'] for r in saved.get('per_label', {}).values()) / len(saved['per_label']) if saved.get('per_label') and saved.get('supported') else None
        multilabel = field in metrics.get('source_contract', {}).get('multilabel', []) or any(isinstance(json.loads(r['expected_json']).get(field), list) for r in instances)
        counter = label_counts.get(field, Counter())
        tp, fp, fn = (sum(n for (label, outcome), n in counter.items() if outcome == key) for key in ('tp', 'fp', 'fn'))
        label_f1 = (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0) if multilabel else None
        for key, value in [('exact_accuracy', exact), ('macro_f1', macro), ('label_f1', label_f1)]:
            if value is None and saved.get(key) is None:
                continue
            if value is None or saved.get(key) is None or not math.isclose(float(saved[key]), value, abs_tol=1e-10):
                errors.append(f'taxonomy_field_metrics.{field}.{key}: incorrect score')
    macros = [r['macro_f1'] for r in metrics.get('taxonomy_field_metrics', {}).values() if r.get('macro_f1') is not None]
    if 'classification_field_macro_f1' in metrics:
        expected_macro = sum(macros) / len(macros) if macros else None
        saved_macro = metrics['classification_field_macro_f1']
        if (saved_macro is None) != (expected_macro is None) or (saved_macro is not None and not math.isclose(saved_macro, expected_macro, abs_tol=1e-10)):
            errors.append('classification_field_macro_f1: incorrect field aggregation')
    for view, result in counts.items():
        saved = metrics.get(view, {})
        for key, value in result.items():
            if saved.get(key) != value:
                errors.append(f"{view}.{key}: stored={saved.get(key)}, recomputed={value}")
        tp, fp, tn, fn = (result[k] for k in ("tp", "fp", "tn", "fn"))
        ratios = {
            "detection_rate": (tp, tp + fn), "false_negative_rate": (fn, tp + fn),
            "false_positive_rate": (fp, fp + tn), "false_discovery_rate": (fp, tp + fp),
            "precision": (tp, tp + fp), "accuracy": (tp + tn, tp + fp + tn + fn),
            "specificity": (tn, tn + fp), "negative_predictive_value": (tn, tn + fn),
        }
        for key, (numerator, denominator) in ratios.items():
            if key in saved and saved[key] is not None:
                value = numerator / denominator if denominator else 0.0
                if not math.isclose(float(saved[key]), value, abs_tol=1e-10):
                    errors.append(f"{view}.{key}: incorrect numerator/denominator")
        aliases = {'DR': 'detection_rate', 'recall': 'detection_rate', 'sensitivity': 'detection_rate', 'FNR': 'false_negative_rate', 'FPR': 'false_positive_rate', 'FDR': 'false_discovery_rate', 'ACC': 'accuracy', 'PREC': 'precision'}
        for alias, origin in aliases.items():
            if alias in saved and saved[alias] is not None:
                numerator, denominator = ratios[origin]
                if not math.isclose(float(saved[alias]), numerator / denominator if denominator else 0, abs_tol=1e-10):
                    errors.append(f'{view}.{alias}: incorrect ratio')
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        balanced = ((tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0)) / 2 if tp + fp + tn + fn else 0
        for key, value in [('f1', f1), ('balanced_accuracy', balanced)]:
            if saved.get(key) is not None and not math.isclose(float(saved[key]), value, abs_tol=1e-10):
                errors.append(f'{view}.{key}: incorrect score')
    for row in fields or []:
        field = row["field"]
        if int(row["comparisons"]) != comparisons[field] or int(row["exact_matches"]) != matches[field]:
            errors.append(f"field_metrics.{field}: counts differ from individual checks")
        value = matches[field] / comparisons[field] if comparisons[field] else None
        if value is None:
            if str(row.get("exact_accuracy", "")) not in {"", "None"}:
                errors.append(f"field_metrics.{field}: empty denominator must be N/A")
        elif not math.isclose(float(row["exact_accuracy"]), value, abs_tol=1e-10):
            errors.append(f"field_metrics.{field}: incorrect exact accuracy")
    op = counts['operational']
    intervals = {'detection_rate': (op['tp'], op['tp'] + op['fn']),
        'false_positive_rate': (op['fp'], op['fp'] + op['tn']),
        'false_discovery_rate': (op['fp'], op['tp'] + op['fp']),
        'accuracy': (op['tp'] + op['tn'], sum(op.values())), 'precision': (op['tp'], op['tp'] + op['fp']),
        'definitive_coverage': (statuses['classified'], total), 'abstention_rate': (abstention, total)}
    for key, saved in metrics.get('confidence_intervals', {}).items():
        successes, size = intervals[key]
        if saved is None:
            if size and key not in metrics.get('not_estimable_metrics', []):
                errors.append(f'confidence_intervals.{key}: missing estimable interval')
            continue
        if not size:
            errors.append(f'confidence_intervals.{key}: empty denominator')
            continue
        z2 = NormalDist().inv_cdf((1 + metrics['confidence_level']) / 2) ** 2
        center = (successes + z2 / 2) / (size + z2)
        radius = math.sqrt(z2 * (successes * (size - successes) / size + z2 / 4)) / (size + z2)
        if len(saved) != 2 or any(not math.isclose(a, b, abs_tol=1e-10) for a, b in zip(saved, [max(0, center - radius), min(1, center + radius)])):
            errors.append(f'confidence_intervals.{key}: incorrect Wilson interval')
    return errors


def _snapshot(item: dict[str, Any], manifest_path: Path) -> Path:
    """Resolve frozen objects after moving a bundle without requiring original files."""
    original = Path(item["snapshot"])
    if original.is_file():
        return original
    for parent in [manifest_path.parent, *manifest_path.parents]:
        candidate = parent / ".provenance" / "objects" / item["sha256"]
        if candidate.is_file():
            return candidate
    return original


def _artifact(item: dict[str, Any], manifest_path: Path) -> Path:
    """Prefer colocated figure artifacts when auditing a relocated reporting bundle."""
    local = manifest_path.parent / Path(item["path"]).name
    return local if local.is_file() else Path(item["path"])


def check_evaluation_table(name: str, table: list[dict[str, str]], metrics: dict[str, Any], instances: list[dict[str, str]]) -> list[str]:
    """Validate plotted values against frozen instance-level outcomes, independently."""
    errors = check_evaluation(metrics, instances)
    counts = Counter((r['expected_disposition'], r['actual_disposition']) for r in instances)
    positives = [r for r in instances if r['expected_disposition'] == 'include']
    recovered = [r for r in positives if r['actual_disposition'] == 'include']
    fields = metrics.get('taxonomy_field_metrics', {})
    wanted_keys, actual_keys = None, None
    if name == '01_decision_views':
        keys = ['DR', 'FNR'] if metrics.get('benchmark_scope') == 'positive_reference_set' else ['DR', 'PREC', 'ACC', 'f1', 'FPR', 'FDR', 'FNR']
        wanted_keys = {(view, key) for view in ('operational', 'strict', 'selective') for key in keys}
        actual_keys = [(r['view'].lower(), r['metric_key']) for r in table]
    elif name == '02_decision_confusion':
        wanted_keys = {r['expected_disposition'] for r in instances}
        actual_keys = [r['expected'] for r in table]
        wanted_columns = {r['actual_disposition'] for r in instances}
        if table and any(set(r) - {'expected'} != wanted_columns for r in table):
            errors.append('decision confusion has a missing or extra outcome column')
    elif name in {'03_taxonomy_fields', '06_taxonomy_accuracy', '07_recovered_taxonomy_accuracy'}:
        wanted_keys = {f for f, values in fields.items() if name != '03_taxonomy_fields' or values['comparisons'] > 0}
        actual_keys = [r['field'] for r in table]
    elif name == '04_taxonomy_categories':
        wanted_keys = {(field, label) for field, values in fields.items() if values['supported'] for label in values['per_label']}
        actual_keys = [(r['field'], r['label']) for r in table]
    elif name == '05_coverage_rates':
        keys = ['classification_coverage', 'definitive_coverage', 'abstention_rate', 'unresolved_artifact_rate'] if metrics.get('evaluation_unit') == 'artifact' else ['retrieval_coverage', 'classification_coverage', 'definitive_coverage', 'abstention_rate']
        wanted_keys = {key for key in keys if metrics.get(key) is not None}
        actual_keys = [r['key'] for r in table]
    elif name == '05_status_counts':
        wanted_keys = {r['status'] for r in instances}
        actual_keys = [r['status_key'] for r in table]
    if wanted_keys is not None and (set(actual_keys) != wanted_keys or len(actual_keys) != len(set(actual_keys))):
        errors.append(f'{name}: missing, extra or duplicated plotted categories')
    if name == '08_reference_uncertainty':
        outcomes = Counter()
        for r in instances:
            if r['classification_present'] == 'true':
                disposition, error = r['actual_disposition'], r['status'] == 'error'
            elif r.get('survey_reference_classification_present') == 'true':
                disposition, error = r['survey_reference_actual_disposition'], r['status'] == 'survey_reference_classification_error'
            else:
                outcomes['not_attempted'] += 1
                continue
            outcomes['error' if error else {'include': 'accept', 'exclude': 'exclude', 'uncertain': 'uncertain'}.get(disposition, 'unavailable')] += 1
        attempted = len(instances) - outcomes['not_attempted']
        expected_rows = [{'rated_entities': len(instances), 'attempted': attempted,
            **{k: outcomes[k] for k in ('accept', 'exclude', 'uncertain', 'error', 'unavailable', 'not_attempted')},
            'value': outcomes['uncertain'] / attempted if attempted else None}]
    else:
        expected_rows = []
        for row in table:
            expected: dict[str, Any] = {}
            if name == '01_decision_views':
                expected['value'] = metrics[row['view'].lower()][row['metric_key']]
            elif name == '02_decision_confusion':
                expected = {key: counts[(row['expected'], key)] for key in row if key != 'expected'}
            elif name in {'03_taxonomy_fields', '06_taxonomy_accuracy'}:
                saved = fields[row['field']]
                expected = {k: saved[k] for k in ('comparisons', 'exact_matches', 'exact_accuracy', 'label_f1', 'macro_f1') if k in row}
                if name == '06_taxonomy_accuracy':
                    expected.update(value=saved['exact_accuracy'], percentage=saved['exact_accuracy'] * 100 if saved['exact_accuracy'] is not None else None)
            elif name == '04_taxonomy_categories':
                expected = fields[row['field']]['per_label'][row['label']]
            elif name == '05_coverage_rates':
                expected['value'] = metrics[row['key']]
            elif name == '05_status_counts':
                expected['count'] = sum(r['status'] == row['status_key'] for r in instances)
            elif name == '07_recovered_taxonomy_accuracy':
                checks = [json.loads(r['field_results_json'])[row['field']] for r in recovered if row['field'] in json.loads(r['field_results_json'])]
                size, correct = len(checks), sum(checks)
                value = correct / size if size else None
                expected = {'comparisons': size, 'exact_matches': correct, 'value': value,
                    'percentage': value * 100 if value is not None else None,
                    'expected_positive_entities': len(positives), 'recovered': len(recovered),
                    'missed': len(positives) - len(recovered), 'recovery_rate': len(recovered) / len(positives) if positives else None,
                    'inventory_comparisons': fields[row['field']]['comparisons']}
            else:
                errors.append(f'unsupported evaluation table: {name}')
            expected_rows.append(expected)
    if len(expected_rows) != len(table):
        errors.append(f'{name}: row count differs')
    for row, expected in zip(table, expected_rows):
        for key, value in expected.items():
            saved = row.get(key, '')
            if value is None:
                valid = saved in {'', 'None', 'nan'}
            else:
                try:
                    valid = math.isclose(float(saved), float(value), abs_tol=1e-10)
                except (ValueError, TypeError):
                    valid = False
            if not valid:
                errors.append(f'{name}.{key}: plotted value differs from audited instances')
    return errors


@guarded_verification
def verify_manifest(path: Path) -> dict[str, Any]:
    """Check frozen inputs, current numeric tables and rendered artifact identities."""
    manifest = json.loads(path.read_text())
    errors, recomputed = [], 0
    if manifest.get("schema_version") != "1.0":
        return {"path": str(path), "ok": False, "errors": ["unsupported manifest version"]}
    inputs = {item["sha256"]: item for item in manifest["inputs"]}
    renderer = manifest.get('definition', {}).get('renderer_snapshot')
    if renderer and (not _snapshot(renderer, path).is_file() or file_hash(_snapshot(renderer, path)) != renderer['sha256']):
        errors.append('renderer snapshot is missing or changed')
    for kind in ("inputs", "tables", "artifacts"):
        for item in manifest[kind]:
            target = _artifact(item, path) if kind == "artifacts" else _snapshot(item, path)
            if not target.is_file() or file_hash(target) != item["sha256"]:
                errors.append(f"{kind}: missing or changed bytes for {item['path']}")
            if kind == "tables":
                current = Path(item["path"])
                local = path.parent / "data" / current.name
                current = local if local.is_file() else current
                if current.is_file() and file_hash(current) != item["sha256"]:
                    errors.append(f"table changed after rendering: {current}")
                recipe = item.get("recipe")
                if recipe:
                    source = inputs.get(recipe.get("input_sha256"))
                    if source is None or not target.is_file():
                        errors.append("recipe lacks its frozen source")
                        continue
                    input_path = _snapshot(source, path)
                    if not input_path.is_file():
                        continue
                    rows, table = read_csv(input_path), read_csv(target)
                    if recipe['kind'] == 'evaluation_table':
                        metadata = inputs.get(recipe['metrics_sha256'])
                        if metadata is None:
                            errors.append('evaluation recipe lacks frozen metrics')
                            continue
                        errors.extend(check_evaluation_table(recipe['table'], table, json.loads(_snapshot(metadata, path).read_text()), rows))
                        recomputed += 1
                    elif recipe["kind"] == "value_counts":
                        field = recipe["column"]
                        expanded = [(row.get(recipe['identity'], ''), value) for row in rows for value in
                            (sorted(set(v.strip() for v in str(row.get(field, '')).split(';') if v.strip())) or ['unknown'] if recipe.get('multilabel') else [row.get(field) or 'unknown'])]
                        expected = Counter(value for _, value in expanded)
                        actual = {row[field]: int(row["count"]) for row in table}
                        wanted = dict(expected.most_common(recipe.get("top_n")))
                        if actual != wanted or len(actual) != len(table):
                            errors.append(f"recomputed counts differ for {field}")
                        members = recipe.get("members", {})
                        if members:
                            id_field = recipe["identity"]
                            for value, identifiers in members.items():
                                wanted_ids = sorted(key for key, label in expanded if label == value)
                                if sorted(identifiers) != wanted_ids:
                                    errors.append(f"incorrect contribution mapping for {field}={value}")
                        recomputed += 1
                    elif recipe['kind'] == 'stages':
                        def members(digest: str | None) -> list[dict[str, str]] | None:
                            """Read one declared stage input from the same frozen bundle."""
                            if digest is None:
                                return None
                            if digest not in inputs:
                                raise ValueError('stage input is absent from the manifest')
                            return read_csv(_snapshot(inputs[digest], path))
                        candidates, exports = members(recipe.get('candidate_sha256')), members(recipe.get('export_sha256'))
                        eligible = None
                        if candidates is not None and all(row.get('prefilter_band') in {'candidate', 'borderline', 'reject'} for row in candidates):
                            eligible = {r['entity_id'] for r in candidates if r['prefilter_band'] in {'candidate', 'borderline'}}
                        expected = {
                            'candidate_selection': {r['entity_id'] for r in candidates} if candidates is not None else None,
                            'enrichment_prefilter': eligible, 'classification': {r['entity_id'] for r in rows} if rows and all(r.get('decision') for r in rows) else None,
                            'inclusion': {r['entity_id'] for r in rows if r.get('decision') in {'accept', 'accept_adjacent'} and not r.get('error')} if rows and all(r.get('decision') for r in rows) else None,
                            'export': {r['entity_id'] for r in exports} if exports is not None else None,
                        }
                        if {r['stage'] for r in table} != set(expected) or len(table) != len(expected):
                            errors.append('missing, extra or duplicate pipeline stage')
                        for row in table:
                            wanted = expected.get(row['stage'])
                            if wanted is None:
                                if row['basis'] != 'not_observed' or row['count']:
                                    errors.append('unobserved stage represented as an observed count')
                            elif row['basis'] != 'observed' or int(float(row['count'])) != len(wanted) or set(row['entity_ids'].split(';')) - {''} != wanted:
                                errors.append(f"stage membership/count mismatch: {row['stage']}")
                        recomputed += 1
                    else:
                        errors.append(f"unknown replay recipe: {recipe['kind']}")
    return {"path": str(path), "ok": not errors, "errors": errors, "recomputed_tables": recomputed,
        "verification": "recomputed" if recomputed else "integrity_only", "lineage": manifest.get("lineage")}


@guarded_verification
def verify_directory(directory: Path) -> dict[str, Any]:
    """Verify a figure bundle and independently recompute any evaluation audits."""
    figures = [verify_manifest(path) for path in sorted(directory.rglob("*.manifest.json"))]
    errors: list[str] = []
    evaluations = []
    for path in sorted(directory.rglob("metrics.json")):
        audit = path.parent / "instances.csv"
        if not audit.is_file():
            errors.append(f"evaluation missing instance audit: {path}")
            continue
        fields = path.parent / "field_metrics.csv"
        result = check_evaluation(json.loads(path.read_text()), read_csv(audit), read_csv(fields) if fields.is_file() else None)
        evaluations.append({"path": str(path), "ok": not result, "errors": result})
    if not figures and not evaluations:
        errors.append("no figure manifests or evaluation audits found")
    return {"ok": not errors and all(r["ok"] for r in figures + evaluations), "figures": figures, "evaluations": evaluations, "errors": errors}


@guarded_verification
def verify_database(path: Path) -> dict[str, Any]:
    """Read-only verification of foreign keys, compressed snapshots and metric audits."""
    errors, evaluations = [], []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            errors.append("SQLite integrity check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            errors.append("reporting database has broken foreign keys")
        if db.execute("SELECT version FROM schema_meta").fetchall() != [(1,)]:
            errors.append("unsupported reporting database schema version")
        for provider, kind, external, entity in db.execute('SELECT provider,entity_kind,external_id,entity_id FROM identifiers'):
            if entity != content_hash([provider, kind, external]):
                errors.append(f'entity identity differs from provider-scoped identifier: {external}')
        snapshot_digests = {}
        for key, digest, compressed in db.execute("SELECT snapshot_id,sha256,contents FROM snapshots"):
            try:
                data = zlib.decompress(compressed)
                actual = hashlib.sha256(data).hexdigest()
                snapshot_digests[key] = actual
                if actual != digest:
                    errors.append(f"corrupted snapshot: {key}")
            except zlib.error:
                errors.append(f"invalid compressed snapshot: {key}")
        row_hashes = {}
        configure_csv_limit()
        for snapshot, entity, payload in db.execute('SELECT snapshot_id,entity_id,payload_json FROM source_records'):
            if snapshot not in row_hashes:
                try:
                    compressed = db.execute('SELECT contents FROM snapshots WHERE snapshot_id=?', (snapshot,)).fetchone()[0]
                    data = zlib.decompress(compressed)
                    rows = csv.DictReader(io.StringIO(data.decode('utf-8')))
                    row_hashes[snapshot] = {content_hash(row) for row in rows}
                except (KeyError, UnicodeDecodeError, zlib.error):
                    row_hashes[snapshot] = set()
            if content_hash(json.loads(payload)) not in row_hashes[snapshot]:
                errors.append(f'source record differs from frozen CSV: {entity}')
        for key, snapshot, entity, outcome, payload in db.execute('SELECT decision_id,snapshot_id,entity_id,outcome,payload_json FROM decisions'):
            row = db.execute('SELECT payload_json FROM source_records WHERE snapshot_id=? AND entity_id=?', (snapshot, entity)).fetchone()
            if not row or json.loads(row[0]) != json.loads(payload) or json.loads(payload).get('decision', '') != outcome:
                errors.append(f'decision differs from source snapshot: {key}')
        for key, metrics in db.execute("SELECT evaluation_id,metrics_json FROM evaluations"):
            metrics = json.loads(metrics)
            rows = [json.loads(r[0]) for r in db.execute("SELECT instance_json FROM reference_matches WHERE evaluation_id=?", (key,))]
            result = check_evaluation(metrics, rows)
            run_id = db.execute('SELECT run_id FROM evaluations WHERE evaluation_id=?', (key,)).fetchone()[0]
            provenance = db.execute('SELECT provenance_status FROM runs WHERE run_id=?', (run_id,)).fetchone()[0]
            if provenance == 'frozen_inputs':
                actual_count = db.execute('SELECT count(*) FROM decisions WHERE run_id=?', (run_id,)).fetchone()[0]
                if actual_count != metrics['actual_entities']:
                    result.append('actual_entities differs from the frozen decision population')
            evaluations.append({"evaluation_id": key, "ok": not result, "errors": result})
            reference_id = db.execute('SELECT reference_id FROM evaluations WHERE evaluation_id=?', (key,)).fetchone()[0]
            for reference_key, decision_id, instance in db.execute('SELECT reference_key,decision_id,instance_json FROM reference_matches WHERE evaluation_id=?', (key,)):
                instance = json.loads(instance)
                expected = db.execute('SELECT expected_json FROM reference_entities WHERE reference_id=? AND reference_key=?', (reference_id, reference_key)).fetchone()
                if reference_key != content_hash([instance['match_field'], instance['match_value']]):
                    errors.append(f'reference identity differs from selector: {key}')
                if not expected or json.loads(expected[0]) != json.loads(instance['expected_json']):
                    errors.append(f'reference mapping differs from expected reference: {key}')
                if decision_id:
                    decision = db.execute('SELECT payload_json FROM decisions WHERE decision_id=?', (decision_id,)).fetchone()
                    if not decision or json.loads(decision[0]) != json.loads(instance['actual_json']):
                        errors.append(f'reference mapping differs from classifier decision: {key}')
        for decision, expected_digest, payload in db.execute("SELECT decision_id,sha256,payload_json FROM evidence"):
            if content_hash(json.loads(payload)) != expected_digest:
                errors.append(f"changed evidence for decision {decision}")
        frozen = {digest: key for key, digest in snapshot_digests.items()}
        with tempfile.TemporaryDirectory(prefix='ids-meta-verify-') as temporary:
            root = Path(temporary)
            for key, manifest_json in db.execute('SELECT figure_id,manifest_json FROM figure_manifests'):
                manifest = json.loads(manifest_json)
                renderer = manifest.get('definition', {}).get('renderer_snapshot')
                missing = False
                for item in [*manifest['inputs'], *manifest['tables'], *manifest['artifacts'], *([renderer] if renderer else [])]:
                    digest = item['sha256']
                    if digest not in frozen:
                        errors.append(f'figure object not frozen in database: {key}')
                        missing = True
                        continue
                    path = root / digest
                    if not path.exists():
                        compressed = db.execute('SELECT contents FROM snapshots WHERE snapshot_id=?', (frozen[digest],)).fetchone()[0]
                        path.write_bytes(zlib.decompress(compressed))
                    item.update(path=str(path), snapshot=str(path))
                if not missing:
                    path = root / 'figure.manifest.json'
                    path.write_text(json.dumps(manifest))
                    result = verify_manifest(path)
                    errors.extend(f'figure {key}: {e}' for e in result['errors'])
    return {"ok": not errors and all(r["ok"] for r in evaluations), "evaluations": evaluations, "errors": errors}
