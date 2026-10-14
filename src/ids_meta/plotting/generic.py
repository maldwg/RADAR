"""Capability-driven reporting for local and future API provider adapters."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.common.provenance import audited_bundle, file_hash, register_input
from ids_meta.common.sources import LocalSourceAdapter, SourceContract, load_source_contract
from ids_meta.plotting.common import configure_style, load_frame, save_figure, write_table, taxonomy_field_label


def _write_records(path: Path, rows: list[dict[str, Any]], contract: SourceContract, extra_columns: tuple[str, ...] = ()) -> None:
    """Export lossless normalized records, retaining a usable schema for empty inputs."""
    columns = list(dict.fromkeys([contract.identity, contract.label, 'entity_id', 'provider', 'entity_kind', 'decision',
        *contract.fields, *extra_columns, *[key for row in rows for key in row]]))
    atomic_write_csv(path, rows, columns)


def _frequency(frame: pd.DataFrame, field: str, directory: Path, stem: str, title: str, formats: list[str], dpi: int, multilabel: bool = False, top_n: int = 15) -> list[Path]:
    """Plot a raw scalar count with exact contributing entity IDs and a replay recipe."""
    expanded = [(str(row['entity_id']), value) for row in frame.to_dict('records') for value in
        (sorted(set(v.strip() for v in str(row[field]).split(';') if v.strip())) or ['unknown'] if multilabel else [row[field] or 'unknown'])]
    counts = pd.DataFrame(Counter(value for _, value in expanded).most_common(), columns=[field, 'count'])
    source = Path(frame.attrs['source_path'])
    counts.attrs['verification_recipe'] = {
        'kind': 'value_counts', 'column': field, 'input_sha256': file_hash(source), 'identity': 'entity_id', 'multilabel': multilabel,
        'members': {str(value): [key for key, label in expanded if label == value] for value in counts[field]},
    }
    write_table(counts, directory / 'data', stem)
    visible = counts.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, .5 * len(visible) + 1.5)))
    ordered = visible.iloc[::-1]
    ax.barh(ordered[field], ordered['count'], color='#286f9b')
    ax.set_title(title, loc='center', fontweight='bold')
    ax.set_xlabel('Records' if len(counts) <= top_n else f'Records · top {top_n} of {len(counts)} categories; complete counts in CSV')
    return save_figure(fig, directory, stem, formats, dpi)


@audited_bundle
def build_report(
    profile_path: Path, input_path: Path, out_dir: Path, config: dict[str, object], *,
    candidates_path: Path | None = None, ratings_path: Path | None = None,
    exports_path: Path | None = None, database_path: Path | None = None,
    confidence_level: float = 0.95,
) -> list[Path]:
    """Import a provider contract and reuse evaluation, databases and generic figures."""
    contract = load_source_contract(profile_path)
    adapter = LocalSourceAdapter(contract)
    rows = adapter.read(input_path)
    candidates = adapter.read(candidates_path) if candidates_path else None
    exports = adapter.read(exports_path) if exports_path else None
    decisions_observed = all(row.get('decision') for row in rows) and bool(rows)
    if not rows and input_path.suffix.casefold() == '.csv':
        with input_path.open(encoding='utf-8', newline='') as handle:
            columns = set(csv.DictReader(handle).fieldnames or [])
        decisions_observed = {'decision', 'classifier_supported_fields'} <= columns
    if any(row.get('decision') for row in rows) and not decisions_observed:
        raise ValueError('classification input mixes decided and unclassified records; supply unclassified records as candidates')
    if ratings_path and not decisions_observed:
        raise ValueError('evaluation requires classification decisions')
    classified_ids = {row['entity_id'] for row in rows} if decisions_observed else None
    included = {row['entity_id'] for row in rows if row.get('decision') in {'accept', 'accept_adjacent'} and not row.get('error')} if decisions_observed else None
    exported = {row['entity_id'] for row in exports} if exports is not None else None
    if exported is not None and (included is None or not exported <= included):
        raise ValueError('export contains records absent from accepted classifications')
    classifications = out_dir / 'normalized' / 'classifications.csv'
    _write_records(classifications, rows, contract, ('classifier_supported_fields',) if decisions_observed else ())
    register_input(classifications)
    candidate_csv = None
    if candidates is not None:
        candidate_csv = out_dir / 'normalized' / 'candidates.csv'
        _write_records(candidate_csv, candidates, contract)
        register_input(candidate_csv)
    if exports is not None:
        _write_records(out_dir / 'normalized' / 'exports.csv', exports, contract)
        register_input(out_dir / 'normalized' / 'exports.csv')
    configure_style({**config, 'field_labels': contract.field_labels})
    formats, dpi = list(config.get('formats', ['svg', 'pdf'])), int(config.get('dpi', 160))
    top_n = int(config.get('top_n', 15))
    if top_n < 1:
        raise ValueError('plots.top_n must be at least one')
    frame = load_frame(classifications, (contract.identity, 'entity_id'))
    directory = out_dir / 'figures'
    outputs: list[Path] = []
    if any(str(row.get('decision', '')).strip() for row in rows):
        outputs += _frequency(frame, 'decision', directory, '01_decisions', 'Classification decisions', formats, dpi, top_n=top_n)
    for index, field in enumerate(contract.fields, 1):
        if field in frame and frame[field].ne('').any():
            outputs += _frequency(frame, field, directory, f'taxonomy_{index:02d}_{field}', taxonomy_field_label(field), formats, dpi, field in contract.multilabel, top_n)
    eligible = None
    if candidates is not None and all(row.get('prefilter_band') in {'candidate', 'borderline', 'reject'} for row in candidates):
        eligible = {row['entity_id'] for row in candidates if row['prefilter_band'] in {'candidate', 'borderline'}}
    memberships = {
        'candidate_selection': {row['entity_id'] for row in candidates} if candidates is not None else None,
        'enrichment_prefilter': eligible, 'classification': classified_ids, 'inclusion': included, 'export': exported,
    }
    stages = pd.DataFrame([{'stage': stage, 'count': len(ids) if ids is not None else None,
        'unit': contract.entity_kind, 'basis': 'observed' if ids is not None else 'not_observed',
        'entity_ids': ';'.join(sorted(ids)) if ids is not None else ''} for stage, ids in memberships.items()])
    stages.attrs['verification_recipe'] = {
        'kind': 'stages', 'input_sha256': file_hash(classifications),
        'candidate_sha256': file_hash(candidate_csv) if candidate_csv else None,
        'export_sha256': file_hash(out_dir / 'normalized' / 'exports.csv') if exports is not None else None,
    }
    write_table(stages, directory / 'data', '02_pipeline_stages')
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, row in enumerate(stages.to_dict('records')):
        if pd.isna(row['count']):
            ax.text(0, index, 'N/A', va='center')
        else:
            ax.barh(index, row['count'], color='#286f9b')
    ax.set_yticks(range(5), ['Candidate selection', 'Enrichment / prefilter', 'Classification', 'Inclusion', 'Export'])
    ax.invert_yaxis()
    ax.set_xlabel(contract.entity_kind.title() + ' records')
    ax.set_title('Pipeline stages', loc='center', fontweight='bold')
    outputs += save_figure(fig, directory, '02_pipeline_stages', formats, dpi)
    availability = []
    for capability, column, stem in [('publication_dates', 'publication_year', '03_publications'), ('citations', 'cited_by_count', '04_citations')]:
        available = capability in contract.capabilities and column in frame and frame[column].ne('').any()
        availability.append({'plot': stem, 'capability': capability, 'available': available, 'reason': '' if available else 'unsupported capability or absent field'})
        if available:
            outputs += _frequency(frame, column, directory, stem, 'Publication years' if capability == 'publication_dates' else 'Citation counts', formats, dpi, top_n=top_n)
    write_table(pd.DataFrame(availability), directory / 'data', 'plot_availability')
    from ids_meta.common.reporting import ReportingDatabase

    store = database_path or out_dir / 'reporting.sqlite3'
    with ReportingDatabase(store) as database:
        run_id = database.ingest_records(contract, classifications, 'classification' if decisions_observed else 'candidate_selection')
        if candidate_csv:
            database.ingest_records(contract, candidate_csv, 'candidate_selection', run_id)
        if exports is not None:
            database.ingest_records(contract, out_dir / 'normalized' / 'exports.csv', 'export', run_id)
        with database.db:
            database.snapshot(input_path, 'provider_input')
            database.snapshot(profile_path, 'source_profile')
    if ratings_path:
        from ids_meta.common.evaluation import evaluate_results, load_rating_profile
        from ids_meta.plotting.evaluation import build as plot_evaluation

        rating = load_rating_profile(ratings_path, contract.name, contract=contract)
        evaluation = out_dir / 'evaluation'
        result = evaluate_results(rating, classifications, evaluation, candidate_csv,
            reporting_database=store, confidence_level=confidence_level)
        outputs.extend(result.outputs)
        outputs.extend(plot_evaluation(evaluation, {**config, 'field_labels': contract.field_labels}))
        with ReportingDatabase(store) as database:
            with database.db:
                database.snapshot(input_path, 'provider_input')
                database.snapshot(profile_path, 'source_profile')
                database.snapshot(ratings_path, 'reference_profile')
            database.add_figures(out_dir)
    else:
        with ReportingDatabase(store) as database:
            database.add_figures(out_dir)
    (out_dir / 'README.md').write_text(
        f'# Reporting bundle\n\nProvider: {contract.display_name or contract.name}\n\nEntity type: {contract.entity_kind}\n\n'
        'Normalized inputs are in `normalized/`. Numeric tables and contribution IDs are in `figures/data/`.\n'
        'Figure manifests distinguish integrity checks from independently recomputable tables.\n'
        'Missing stages and unsupported capabilities are N/A. Stage memberships do not imply a nested pipeline.\n', encoding='utf-8')
    return outputs
