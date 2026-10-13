"""Current-format fixtures shared by pipeline and reporting tests."""
from pathlib import Path
import tomllib

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.common.run import utc_now

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'config/studies/ids_repositories/taxonomy.toml').open('rb') as handle:
    SYSTEM_TYPE_PROFILE = tomllib.load(handle)


def write_csv(path, rows, columns):
    """Write classification fixtures with an explicit supported-field declaration."""
    rows = [dict(row) for row in rows]
    columns = list(columns)
    identities = {'github_repo_id', 'openalex_id', 'repo_id', 'paper_id', 'doi', 'entity_id'}
    classified = any('decision' in row and identities.intersection(row) for row in rows)
    if classified:
        for row in rows:
            row.setdefault('classifier_supported_fields', '*')
            if 'openalex_id' in row and row.get('artifact_name'):
                row['artifact_display_name'] = row.get('artifact_display_name') or row['artifact_name']
                row['resource_name'] = row['artifact_display_name']
        for field in ('artifact_display_name', 'resource_name'):
            if any(field in row for row in rows) and field not in columns:
                columns.append(field)
        if 'classifier_supported_fields' not in columns:
            columns.append('classifier_supported_fields')
    return atomic_write_csv(path, rows, columns)


def fixture_collection(cache, source):
    """Record the explicit selection used by a prefilter fixture."""
    namespace, field = ('github.repository', 'repository_ids') if source == 'github' else ('openalex.work', 'work_ids')
    selection = cache.get_record(source + '.collection', 'latest') or {field: [key for key, _, _ in cache.records(namespace)]}
    selection['selected_at'] = utc_now()
    cache.put_record(source + '.collection', 'latest', selection, selection['selected_at'])
    return cache
