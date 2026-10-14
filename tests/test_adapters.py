"""Exercise provider contracts, union discovery and reporting without network calls."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

import requests

from ids_meta.adapters.contracts import ProviderDescriptor, ProviderQuery, SearchPage, SourceRecord
from ids_meta.adapters.github import GitHubAdapter
from ids_meta.adapters.http import JsonTransport
from ids_meta.adapters.local import LocalDiscoveryAdapter
from ids_meta.adapters.openalex import OpenAlexAdapter
from ids_meta.adapters.registry import PROVIDERS, create_adapter, register_provider
from ids_meta.adapters.semantic_scholar import SemanticScholarAdapter
from ids_meta.cli import main
from ids_meta.common.config import ConfigError, load_settings
from ids_meta.common.discovery import DiscoverySource, collect_sources
from ids_meta.common.provenance import file_hash
from ids_meta.common.verification import verify_directory
from ids_meta.plotting.generic import build_report

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'config/studies/conformal_treatment_effects'


class FixtureAdapter:
    """Supply provider-native records through the real shared runner."""
    def __init__(self, name, rows, cursor=None):
        """Bind a fixture provider and optional deliberately cyclic cursor."""
        self.descriptor = ProviderDescriptor(name, ('paper',))
        self.rows, self.cursor = rows, cursor

    def queries(self, profile):
        """Expose one provider-native query."""
        return [ProviderQuery('concepts', 'native syntax')]

    def search(self, query, cursor, page_size):
        """Return a full raw page independent of the retained study cohort."""
        return SearchPage(self.rows, self.cursor, {'records': self.rows}, len(self.rows))

    def validate_query(self, query):
        """Ensure fixtures exercise structural validation too."""
        if not query.name or not query.text or query.parameters:
            raise ValueError('invalid fixture query')

    def normalize(self, raw):
        """Normalize fixture identities and evidence without applying decisions."""
        return SourceRecord(self.descriptor.name, raw['id'], raw.get('kind', 'paper'),
            raw.get('identifiers', {}), {'title': raw.get('title', 'Fixture')},
            {'abstract': raw.get('abstract', '')}, raw)


class AdapterTests(unittest.TestCase):
    """Check observable adapter semantics and the shared cohort boundary."""
    def _source(self, root, name, rows, cursor=None):
        """Create a fingerprinted profile and injected discovery fixture."""
        path = root / f'{name}.toml'
        path.write_text('schema_version = "1.0"\n')
        return DiscoverySource(name, path, {}, FixtureAdapter(name, rows, cursor))

    def test_union_deduplicates_strong_ids_and_retains_source_recovery(self):
        """The same DOI becomes one entity with both provider memberships and raw evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [self._source(root, 'a', [{'id': '1', 'identifiers': {'doi': '10.1/one'}}]),
                self._source(root, 'b', [{'id': '2', 'identifiers': {'doi': '10.1/one'}, 'abstract': 'More evidence'},
                    {'id': '3', 'title': 'Other'}])]
            output = root / 'cohort.json'
            rows = collect_sources(sources, output, limit=None, entity_kind='paper')
            self.assertEqual(len(rows), 2)
            shared = next(row for row in rows if row['paper_id'] == 'doi:10.1/one')
            self.assertEqual({member['provider'] for member in shared['source_memberships']}, {'a', 'b'})
            self.assertEqual(shared['abstract'], 'More evidence')
            self.assertEqual(len(shared['source_records']), 2)
            audit = json.loads(output.with_suffix('.collection.json').read_text())
            self.assertEqual(audit['candidate_sha256'], file_hash(output))
            self.assertEqual(audit['source_recovery']['b']['retained'], 2)
            self.assertEqual(audit['queries'][1]['pages'][0]['response']['records'][1]['id'], '3')

    def test_transitive_identity_bridge_and_conflicting_identifiers(self):
        """A DOI/arXiv bridge merges identities, while contradictory DOI assignments stay flagged."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [self._source(root, 'a', [{'id': '1', 'identifiers': {'doi': '10.1/one'}}]),
                self._source(root, 'b', [{'id': '2', 'identifiers': {'arxiv': '1234.5678'}}]),
                self._source(root, 'c', [{'id': '3', 'identifiers': {'doi': '10.1/one', 'arxiv': '1234.5678'}}])]
            output = root / 'cohort.json'
            self.assertEqual(len(collect_sources(sources, output, limit=None)), 1)
            sources.append(self._source(root, 'd', [{'id': '4', 'identifiers': {'doi': '10.1/two', 'arxiv': '1234.5678'}}]))
            rows = collect_sources(sources, output, limit=None)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row['identity_conflict'] for row in rows))

    def test_native_url_identities_do_not_collide_across_providers(self):
        """Equal native strings are distinct without a declared strong identifier."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [self._source(root, name, [{'id': 'https://example.org/one'}]) for name in ('a', 'b')]
            rows = collect_sources(sources, root / 'cohort.json', limit=None)
            self.assertEqual(len({row['paper_id'] for row in rows}), 2)
            self.assertEqual(len({row['entity_id'] for row in rows}), 2)

    def test_invalid_advertised_capability_is_rejected(self):
        """An extension cannot claim evidence support without implementing enrichment."""
        descriptor = ProviderDescriptor('incomplete_papers', ('paper',), ('discovery', 'evidence'))
        adapter = FixtureAdapter(descriptor.name, [])
        adapter.descriptor = descriptor
        register_provider(descriptor, lambda settings, path, client=None: adapter)
        self.addCleanup(PROVIDERS.pop, descriptor.name)
        with self.assertRaisesRegex(ValueError, 'unimplemented evidence'):
            create_adapter(descriptor.name, {}, Path())

    def test_titles_do_not_merge_and_entity_kinds_remain_separate(self):
        """Equal titles flag possible duplicates; a paper and artifact never collapse."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, 'a', [{'id': '1'}, {'id': '2'}, {'id': '3', 'kind': 'artifact'}])
            source.adapter.descriptor = ProviderDescriptor('a', ('paper', 'artifact'))
            rows = collect_sources([source], root / 'cohort.json', limit=None)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]['possible_title_duplicates']), 1)
            self.assertNotIn('possible_title_duplicates', rows[2])
            with self.assertRaisesRegex(ValueError, 'incompatible entity kinds'):
                collect_sources([source], root / 'invalid.json', entity_kind='paper')
            self.assertFalse((root / 'invalid.json').exists())

    def test_cursor_cycles_fail_without_publishing_a_partial_cohort(self):
        """Broken pagination cannot silently truncate or loop forever."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, 'a', [{'id': '1'}], 'repeat')
            with self.assertRaisesRegex(RuntimeError, 'repeated study-search cursor'):
                collect_sources([source], root / 'cohort.json', limit=None)
            self.assertFalse((root / 'cohort.json').exists())

    def test_local_mapping_pagination_and_judgment_isolation(self):
        """Local evidence is usable, but previous decisions are retained only in raw audit data."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / 'papers.csv'
            records.write_text('id,name,doi,full_text_complete,decision\n1,One,https://doi.org/10.1234/ABC,true,accept\n2,Two,,false,exclude\n')
            profile = root / 'local.toml'
            profile.write_text('schema_version="1.0"\nprovider="local"\npath="papers.csv"\nidentity="id"\nentity_kind="paper"\nnamed_entity_seeding=false\n[mapping]\ntitle="name"\n')
            rows = collect_sources([DiscoverySource('local', profile, {'per_page': 1})], root / 'cohort.json', limit=None)
            self.assertEqual([row['title'] for row in rows], ['One', 'Two'])
            self.assertEqual(rows[0]['paper_id'], 'doi:10.1234/abc')
            self.assertNotIn('decision', rows[0])
            self.assertEqual(rows[0]['source_records'][0]['native']['decision'], 'accept')
            self.assertIs(rows[0]['full_text_complete'], True)
            adapter = LocalDiscoveryAdapter({}, profile)
            query = adapter.queries(tomllib.loads(profile.read_text()))[0]
            records.write_text(records.read_text() + '3,Three,,false,accept\n')
            with self.assertRaisesRegex(ValueError, 'changed during collection'):
                adapter.search(query, None, 1)

    def test_semantic_scholar_uses_native_bulk_tokens_and_identifiers(self):
        """No OpenAlex query translation or relevance-search pagination leaks into bulk retrieval."""
        client = Mock()
        client.get.return_value = {'data': [{'paperId': 'native', 'title': 'Paper',
            'externalIds': {'DOI': '10.1234/ABC', 'ArXiv': '2401.00001v2'}}], 'token': 'next', 'total': 50}
        adapter = SemanticScholarAdapter({}, REVIEW / 'search/semantic_scholar.toml', client)
        profile = tomllib.loads((REVIEW / 'search/semantic_scholar.toml').read_text())
        query = adapter.queries(profile)[0]
        page = adapter.search(query, 'opaque-token', 2)
        params = client.get.call_args.args[1]
        self.assertEqual(params['query'], profile['queries'][0]['search'])
        self.assertEqual(params['token'], 'opaque-token')
        self.assertNotIn('limit', params)
        self.assertEqual(page.next_cursor, 'next')
        record = adapter.normalize(page.records[0])
        self.assertEqual(record.identifiers, {'doi': '10.1234/abc', 'arxiv': '2401.00001'})
        with self.assertRaisesRegex(ValueError, 'search_parameter must be bulk'):
            adapter.queries({**profile, 'search_parameter': 'search.exact'})

    def test_semantic_scholar_title_lookup_abstains_on_ambiguous_matches(self):
        """Evidence lookup does not select one of several equally titled papers."""
        client = Mock()
        client.get.return_value = {'data': [{'paperId': '1', 'title': 'Same'}, {'paperId': '2', 'title': 'Same'}]}
        adapter = SemanticScholarAdapter({}, Path(), client)
        self.assertEqual(adapter.enrich({'title': 'Same'}), {})

    def test_github_ceiling_is_bounded_and_native_catalogue_strategies_are_preserved(self):
        """The simple search advertises its API ceiling while catalogue collection retains sharding."""
        client = Mock()
        client.search.return_value = {'items': [{'id': 1, 'full_name': 'a/b'}], 'total_count': 2000}
        adapter = GitHubAdapter({}, Path(), client)
        self.assertFalse(adapter.search(ProviderQuery('q', 'intrusion'), '10', 100).complete)
        self.assertEqual(adapter.normalize({'id': 1}).native_id, '1')
        with patch('ids_meta.adapters.github.collect', return_value={'discovered': 1}) as github:
            adapter.collect_catalogue({}, {}, Mock(), 5, resume=False)
            self.assertFalse(github.call_args.kwargs['resume'])
            self.assertIs(github.call_args.kwargs['client'], client)
        with patch('ids_meta.adapters.openalex.collect_papers', return_value=({'discovered': 1}, {}, {})) as openalex:
            OpenAlexAdapter({}, Path(), client).collect_catalogue({}, {}, Mock(), 5)
            self.assertIs(openalex.call_args.args[-1].client, client)

    def test_shared_transport_retries_only_transient_failures(self):
        """Rate-limited requests honor bounded Retry-After; authentication errors fail immediately."""
        limited, success = Mock(status_code=429, headers={'Retry-After': '999'}), Mock(status_code=200)
        success.json.return_value = {'data': []}
        with patch('ids_meta.adapters.http.requests.get', side_effect=[limited, success]) as get, patch(
            'ids_meta.adapters.http.time.sleep') as sleep:
            self.assertEqual(JsonTransport(pause=0, max_wait=4).get('https://example.org'), {'data': []})
            self.assertEqual(get.call_count, 2)
            sleep.assert_called_once_with(4)
        forbidden = Mock(status_code=401)
        forbidden.raise_for_status.side_effect = requests.HTTPError('unauthorized')
        with patch('ids_meta.adapters.http.requests.get', return_value=forbidden) as get:
            with self.assertRaises(requests.HTTPError):
                JsonTransport(pause=0).get('https://example.org')
            self.assertEqual(get.call_count, 1)

    def test_registry_extension_and_unknown_capabilities_validate_offline(self):
        """A provider registration enables reviews without adding provider-name branches."""
        name = 'fixture_papers'
        descriptor = ProviderDescriptor(name, ('paper',))
        factory = lambda settings, path, client=None: FixtureAdapter(name, [{'id': '1'}])
        register_provider(descriptor, factory)
        self.addCleanup(PROVIDERS.pop, name)
        with self.assertRaisesRegex(ValueError, 'duplicate provider'):
            register_provider(descriptor, factory)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'config', root / 'config')
            manifest = root / 'config/studies/conformal_treatment_effects/study.toml'
            manifest.write_text(manifest.read_text().replace('name = "openalex"', f'name = "{name}"'))
            provider = root / 'config/shared/sources/openalex.toml'
            provider.write_text(provider.read_text().replace('name = "openalex"', f'name = "{name}"'))
            with patch('requests.get', side_effect=AssertionError('network')):
                self.assertEqual(load_settings(manifest).section('study')['sources'], [name])
                adapter = create_adapter(name, {}, Path())
                self.assertEqual(adapter.descriptor, descriptor)
            provider.write_text(provider.read_text().replace('entity_kind = "paper"', 'entity_kind = "repository"'))
            with self.assertRaisesRegex(ConfigError, 'must provide paper'):
                load_settings(manifest)

    def test_review_cli_collects_union_and_explicit_source_is_separate(self):
        """One review manifest selects two adapters without requiring a source switch."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'config', root / 'config')
            manifest = root / 'config/studies/conformal_treatment_effects/study.toml'
            # Place the source table before evaluation to retain valid TOML ownership.
            text = manifest.read_text().replace('[evaluation]', '[[sources]]\nname="semantic_scholar"\nprofile="../../shared/sources/semantic_scholar.toml"\nsearch="search/semantic_scholar.toml"\n\n[evaluation]')
            manifest.write_text(text)
            oa = Mock()
            oa.select = 'id,title'
            oa.get.return_value = {'results': [{'id': 'https://openalex.org/W1', 'doi': 'https://doi.org/10.1234/one', 'title': 'Paper'}], 'meta': {}}
            scholar = Mock()
            scholar.get.return_value = {'data': [{'paperId': 's1', 'externalIds': {'DOI': '10.1234/one'}, 'title': 'Paper'}]}
            with patch('ids_meta.adapters.openalex.OpenAlexClient', return_value=oa), patch(
                'ids_meta.adapters.semantic_scholar.JsonTransport', return_value=scholar), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(manifest), 'collect']), 0)
                self.assertEqual(main(['--config', str(manifest), 'collect', '--source', 'semantic_scholar']), 0)
            interim = root / 'artifacts/studies/conformal_treatment_effects/interim'
            union = json.loads((interim / 'cohort/candidates.json').read_text())
            self.assertEqual(len(union), 1)
            self.assertEqual({member['provider'] for member in union[0]['source_memberships']}, {'openalex', 'semantic_scholar'})
            selected = json.loads((interim / 'cohort_semantic_scholar/candidates.collection.json').read_text())
            self.assertEqual(selected['providers'], ['semantic_scholar'])

    def test_source_reports_score_each_provider_against_the_same_references(self):
        """Union recovery and per-provider misses remain auditable and independently verifiable."""
        examples = ROOT / 'examples/reporting'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = json.loads((examples / 'papers.json').read_text())
            for index, row in enumerate(rows):
                row['source_memberships'] = [{'provider': 'a' if index == 0 else 'b', 'native_id': str(index)}]
            candidates = root / 'cohort.json'
            candidates.write_text(json.dumps(rows))
            candidates.with_suffix('.collection.json').write_text(json.dumps({
                'providers': ['a', 'b', 'empty'], 'candidate_sha256': file_hash(candidates)}))
            report = root / 'report'
            build_report(examples / 'papers.toml', candidates, report, {'formats': ['svg']},
                candidates_path=candidates, ratings_path=examples / 'paper_references.toml')
            metrics = [json.loads((report / 'evaluation' / path / 'metrics.json').read_text())
                for path in ('', 'sources/a', 'sources/b', 'sources/empty')]
            self.assertEqual([value['operational']['tp'] for value in metrics], [1, 1, 0, 0])
            self.assertEqual([value['operational']['fn'] for value in metrics], [1, 1, 2, 2])
            result = verify_directory(report)
            self.assertTrue(result['ok'], result)
