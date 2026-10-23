"""Keep baseline evidence independent of RADAR's later enrichment and run order."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from radar.common.baselines import KeywordBaselineConfig, classify_keyword_baseline, load_keyword_taxonomy_profile
from radar.common.cache import COLLECTED_EVIDENCE_POLICY, PipelineCache, collected_record_hash
from radar.common.csvio import read_csv
from radar.common.nli import EvidenceChunk, NLIField, NLIProfile, NLIRuntimeConfig, NLIScore, classify_nli_baseline
from radar.common.ollama import OllamaClient
from tests.fixtures import SYSTEM_TYPE_PROFILE, write_csv
from radar.common.config import Settings


class RecordingBackend:
    """Record the text actually submitted to the shared NLI scoring paths."""
    model_id = 'fixture/evidence'
    revision = 'fixture-revision'

    def __init__(self):
        """Start an empty observed-evidence audit."""
        self.seen = []

    def chunk_sections(self, sections, premise_tokens, overlap):
        """Capture original text and supply deterministic chunks without model downloads."""
        self.seen.extend(section.text for section in sections)
        return [EvidenceChunk(i,section.source,section.text) for i,section in enumerate(sections)]

    def score(self, pairs, batch_size):
        """Return a decisive entailment for the single configured excluded role."""
        return [NLIScore(.95,.04,.01) for _ in pairs]

    def start_record(self):
        """Implement the generative-NLI audit hook without making any requests."""
        return None

    def record_audit(self):
        """Return the extra column expected from a generative-NLI scoring backend."""
        return {'llm_nli_metadata_json': json.dumps({'fixture': True})}


class BaselineEvidenceTests(unittest.TestCase):
    """Exercise original-record capture, isolation, integrity and honest failure modes."""
    def fixtures(self, root, source):
        """Create contrasting original and enriched evidence for a candidate."""
        identity,namespace = ('github_repo_id','github.repository') if source=='github' else ('openalex_id','openalex.work')
        candidate = {identity:'1','evidence_hash':'enriched-candidate-hash','prefilter_band':'candidate',
            'full_name':'fixture/repo','title':'ORIGINAL paper','abstract':'ENRICHED alpha detector release'}
        original = ({'repository':{'full_name':'fixture/repo'},'readme_text':'ORIGINAL unrelated image processing library'}
            if source=='github' else {'work':{'title':'ORIGINAL paper','abstract':'ORIGINAL unrelated image processing study'}})
        enriched = ({'repository':{'full_name':'fixture/repo'},'readme_text':'ENRICHED alpha detector intrusion detection system',
            'docs_text':'ENRICHED runtime network traffic attack detection'} if source=='github' else
            {'work':{'title':'ORIGINAL paper','abstract':'ENRICHED alpha detector release',
                'full_text':'ENRICHED intrusion detection dataset release full text'}})
        path = root/'candidates.csv'
        write_csv(path,[candidate],list(candidate))
        return namespace,original,enriched,path

    def test_original_snapshot_survives_enrichment_mutation_and_reopening(self):
        """Persist initial evidence separately for both repository and paper records."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ['github','openalex']:
                with self.subTest(source=source):
                    namespace,original,enriched,_ = self.fixtures(root,source)
                    with PipelineCache(root/'cache.sqlite3') as cache:
                        cache.put_collected_record(namespace,'1',original,'collected-at')
                        cache.put_record(namespace,'1',enriched,'enriched-at')
                        self.assertEqual(cache.get_record(namespace,'1'),enriched)
                    with PipelineCache(root/'cache.sqlite3',read_only=True) as cache:
                        restored = cache.get_collected_record(namespace,'1')
                        self.assertEqual(restored,original)
                        self.assertNotIn('ENRICHED',json.dumps(restored))

    def test_keyword_predictions_do_not_change_after_live_cache_enrichment(self):
        """A later RADAR run cannot inject new keyword hits into a baseline."""
        with tempfile.TemporaryDirectory() as directory, patch.object(OllamaClient,'generate',side_effect=AssertionError('keyword must not call an LLM')):
            root = Path(directory)
            for source in ['github','openalex']:
                with self.subTest(source=source), PipelineCache(root/f'{source}.sqlite3') as cache:
                    namespace,original,enriched,candidates = self.fixtures(root,source)
                    cache.put_collected_record(namespace,'1',original,'collected-at')
                    study = 'ids_repositories' if source=='github' else 'ids_datasets'
                    config_root = Path(__file__).resolve().parents[1]/'config/studies'
                    settings = Settings(root,{'paths':{source+'_keyword':str(config_root/study/'baselines/keyword.toml'),
                        'github_system_types':str(config_root/'ids_repositories/taxonomy.toml')}})
                    taxonomy = load_keyword_taxonomy_profile(settings,source)
                    terms = {'positive':{'release':{'terms':['alpha detector'],'weight':7}}}
                    before = classify_keyword_baseline(source,candidates,root/'before.csv',cache,terms,
                        KeywordBaselineConfig(7,4),taxonomy,resume=False,adapter_config={'system_type_profile':SYSTEM_TYPE_PROFILE})
                    cache.put_record(namespace,'1',enriched,'enriched-at')
                    after = classify_keyword_baseline(source,candidates,root/'after.csv',cache,terms,
                        KeywordBaselineConfig(7,4),taxonomy,resume=False,adapter_config={'system_type_profile':SYSTEM_TYPE_PROFILE})
                    self.assertEqual(before[0]['decision'],after[0]['decision'])
                    self.assertEqual(float(after[0]['keyword_score']),0)
                    self.assertEqual(before[0]['evidence_hash'],after[0]['evidence_hash'])
                    saved = read_csv(root/'after.csv')[0]
                    self.assertEqual(saved['classifier_evidence_source'],'collection_snapshot')
                    self.assertEqual(saved['classifier_evidence_policy'],COLLECTED_EVIDENCE_POLICY)
                    self.assertEqual(saved['classifier_evidence_snapshot_hash'],collected_record_hash(original))
                    cache.put_collected_record(namespace,'1',enriched,'new-collection')
                    fresh = classify_keyword_baseline(source,candidates,root/'fresh.csv',cache,terms,
                        KeywordBaselineConfig(7,4),taxonomy,resume=True,adapter_config={'system_type_profile':SYSTEM_TYPE_PROFILE})
                    self.assertEqual(float(fresh[0]['keyword_score']),7)
                    self.assertNotEqual(fresh[0]['evidence_hash'],after[0]['evidence_hash'])
                    self.assertEqual(fresh[0]['classifier_evidence_snapshot_hash'],collected_record_hash(enriched))


    def test_both_nli_backends_receive_original_text_only(self):
        """Local NLI and generative NLI share the same acquisition-only input boundary."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ['github','openalex']:
                for method in ['nli','llm_nli']:
                    with self.subTest(source=source,method=method), PipelineCache(root/f'{source}.sqlite3') as cache:
                        namespace,original,enriched,candidates = self.fixtures(root,source)
                        cache.put_collected_record(namespace,'1',original,'collected-at')
                        cache.put_record(namespace,'1',enriched,'enriched-at')
                        backend = RecordingBackend()
                        runtime = NLIRuntimeConfig(backend.model_id,backend.revision,'cpu',4,128,64,8,4,1,.5,.1,.5,.1)
                        role = 'role' if source=='github' else 'paper_role'
                        excluded = 'out_of_scope' if source=='github' else 'unrelated'
                        profile = NLIProfile(source,(NLIField(role,'exclusive','uncertain',{excluded:'An unrelated work'}),),{},root/'nli.toml')
                        rows = classify_nli_baseline(source,candidates,root/f'{method}.csv',cache,profile,runtime,
                            backend=backend,method=method,resume=False,adapter_config={'system_type_profile':SYSTEM_TYPE_PROFILE})
                        self.assertTrue(backend.seen)
                        self.assertIn('ORIGINAL',' '.join(backend.seen))
                        self.assertNotIn('ENRICHED',' '.join(backend.seen))
                        self.assertEqual(rows[0]['classifier_evidence_snapshot_hash'],collected_record_hash(original))
                        self.assertEqual(rows[0]['decision'],'exclude')

    def test_enriched_records_cannot_be_used_to_invent_a_missing_snapshot(self):
        """Fail before NLI inference when only mutated cache data or CSV evidence remains."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace,_,enriched,candidates = self.fixtures(root,'github')
            with PipelineCache(root/'cache.sqlite3') as cache:
                cache.put_record(namespace,'1',enriched,'enriched-at')
                backend = RecordingBackend()
                runtime = NLIRuntimeConfig(backend.model_id,backend.revision,'cpu',4,128,64,8,4,1,.5,.1,.5,.1)
                profile = NLIProfile('github',(NLIField('role','exclusive','uncertain',{'out_of_scope':'unrelated'}),),{},root/'profile')
                with self.assertRaisesRegex(ValueError,'missing collected evidence snapshot.*--no-resume'):
                    classify_nli_baseline('github',candidates,root/'nli.csv',cache,profile,runtime,backend=backend)
                self.assertEqual(backend.seen,[])
                self.assertIsNone(cache.get_record(namespace+'.collected','1'))

    def test_corrupt_snapshot_is_rejected_and_fresh_collection_can_replace_it(self):
        """Detect altered snapshot bytes and distinguish a genuine new acquisition."""
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory)/'cache.sqlite3') as cache:
            original = {'readme_text':'original text with \ud800 surrogate'}
            cache.put_collected_record('github.repository','1',original,'first')
            self.assertEqual(cache.get_collected_record('github.repository','1')['readme_text'],'original text with \ufffd surrogate')
            snapshot = cache.get_record('github.repository.collected','1')
            snapshot['payload']['readme_text'] = 'tampered'
            cache.put_record('github.repository.collected','1',snapshot,'tampered')
            with self.assertRaisesRegex(ValueError,'invalid collected evidence snapshot'):
                cache.get_collected_record('github.repository','1')
            fresh = {'readme_text':'genuinely recollected text'}
            cache.put_collected_record('github.repository','1',fresh,'second')
            self.assertEqual(cache.get_collected_record('github.repository','1'),fresh)

    def test_working_record_and_snapshot_are_written_atomically(self):
        """A failed acquisition write cannot leave a misleading baseline-only snapshot."""
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory)/'cache.sqlite3') as cache:
            cache.connection.execute("CREATE TRIGGER reject_working_record BEFORE INSERT ON records WHEN NEW.namespace='github.repository' BEGIN SELECT RAISE(ABORT,'fixture failure'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                cache.put_collected_record('github.repository','1',{'readme_text':'original'},'now')
            self.assertIsNone(cache.get_record('github.repository.collected','1'))
            self.assertIsNone(cache.get_record('github.repository','1'))
