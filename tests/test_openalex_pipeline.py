from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from tests.fixtures import fixture_collection
from typing import Any
from unittest.mock import Mock, patch

from radar.common.cache import PipelineCache
from radar.common.config import load_prompt_profile, load_settings, load_term_profile
from radar.openalex.collect import OpenAlexClient, normalize_work, reconstruct_abstract
from radar.common.paper_collection import is_survey_work, publisher_diverse_surveys, rank_work_pool
from tests.fixtures import collect_paper_fixture as collect
from radar.openalex.pipeline import classify, export, export_survey_sources, prefilter
from radar.openalex.reconcile import reconcile
from radar.openalex.entities import canonicalize_artifact_ids
from radar.openalex.schema import CANDIDATE_COLUMNS, CATALOGUE_COLUMNS, CLASSIFICATION_COLUMNS, RESPONSE_SCHEMA, ensure_resource_name, normalize_response
from tests.fixtures import write_csv as atomic_write_csv


ROOT = Path(__file__).resolve().parents[1]


class FakePaperModel:
    model = "fixture-model"

    def __init__(self) -> None:
        """Initialize the fixture model's invocation counter."""
        self.calls = 0

    def generate(self, prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic release or downstream-paper classifications."""
        self.calls += 1
        downstream = "classification accuracy" in prompt
        if downstream:
            return {
                "paper_role": "downstream_application", "relevance_tier": "excluded", "artifact_name": "",
                "artifact_display_name": "", "artifact_aliases": [], "artifact_type": "unknown", "domain": "unknown", "data_form": "unknown",
                "capture_environment": "unknown", "labeling": "unknown",
                "direct_release_evidence": False, "primary_reason": "Applies existing data.",
                "evidence": [{"source": "Abstract", "quote": "classification accuracy"}],
            }
        return {
            "paper_role": "original_release", "relevance_tier": "core", "artifact_name": "Network Attack Trace Corpus",
            "artifact_display_name": "NAT Corpus", "artifact_aliases": ["NAT Corpus"], "artifact_type": "dataset", "domain": "general_network",
            "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled",
            "direct_release_evidence": True, "primary_reason": "Introduces and releases attack traces.",
            "evidence": [{"source": "Abstract", "quote": "introduce a new labeled intrusion detection dataset named NAT Corpus"}],
        }


class OpenAlexPipelineTests(unittest.TestCase):
    def test_retrieval_ranking_uses_evidence_not_citations(self) -> None:
        """Verify textual evidence outranks citation popularity during retrieval."""
        profile = {
            "positive": {
                "scope": {"weight": 3, "terms": ["intrusion detection"]},
                "release": {"weight": 4, "terms": ["introduce"]},
            },
        }
        pool = {
            "W1": {"native_id": "W1", "title": "Highly cited unrelated analysis paper", "abstract": "", "cited_by_count": 10000},
            "W2": {"native_id": "W2", "title": "We introduce an intrusion detection corpus", "abstract": "", "cited_by_count": 0},
        }
        ranked, _scores = rank_work_pool(pool, profile, 0, 3)
        self.assertEqual([row["native_id"] for row in ranked], ["W2", "W1"])

    def test_survey_consensus_can_bypass_textual_retrieval_floor(self) -> None:
        """Verify opaque dataset-paper titles survive when multiple surveys cite them."""
        pool = {
            "W-ctu": {
                "native_id": "W-ctu", "title": "An empirical comparison of botnet detection methods",
                "abstract": "", "cited_by_count": 937,
            },
        }
        ranked, scores = rank_work_pool(
            pool, {"positive": {}}, 1, 3, retain_below_minimum={"W-ctu"},
        )
        self.assertEqual(scores["W-ctu"][0].score, 0)
        self.assertEqual([row["native_id"] for row in ranked], ["W-ctu"])

    def test_survey_selection_reserves_available_publisher_families(self) -> None:
        """Verify publisher targets are represented before rank-only survey filling."""
        ranked = [
            {"native_id": "W1", "venue": "IEEE Access"},
            {"native_id": "W2", "publisher": "Elsevier"},
            {"native_id": "W3", "publisher": "IEEE"},
        ]
        selected, covered = publisher_diverse_surveys(
            ranked, 2, {"IEEE": ["IEEE"], "Elsevier": ["Elsevier"]},
        )
        self.assertEqual({row["native_id"] for row in selected}, {"W1", "W2"})
        self.assertEqual(covered, {"IEEE", "Elsevier"})

    def test_mixed_survey_and_release_work_remains_a_candidate(self) -> None:
        """Verify a survey section does not hide an explicit dataset-toolkit release."""
        mixed = {
            "title": "On Generating Network Traffic Datasets",
            "abstract": (
                "First, we present a comprehensive survey on intrusion detection datasets. "
                "Second, we present the Intrusion Detection Dataset Toolkit. "
                "ID2T is open source software and is made available to the community."
            ),
            "work_type": "article",
        }
        pure = {
            "title": "A comprehensive survey of intrusion detection datasets",
            "abstract": "We review and compare existing datasets.",
            "work_type": "review",
        }
        self.assertFalse(is_survey_work(mixed))
        self.assertTrue(is_survey_work(pure))

    def test_client_paces_every_request(self) -> None:
        """Verify OpenAlex calls respect the configured minimum interval."""
        client = OpenAlexClient(request_pause_seconds=.5)
        client.transport.last_request = 20.0
        response = Mock(status_code=200)
        response.json.return_value = {"results": []}
        with patch("radar.adapters.http.time.monotonic", side_effect=[20.2, 20.5, 20.7, 21.0]), patch(
            "radar.adapters.http.time.sleep") as sleep, patch.object(client.transport.session, "get", return_value=response):
            self.assertEqual(client.get("/works"), {"results": []})
            self.assertEqual(client.get("/works"), {"results": []})
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(abs(call.args[0] - .3) < 1e-8 for call in sleep.call_args_list))
        self.assertEqual(client.transport.last_request, 21.0)

    def test_release_roles_and_adjacent_tier(self) -> None:
        """Verify release roles map correctly to core, adjacent, and exclusions."""
        self.assertNotIn("confidence", RESPONSE_SCHEMA["properties"])
        self.assertNotIn("classification_confidence", CLASSIFICATION_COLUMNS)
        config = {"require_verified_quote": True}
        cases = (
            ("original_release", "core", "dataset", "accept"),
            ("versioned_release", "core", "dataset_and_generator", "accept"),
            ("generator_release", "core", "generator_framework", "accept"),
            ("original_release", "adjacent", "dataset", "accept_adjacent"),
            ("survey", "excluded", "unknown", "exclude"),
            ("dataset_analysis", "excluded", "unknown", "exclude"),
        )
        for role, tier, artifact_type, expected in cases:
            with self.subTest(role=role, tier=tier):
                result = normalize_response({
                    "paper_role": role, "relevance_tier": tier, "artifact_name": "Generic Trace Corpus" if "release" in role else "",
                    "artifact_display_name": "Generic Trace", "artifact_aliases": [], "artifact_type": artifact_type, "domain": "unknown", "data_form": "unknown",
                    "capture_environment": "unknown", "labeling": "unknown",
                    "direct_release_evidence": "release" in role, "primary_reason": "fixture",
                }, [{"source": "Abstract", "quote": "We release Generic Trace Corpus."}], config, "artifact-id")
                self.assertEqual(result["decision"], expected)

        unsupported_identity = normalize_response({
            "paper_role": "original_release", "relevance_tier": "core", "artifact_name": "CSE-CIC-IDS2018",
            "artifact_display_name": "CSE-CIC-IDS2018", "artifact_aliases": [], "artifact_type": "dataset",
            "domain": "general_network", "data_form": "flow", "capture_environment": "synthetic_lab",
            "labeling": "labeled", "direct_release_evidence": True, "primary_reason": "Introduces a dataset.",
        }, [{"source": "Abstract", "quote": "This paper produces a reliable intrusion detection dataset."}], config, "artifact-id")
        self.assertEqual(unsupported_identity["decision"], "uncertain")
        self.assertIn("artifact_identity_evidence_missing", unsupported_identity["classification_validation_flags"])

        mismatch = normalize_response({
            "paper_role": "generator_release", "relevance_tier": "core", "artifact_name": "Generic Toolkit",
            "artifact_display_name": "Generic Toolkit", "artifact_aliases": [], "artifact_type": "dataset", "domain": "unknown", "data_form": "unknown",
            "capture_environment": "unknown", "labeling": "unknown",
            "direct_release_evidence": True, "primary_reason": "Introduces a generator.",
            "evidence": [{"source": "Abstract", "quote": "introduces a generator"}],
        }, [], config, "artifact-id")
        self.assertEqual(mismatch["decision"], "uncertain")
        self.assertIn("role_artifact_type_mismatch", mismatch["classification_validation_flags"])
        self.assertIn("verified_evidence_missing", mismatch["classification_validation_flags"])
        self.assertIn("introduces a generator", mismatch["model_evidence_json"])

    def test_generic_alias_linking_uses_one_stable_artifact_id(self) -> None:
        """Verify generic spelling variants share a stable artifact identifier."""
        rows = [
            {"artifact_name": "Network Attack Trace Corpus", "artifact_aliases": "NAT Corpus", "artifact_id": "one", 'artifact_display_name': "Network Attack Trace Corpus", 'resource_name': "Network Attack Trace Corpus"},
            {"artifact_name": "Network-Attack Trace Corpus", "artifact_aliases": "", "artifact_id": "two", 'artifact_display_name': "Network-Attack Trace Corpus", 'resource_name': "Network-Attack Trace Corpus"},
        ]
        canonicalize_artifact_ids(rows)
        self.assertEqual(rows[0]["artifact_id"], rows[1]["artifact_id"])

    def test_derived_resource_prefix_does_not_merge_artifact_families(self) -> None:
        """Do not treat a derived NF resource as a spelling variant of its source dataset."""
        rows = [
            {"artifact_name": "CSE-CIC-IDS2018", "artifact_aliases": "", "artifact_id": "base", 'artifact_display_name': "CSE-CIC-IDS2018", 'resource_name': "CSE-CIC-IDS2018"},
            {"artifact_name": "NF-CSE-CIC-IDS2018", "artifact_aliases": "", "artifact_id": "derived", 'artifact_display_name': "NF-CSE-CIC-IDS2018", 'resource_name': "NF-CSE-CIC-IDS2018"},
        ]
        canonicalize_artifact_ids(rows)
        self.assertNotEqual(rows[0]["artifact_id"], rows[1]["artifact_id"])

    def test_resource_name_requires_the_explicit_current_field(self) -> None:
        """Verify processed rows expose a concise resource name independently of the paper title."""
        current = {"artifact_name": "CICIDS2017", "artifact_display_name": "CICIDS2017"}
        self.assertEqual(ensure_resource_name(current), "CICIDS2017")
        self.assertEqual(current["resource_name"], "CICIDS2017")
        with self.assertRaisesRegex(ValueError, 'artifact_display_name'):
            ensure_resource_name({"artifact_name": "CICIDS2017"})

    def test_abstract_reconstruction_and_metadata(self) -> None:
        """Verify abstract reconstruction and normalized source metadata."""
        self.assertEqual(reconstruct_abstract({"world": [1], "hello": [0]}), "hello world")
        work = normalize_work({"id": "https://openalex.org/W1", "title": "Title", "cited_by_count": 9, "open_access": {"is_oa": True}})
        self.assertEqual(work["cited_by_count"], 9)
        self.assertEqual(work["is_oa"], "true")

    def test_survey_references_are_reserved_batched_and_provenanced(self) -> None:
        """Verify survey snowballing reserves, batches, and traces references."""
        def raw(work_id: str, title: str, work_type: str = "article", references: list[str] | None = None) -> dict[str, Any]:
            """Build a minimal raw OpenAlex work fixture."""
            return {
                "id": f"https://openalex.org/{work_id}", "title": title, "type": work_type,
                "publication_year": 2020, "referenced_works": references or [], "open_access": {},
            }

        class Client:
            def __init__(self) -> None:
                """Initialize storage for observed reference batches."""
                self.batches: list[list[str]] = []

            def search(self, query: str, _per_page: int, _max_pages: int, sort="best_match", order="desc"):
                """Yield survey or direct-search fixture works by query type."""
                if "survey" in query:
                    yield raw("W1", "A survey of intrusion detection datasets", "review", ["https://openalex.org/W2", "https://openalex.org/W3"])
                else:
                    yield raw("W4", "A direct candidate")
                    yield raw("W5", "Another direct candidate")

            def works_by_ids(self, ids: list[str]):
                """Resolve fixture references while recording batch membership."""
                self.batches.append(ids)
                for value in ids:
                    yield raw(value.rsplit("/", 1)[-1], f"Referenced release {value.rsplit('/', 1)[-1]}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = Client()
            settings = {
                "per_page": 10, "max_pages_per_query": 1, "max_surveys": 1,
                "max_reference_works_per_survey": 10, "max_survey_reference_candidates_total": 10,
                "survey_reference_fraction": .4, "progress_every": 10, "retrieval_minimum_score": 0,
            }
            profile = {"survey_queries": ["dataset survey"], "queries": [{"name": "direct", "search": "intrusion dataset"}]}
            with PipelineCache(root / "cache.sqlite3") as cache:
                counts = collect(settings, profile, cache, limit=5, client=fake)  # type: ignore[arg-type]
                referenced = cache.get_record("openalex.work", "https://openalex.org/W2")
                active = cache.get_record("openalex.collection", "latest")
                survey_rows = export_survey_sources(cache, root / "surveys.csv", "run")
            self.assertEqual(counts["discovered"], 4)
            self.assertEqual(counts["auxiliary_cached_works"], 1)
            self.assertEqual(counts["survey_sources"], 1)
            self.assertEqual(counts["survey_reference_works"], 2)
            self.assertEqual(len(fake.batches), 1)
            self.assertIn("survey_reference:W1", referenced["discovery_relations"])
            self.assertNotIn("https://openalex.org/W1", active["work_ids"])
            self.assertEqual([row["openalex_id"] for row in survey_rows], ["https://openalex.org/W1"])

    def test_backward_snowballing_repeats_until_convergence(self) -> None:
        """Verify a relevant reference becomes the next frontier and exposes its references."""
        def raw(work_id: str, title: str, work_type: str = "article", references: list[str] | None = None) -> dict[str, Any]:
            """Build one raw work in the iterative reference-chain fixture."""
            return {
                "id": f"https://openalex.org/{work_id}", "title": title, "type": work_type,
                "publication_year": 2020, "referenced_works": references or [], "open_access": {},
            }

        class Client:
            authenticated = True
            def __init__(self) -> None:
                """Initialize the observed reference batches."""
                self.batches: list[list[str]] = []
            def search(self, query: str, _per_page: int, _max_pages: int, sort="best_match", order="desc"):
                """Yield one survey or one direct seed according to the query."""
                if "survey" in query:
                    yield raw("W1", "Survey of intrusion datasets", "review", ["https://openalex.org/W2"])
                else:
                    yield raw("W4", "Direct intrusion dataset paper")
            def works_by_ids(self, ids: list[str]):
                """Resolve a two-level chain while recording each batch."""
                self.batches.append(ids)
                for value in ids:
                    key = value.rsplit("/", 1)[-1]
                    refs = ["https://openalex.org/W3"] if key == "W2" else []
                    yield raw(key, f"Relevant intrusion dataset {key}", references=refs)

        with tempfile.TemporaryDirectory() as directory:
            fake = Client()
            with PipelineCache(Path(directory) / "cache.sqlite3") as cache:
                counts = collect({
                    "per_page": 10, "max_pages_per_query": 1, "max_surveys": 1,
                    "max_reference_works_per_survey": 10, "max_survey_reference_candidates_total": 10,
                    "survey_reference_fraction": .75, "reference_sample_size": 10,
                    "snowball_start_papers": 1, "snowball_max_depth": 10,
                    "retrieval_minimum_score": 0,
                }, {"survey_queries": ["dataset survey"], "queries": [{"name": "direct", "search": "intrusion dataset"}]},
                    cache, limit=4, client=fake)  # type: ignore[arg-type]
                latest = cache.get_record("openalex.collection", "latest")
                depth_two = cache.get_record("openalex.work", "https://openalex.org/W3")
            self.assertEqual(len(fake.batches), 2)
            self.assertEqual(counts["snowball_reached_depth"], 2)
            self.assertEqual(latest["snowball_reached_depth"], 2)
            self.assertEqual(depth_two["snowball_depth"], 2)

    def test_prefilter_enriches_missing_abstract_before_scoring(self) -> None:
        """Verify an opaque survey reference gains evidence before prefilter scoring."""
        class Enricher:
            def __init__(self) -> None:
                """Initialize the enrichment call counter."""
                self.calls = 0

            def enrich(self, _work: dict[str, Any]) -> dict[str, str]:
                """Return direct dataset-release evidence."""
                self.calls += 1
                return {
                    "abstract": "We introduce a new public intrusion detection dataset containing labeled botnet network traffic.",
                    "abstract_source": "Publisher lookup",
                }

        settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        profile = load_term_profile(settings, "openalex")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            enricher = Enricher()
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("openalex.work", "W-ctu", {
                    "work": {
                        "openalex_id": "W-ctu", "title": "An empirical comparison of botnet detection methods",
                        "abstract": "", "cited_by_count": 937,
                    },
                    "discovery_relations": [
                        "survey_reference", "survey_reference:W1", "survey_reference:W2",
                    ],
                }, "2026-01-01T00:00:00+00:00")
                rows = prefilter(
                    settings.section("openalex"), profile, fixture_collection(cache, 'openalex'), root / "candidates.csv", "run", enricher=enricher,
                )
                cached = cache.get_record("openalex.work", "W-ctu")
            self.assertEqual(enricher.calls, 1)
            self.assertEqual(rows[0]["prefilter_band"], "candidate")
            self.assertEqual(rows[0]["abstract_source"], "Publisher lookup")
            self.assertTrue(cached["work"]["abstract"])

    def test_survey_consensus_reaches_classification_if_enrichment_fails(self) -> None:
        """Verify two survey citations prevent an opaque release paper from being rejected."""
        class EmptyEnricher:
            def enrich(self, _work: dict[str, Any]) -> dict[str, str]:
                """Simulate unavailable external evidence."""
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("openalex.work", "W-opaque", {
                    "work": {"openalex_id": "W-opaque", "title": "An empirical comparison", "cited_by_count": 50},
                    "discovery_relations": ["survey_reference:W1", "survey_reference:W2"],
                }, "2026-01-01T00:00:00+00:00")
                rows = prefilter(
                    {
                        "candidate_threshold": 7, "borderline_threshold": 4, "minimum_citations": 50,
                        "survey_consensus_threshold": 2,
                    },
                    {"positive": {}}, fixture_collection(cache, 'openalex'), root / "candidates.csv", "run", enricher=EmptyEnricher(),
                )
            self.assertEqual(rows[0]["prefilter_band"], "borderline")
            self.assertIn("survey_consensus:2", rows[0]["prefilter_signals"])

    def test_classification_enriches_missing_abstract_and_candidate_csv(self) -> None:
        """Verify missing evidence is cached and changes the decision fingerprint."""
        class Enricher:
            def enrich(self, _work: dict[str, Any]) -> dict[str, str]:
                """Return deterministic external abstract evidence."""
                return {
                    "abstract": "We introduce a new labeled intrusion detection dataset named NAT Corpus containing network attack packet traces.",
                    "abstract_source": "Semantic Scholar",
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified = root / "candidates.csv", root / "classified.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update({
                "openalex_id": "W-missing", "title": "Opaque comparison title", "cited_by_count": "50",
                "prefilter_score": "7", "prefilter_band": "candidate", "evidence_hash": "old-hash",
            })
            atomic_write_csv(candidates, [row], CANDIDATE_COLUMNS)
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("openalex.work", "W-missing", {
                    "work": {"openalex_id": "W-missing", "title": "Opaque comparison title", "abstract": ""},
                    "discovery_relations": ["survey_reference"],
                }, "2026-01-01T00:00:00+00:00")
                decisions = classify(
                    candidates, classified, cache, {"require_verified_quote": True},
                    generator=FakePaperModel(), enricher=Enricher(), openalex_config={"enrich_missing_abstracts": True},
                    prompt_config=load_prompt_profile(load_settings(ROOT / "config/studies/ids_datasets/study.toml"), "openalex"),
                )
                cached_work = cache.get_record("openalex.work", "W-missing")["work"]
            with candidates.open(newline="", encoding="utf-8") as handle:
                enriched_row = next(__import__("csv").DictReader(handle))
            self.assertEqual(decisions[0]["decision"], "accept")
            self.assertEqual(cached_work["abstract_source"], "Semantic Scholar")
            self.assertEqual(enriched_row["abstract_source"], "Semantic Scholar")
            self.assertNotEqual(enriched_row["evidence_hash"], "old-hash")

    def test_uncertain_classification_retries_with_full_text(self) -> None:
        """Verify an existing abstract is supplemented with full text after uncertainty."""
        class Model:
            model = "fixture-model"
            def __init__(self) -> None:
                """Initialize the model call count."""
                self.calls = 0
            def generate(self, prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
                """Make only initial run three uncertain; accept the restarted round."""
                self.calls += 1
                expanded = "decisive full text release evidence" in prompt
                accepted = expanded or self.calls != 3
                return {
                    "paper_role": "original_release" if accepted else "uncertain",
                    "relevance_tier": "core" if accepted else "uncertain",
                    "artifact_name": "Fixture Dataset" if accepted else "",
                    "artifact_display_name": "Fixture Dataset" if accepted else "",
                    "artifact_aliases": [], "artifact_type": "dataset" if accepted else "unknown",
                    "domain": "unknown", "data_form": "unknown", "capture_environment": "unknown",
                    "labeling": "unknown", "direct_release_evidence": accepted,
                    "primary_reason": "fixture",
                    "evidence": [{"source": "Full text" if expanded else "Abstract",
                                  "quote": "decisive full text release evidence for Fixture Dataset" if expanded else "ambiguous abstract naming Fixture Dataset"}],
                }

        class Enricher:
            def enrich(self, _work: dict[str, Any]) -> dict[str, str]:
                """Return the decisive open full-text fixture."""
                return {"full_text": "decisive full text release evidence for Fixture Dataset", "full_text_url": "https://example.invalid/paper.pdf"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified = root / "candidates.csv", root / "classified.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update({"openalex_id": "W-full", "title": "Ambiguous paper", "abstract": "ambiguous abstract naming Fixture Dataset",
                        "prefilter_band": "candidate", "prefilter_score": "7", "evidence_hash": "old"})
            atomic_write_csv(candidates, [row], CANDIDATE_COLUMNS)
            model = Model()
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("openalex.work", "W-full", {
                    "work": {"openalex_id": "W-full", "title": "Ambiguous paper", "abstract": "ambiguous abstract naming Fixture Dataset"},
                }, "now")
                decisions = classify(candidates, classified, cache, {"require_verified_quote": True},
                                     generator=model, enricher=Enricher(),
                                     openalex_config={"enrich_uncertain_with_full_text": True},
                                     prompt_config=load_prompt_profile(load_settings(ROOT / "config/studies/ids_datasets/study.toml"), "openalex"),
                                     consecutive_runs=5, consensus_threshold=.66)
            self.assertEqual(model.calls, 8, decisions)
            self.assertEqual(decisions[0]["decision"], "accept")
            self.assertNotIn("uncertain", decisions[0]["ensemble_model_decisions"])
            self.assertEqual(decisions[0]["evidence_expansion_required"], "true")
            self.assertEqual(decisions[0]["evidence_expansion_changed"], "true")
            self.assertEqual(decisions[0]["evidence_expansion_improved"], "true")
            self.assertEqual(decisions[0]["evidence_expansion_resolved"], "false")

    def test_classification_errors_are_not_cached_and_are_retried(self) -> None:
        """Verify a malformed model response cannot become a terminal cached decision."""
        class RecoveringModel:
            model = "fixture-model"

            def __init__(self) -> None:
                """Fail once before returning a valid release classification."""
                self.calls = 0

            def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
                """Raise on the first call and delegate the successful retry."""
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("malformed JSON")
                return FakePaperModel().generate(prompt, schema)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified = root / "candidates.csv", root / "classified.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update({
                "openalex_id": "W-retry", "title": "A New Intrusion Detection Dataset",
                "abstract": "We introduce a new labeled intrusion detection dataset named NAT Corpus containing network attack packet traces.",
                "cited_by_count": "50", "prefilter_score": "7", "prefilter_band": "candidate",
                "evidence_hash": "retry-evidence",
            })
            atomic_write_csv(candidates, [row], CANDIDATE_COLUMNS)
            model = RecoveringModel()
            prompts = load_prompt_profile(load_settings(ROOT / "config/studies/ids_datasets/study.toml"), "openalex")
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("openalex.work", "W-retry", {
                    "work": {
                        "openalex_id": "W-retry", "title": row["title"], "abstract": row["abstract"],
                    },
                }, "2026-01-01T00:00:00+00:00")
                first = classify(
                    candidates, classified, cache, {"require_verified_quote": True},
                    generator=model, prompt_config=prompts,
                )
                self.assertEqual(first[0]["decision"], "error")
                self.assertIsNone(cache.get_decision(
                    "openalex.classification", "W-retry", first[0]["profile_hash"], "retry-evidence",
                ))
                second = classify(
                    candidates, classified, cache, {"require_verified_quote": True},
                    generator=model, prompt_config=prompts,
                )
            self.assertEqual(model.calls, 2)
            self.assertEqual(second[0]["decision"], "accept")

    def test_export_preserves_accepted_generators_in_both_catalogues(self) -> None:
        """Keep generator releases alongside datasets without another inclusion gate."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for decision in ("accept", "accept_adjacent"):
                for artifact_type in ("dataset", "dataset_and_generator", "generator_framework"):
                    identity = f"{decision}-{artifact_type}"
                    rows.append({
                        "openalex_id": identity, "artifact_id": identity,
                        "artifact_name": identity, "artifact_display_name": identity,
                        "resource_name": identity, "title": identity,
                        "publication_year": "2024", "decision": decision,
                        "artifact_type": artifact_type,
                        "paper_role": "generator_release" if artifact_type == "generator_framework" else "original_release",
                        "run_id": "fixture", "evidence_json": "[]",
                        "publication_family_id": identity,
                        "canonical_openalex_id": identity,
                        "paper_relationship": "canonical_base",
                     'classifier_supported_fields': '*'})
            rows.extend([
                dict(rows[-1], openalex_id="excluded-generator", decision="exclude"),
                dict(rows[-1], openalex_id="uncertain-generator", decision="uncertain"),
            ])
            reconciled = root / "reconciled.csv"
            atomic_write_csv(reconciled, rows, CATALOGUE_COLUMNS)
            main, adjacent = export(reconciled, root / "base.csv", root / "adjacent.csv")
            self.assertEqual(len(main), 3)
            self.assertEqual(len(adjacent), 3)
            self.assertEqual(
                {row["openalex_id"] for row in main + adjacent},
                {row["openalex_id"] for row in rows if row["decision"] in {"accept", "accept_adjacent"}},
            )

    def test_work_level_classification_and_export(self) -> None:
        """Verify work-level adjudication excludes downstream applications."""
        settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        profile = load_term_profile(settings, "openalex")
        prompts = load_prompt_profile(settings, "openalex")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified, reconciled = root / "candidates.csv", root / "classified.csv", root / "reconciled.csv"
            main, adjacent = root / "base.csv", root / "adjacent.csv"
            with PipelineCache(root / "cache.sqlite3") as cache:
                base = {
                    "openalex_id": "https://openalex.org/W1", "doi": "10.1/example", "title": "A New Intrusion Detection Dataset",
                    "abstract": "We introduce a new labeled intrusion detection dataset named NAT Corpus containing network attack packet traces.",
                    "authors": "A Researcher", "venue": "Security Venue", "publication_year": 2024, "publication_date": "2024-01-01",
                    "work_type": "article", "primary_url": "https://doi.org/10.1/example", "pdf_url": "", "is_oa": "true", "cited_by_count": 50,
                }
                downstream = dict(base, openalex_id="https://openalex.org/W2", doi="10.1/downstream", title="Improving a classifier", abstract="We report classification accuracy for network traffic intrusion detection using a publicly available existing dataset.")
                cache.put_record("openalex.work", base["openalex_id"], {"work": base, "discovery_relations": ["direct_search"], "matched_queries": ["intrusion detection dataset"]}, "2026-01-01T00:00:00+00:00")
                cache.put_record("openalex.work", downstream["openalex_id"], {"work": downstream, "discovery_relations": ["direct_search"], "matched_queries": ["intrusion detection dataset"]}, "2026-01-01T00:00:00+00:00")
                rows = prefilter(settings.section("openalex"), profile, fixture_collection(cache, 'openalex'), candidates, "fixture-run")
                self.assertEqual(len(rows), 2)
                model = FakePaperModel()
                decisions = classify(candidates, classified, cache, settings.section("ollama"), generator=model, prompt_config=prompts, consecutive_runs=3, consensus_threshold=.66)
                self.assertEqual(model.calls, 6)
                self.assertEqual({row["decision"] for row in decisions}, {"accept", "exclude"})
                self.assertTrue(all(row["consensus_field_audit"] for row in decisions))
                accepted_decision = next(row for row in decisions if row["decision"] == "accept")
                self.assertEqual(accepted_decision["classification_validation_flags"], "")
                self.assertEqual(accepted_decision["artifact_display_name"], "NAT Corpus")
                self.assertIn("introduce a new labeled", accepted_decision["model_evidence_json"])
                reconcile(classified, reconciled, cache, settings.section("ollama"), settings.section("openalex"), dry_run=True)
                accepted, adjacent_rows = export(reconciled, main, adjacent)
                self.assertEqual(len(accepted), 1)
                self.assertEqual(adjacent_rows, [])
                self.assertEqual(accepted[0]["openalex_id"], "https://openalex.org/W1")
                self.assertTrue(accepted[0]["artifact_id"])
                self.assertEqual(accepted[0]["resource_name"], "NAT Corpus")
                self.assertEqual(accepted[0]["paper_relationship"], "canonical_base")

    def test_prefilter_uses_latest_collection_without_deleting_history(self) -> None:
        """Verify stale cached works do not leak into the latest candidate pool."""
        profile = {"positive": {"scope": {"weight": 3, "terms": ["intrusion detection"]}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with PipelineCache(root / "cache.sqlite3") as cache:
                for work_id in ("W-old", "W-new", "W-survey"):
                    work = {"openalex_id": work_id, "title": "An intrusion detection dataset"}
                    if work_id == "W-survey":
                        work.update({"title": "A survey of intrusion detection datasets", "work_type": "review"})
                    cache.put_record("openalex.work", work_id, {
                        "work": work,
                        "discovery_relations": ["direct_search"], "matched_queries": ["query"],
                    }, "2026-01-01T00:00:00+00:00")
                cache.put_record("openalex.collection", "latest", {
                    "work_ids": ["W-new"], "survey_ids": ["W-survey"], "selected_at": "now",
                }, "now")
                rows = prefilter({"candidate_threshold": 3, "borderline_threshold": 1}, profile, fixture_collection(cache, 'openalex'), root / "candidates.csv", "run")
                self.assertIsNotNone(cache.get_record("openalex.work", "W-old"))
            self.assertEqual([row["openalex_id"] for row in rows], ["W-new"])

    def test_prefilter_enforces_minimum_citations(self) -> None:
        """Verify papers below the configured citation floor never become candidates."""
        profile = {"positive": {"scope": {"weight": 3, "terms": ["intrusion detection"]}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with PipelineCache(root / "cache.sqlite3") as cache:
                for work_id, citations in (("W-low", 49), ("W-boundary", 50)):
                    cache.put_record("openalex.work", work_id, {
                        "work": {"openalex_id": work_id, "title": "An intrusion detection dataset", "cited_by_count": citations},
                        "discovery_relations": ["direct_search"], "matched_queries": ["query"],
                    }, "2026-01-01T00:00:00+00:00")
                rows = prefilter(
                    {"candidate_threshold": 3, "borderline_threshold": 1, "minimum_citations": 50},
                    profile, fixture_collection(cache, 'openalex'), root / "candidates.csv", "run",
                )
            self.assertEqual([row["openalex_id"] for row in rows], ["W-boundary"])

    def test_paper_reconciliation_is_order_independent(self) -> None:
        """Verify base-paper selection is independent of input row order."""
        settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def row(work_id: str, title: str, year: str, role: str) -> dict[str, str]:
                """Build one accepted paper classification fixture."""
                value = {column: "" for column in CLASSIFICATION_COLUMNS}
                value.update({
                    "openalex_id": work_id, "title": title, "publication_year": year,
                    "decision": "accept", "paper_role": role, "relevance_tier": "core",
                    "artifact_id": "initial", "artifact_name": "Generic Network Trace Corpus",
                    "artifact_type": "dataset", "direct_release_evidence": "true",
                 'classifier_supported_fields': '*', 'artifact_display_name': "Generic Network Trace Corpus", 'resource_name': "Generic Network Trace Corpus"})
                return value
            source_rows = [
                row("https://openalex.org/W2", "Generic Network Trace Corpus: Extended Release", "2022", "versioned_release"),
                row("https://openalex.org/W1", "Introducing the Generic Network Trace Corpus", "2019", "original_release"),
            ]
            selections = []
            with PipelineCache(root / "cache.sqlite3") as cache:
                for index, ordered in enumerate((source_rows, list(reversed(source_rows)))):
                    source, output = root / f"source-{index}.csv", root / f"output-{index}.csv"
                    atomic_write_csv(source, ordered, CLASSIFICATION_COLUMNS)
                    result = reconcile(source, output, cache, settings.section("ollama"), settings.section("openalex"), dry_run=True)
                    accepted = [item for item in result if item.get("decision") == "accept"]
                    selections.append({item["canonical_openalex_id"] for item in accepted})
                    relationships = {item["openalex_id"]: item["paper_relationship"] for item in accepted}
                    self.assertEqual(relationships["https://openalex.org/W1"], "canonical_base")
                    self.assertEqual(relationships["https://openalex.org/W2"], "versioned_release")
            self.assertEqual(selections, [{"https://openalex.org/W1"}, {"https://openalex.org/W1"}])

    def test_ambiguous_release_pair_is_adjudicated_from_evidence(self) -> None:
        """Use evidence adjudication rather than fixed title phrases for close release candidates."""
        class ReleaseAdjudicator:
            model = "fixture-reconciler"

            def __init__(self) -> None:
                """Capture the prompt supplied to the fixture adjudicator."""
                self.prompt = ""

            def generate(self, prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
                """Select the release paper based on its supplied evidence."""
                self.prompt = prompt
                return {
                    "canonical_openalex_id": "https://openalex.org/W-release",
                    "reason": "The verified evidence identifies this work as the resource release.",
                }

        settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for work_id, title, year in (
                ("https://openalex.org/W-old", "Towards a Reliable Intrusion Detection Benchmark Dataset", "2017"),
                ("https://openalex.org/W-release", "Toward Generating a New Intrusion Detection Dataset", "2018"),
            ):
                value = {column: "" for column in CLASSIFICATION_COLUMNS}
                value.update({
                    "openalex_id": work_id, "title": title, "publication_year": year,
                    "decision": "accept", "paper_role": "original_release", "relevance_tier": "core",
                    "artifact_id": "initial", "artifact_name": "CICIDS2017",
                    "artifact_display_name": "CICIDS2017", "artifact_type": "dataset",
                    "direct_release_evidence": "true",
                    "evidence_json": "release evidence naming CICIDS2017",
                 'classifier_supported_fields': '*'})
                rows.append(value)
            source, output = root / "source.csv", root / "output.csv"
            atomic_write_csv(source, rows, CLASSIFICATION_COLUMNS)
            model = ReleaseAdjudicator()
            with PipelineCache(root / "cache.sqlite3") as cache:
                result = reconcile(
                    source, output, cache, settings.section("ollama"), settings.section("openalex"), generator=model,
                    prompt_config=load_prompt_profile(settings, "openalex"),
                )
            self.assertEqual({item["canonical_openalex_id"] for item in result}, {"https://openalex.org/W-release"})
            self.assertTrue(all(item["canonical_resolution"] == "llm_group_adjudication" for item in result))
            self.assertIn("verified_release_evidence", model.prompt)

    def test_raw_citations_are_a_configurable_capped_canonical_signal(self) -> None:
        """Let citations resolve close candidates without age normalization or overriding release evidence."""
        settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for work_id, title, citations in (
                ("https://openalex.org/W1", "Initial collection methodology", "5"),
                ("https://openalex.org/W2", "Official benchmark release", "5000"),
            ):
                value = {column: "" for column in CLASSIFICATION_COLUMNS}
                value.update({
                    "openalex_id": work_id, "title": title,
                    "publication_year": "2020", "cited_by_count": citations,
                    "decision": "accept", "paper_role": "original_release", "relevance_tier": "core",
                    "artifact_id": "initial", "artifact_name": "Generic Trace Corpus",
                    "artifact_type": "dataset", "direct_release_evidence": "true",
                 'classifier_supported_fields': '*', 'artifact_display_name': "Generic Trace Corpus", 'resource_name': "Generic Trace Corpus"})
                rows.append(value)
            source, output = root / "source.csv", root / "output.csv"
            atomic_write_csv(source, rows, CLASSIFICATION_COLUMNS)
            with PipelineCache(root / "cache.sqlite3") as cache:
                result = reconcile(
                    source, output, cache, settings.section("ollama"), settings.section("openalex"), dry_run=True,
                )
            self.assertEqual({item["canonical_openalex_id"] for item in result}, {"https://openalex.org/W2"})
            self.assertIn("citations=2.0 from 5000 citations", result[0]["canonical_reason"])

    def test_published_record_wins_over_duplicate_preprint_manifestation(self) -> None:
        """Verify a published record is canonical over its duplicate preprint."""
        settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for work_id, work_type, year, doi, venue in (
                ("https://openalex.org/W10", "preprint", "2018", "", ""),
                ("https://openalex.org/W11", "article", "2020", "10.1/published", "Security Venue"),
            ):
                value = {column: "" for column in CLASSIFICATION_COLUMNS}
                value.update({
                    "openalex_id": work_id, "title": "Introducing a Generic Attack Trace Corpus",
                    "publication_year": year, "work_type": work_type, "doi": doi, "venue": venue,
                    "decision": "accept", "paper_role": "original_release", "relevance_tier": "core",
                    "artifact_id": "initial", "artifact_name": "Generic Attack Trace Corpus",
                    "artifact_type": "dataset", "direct_release_evidence": "true",
                 'classifier_supported_fields': '*', 'artifact_display_name': "Generic Attack Trace Corpus", 'resource_name': "Generic Attack Trace Corpus"})
                rows.append(value)
            source, output = root / "source.csv", root / "output.csv"
            atomic_write_csv(source, rows, CLASSIFICATION_COLUMNS)
            with PipelineCache(root / "cache.sqlite3") as cache:
                result = reconcile(source, output, cache, settings.section("ollama"), settings.section("openalex"), dry_run=True)
            relationships = {item["openalex_id"]: item["paper_relationship"] for item in result}
            self.assertEqual({item["canonical_openalex_id"] for item in result}, {"https://openalex.org/W11"})
            self.assertEqual(relationships["https://openalex.org/W10"], "alternate_manifestation")


if __name__ == "__main__":
    unittest.main()
