from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import load_settings, load_term_profile
from ids_meta.openalex.collect import OpenAlexClient, collect, normalize_work, reconstruct_abstract
from ids_meta.openalex.pipeline import classify, export, prefilter
from ids_meta.openalex.reconcile import reconcile
from ids_meta.openalex.entities import canonicalize_artifact_ids
from ids_meta.openalex.schema import CLASSIFICATION_COLUMNS, normalize_response
from ids_meta.common.csvio import atomic_write_csv


ROOT = Path(__file__).resolve().parents[1]


class FakePaperModel:
    model = "fixture-model"

    def generate(self, prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic release or downstream-paper classifications."""
        downstream = "classification accuracy" in prompt
        if downstream:
            return {
                "paper_role": "downstream_application", "relevance_tier": "excluded", "artifact_name": "",
                "artifact_aliases": [], "artifact_type": "unknown", "domain": "unknown", "data_form": "unknown",
                "capture_environment": "unknown", "labeling": "unknown", "confidence": .95,
                "direct_release_evidence": False, "primary_reason": "Applies existing data.",
                "evidence": [{"source": "Abstract", "quote": "classification accuracy"}],
            }
        return {
            "paper_role": "original_release", "relevance_tier": "core", "artifact_name": "Network Attack Trace Corpus",
            "artifact_aliases": ["NAT Corpus"], "artifact_type": "dataset", "domain": "general_network",
            "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled", "confidence": .96,
            "direct_release_evidence": True, "primary_reason": "Introduces and releases attack traces.",
            "evidence": [{"source": "Abstract", "quote": "introduce a new labeled intrusion detection dataset"}],
        }


class OpenAlexPipelineTests(unittest.TestCase):
    def test_client_paces_every_request(self) -> None:
        """Verify OpenAlex calls respect the configured minimum interval."""
        client = OpenAlexClient(request_pause_seconds=.5)
        client._last_request_at = 20.0
        with patch("ids_meta.openalex.collect.time.monotonic", side_effect=[20.2, 20.5]), patch("ids_meta.openalex.collect.time.sleep") as sleep:
            client._wait_before_request()
        self.assertEqual(sleep.call_count, 1)
        self.assertAlmostEqual(sleep.call_args.args[0], .3)
        self.assertEqual(client._last_request_at, 20.5)

    def test_release_roles_and_adjacent_tier(self) -> None:
        """Verify release roles map correctly to core, adjacent, and exclusions."""
        config = {"accept_confidence": .8, "exclude_confidence": .75, "require_verified_quote": True}
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
                    "artifact_aliases": [], "artifact_type": artifact_type, "domain": "unknown", "data_form": "unknown",
                    "capture_environment": "unknown", "labeling": "unknown", "confidence": .95,
                    "direct_release_evidence": "release" in role, "primary_reason": "fixture",
                }, [{"source": "Abstract", "quote": "fixture"}], config, "artifact-id")
                self.assertEqual(result["decision"], expected)

    def test_generic_alias_linking_uses_one_stable_artifact_id(self) -> None:
        """Verify generic spelling variants share a stable artifact identifier."""
        rows = [
            {"artifact_name": "Network Attack Trace Corpus", "artifact_aliases": "NAT Corpus", "artifact_id": "one"},
            {"artifact_name": "Network-Attack Trace Corpus", "artifact_aliases": "", "artifact_id": "two"},
        ]
        canonicalize_artifact_ids(rows)
        self.assertEqual(rows[0]["artifact_id"], rows[1]["artifact_id"])

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

            def search(self, query: str, _per_page: int, _max_pages: int):
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
                "survey_reference_fraction": .4, "progress_every": 10,
            }
            profile = {"survey_queries": ["dataset survey"], "direct_queries": ["intrusion dataset"]}
            with PipelineCache(root / "cache.sqlite3") as cache:
                counts = collect(settings, profile, cache, limit=5, client=fake)  # type: ignore[arg-type]
                referenced = cache.get_record("openalex.work", "https://openalex.org/W2")
            self.assertEqual(counts["discovered"], 5)
            self.assertEqual(counts["survey_sources"], 1)
            self.assertEqual(counts["survey_reference_works"], 2)
            self.assertEqual(len(fake.batches), 1)
            self.assertIn("survey_reference:W1", referenced["discovery_relations"])

    def test_work_level_classification_and_export(self) -> None:
        """Verify work-level adjudication excludes downstream applications."""
        settings = load_settings(ROOT / "config/default.toml")
        profile = load_term_profile(settings, "openalex")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified, reconciled = root / "candidates.csv", root / "classified.csv", root / "reconciled.csv"
            main, adjacent = root / "base.csv", root / "adjacent.csv"
            with PipelineCache(root / "cache.sqlite3") as cache:
                base = {
                    "openalex_id": "https://openalex.org/W1", "doi": "10.1/example", "title": "A New Intrusion Detection Dataset",
                    "abstract": "We introduce a new labeled intrusion detection dataset containing network attack packet traces.",
                    "authors": "A Researcher", "venue": "Security Venue", "publication_year": 2024, "publication_date": "2024-01-01",
                    "work_type": "article", "primary_url": "https://doi.org/10.1/example", "pdf_url": "", "is_oa": "true", "cited_by_count": 0,
                }
                downstream = dict(base, openalex_id="https://openalex.org/W2", doi="10.1/downstream", title="Improving a classifier", abstract="We report classification accuracy for network traffic intrusion detection using an existing dataset.")
                cache.put_record("openalex.work", base["openalex_id"], {"work": base, "discovery_relations": ["direct_search"], "matched_queries": ["intrusion detection dataset"]}, "2026-01-01T00:00:00+00:00")
                cache.put_record("openalex.work", downstream["openalex_id"], {"work": downstream, "discovery_relations": ["direct_search"], "matched_queries": ["intrusion detection dataset"]}, "2026-01-01T00:00:00+00:00")
                rows = prefilter(settings.section("openalex"), profile, cache, candidates, "fixture-run")
                self.assertEqual(len(rows), 2)
                decisions = classify(candidates, classified, cache, settings.section("ollama"), generator=FakePaperModel())
                self.assertEqual({row["decision"] for row in decisions}, {"accept", "exclude"})
                reconcile(classified, reconciled, cache, settings.section("ollama"), settings.section("openalex"), dry_run=True)
                accepted, adjacent_rows = export(reconciled, main, adjacent)
                self.assertEqual(len(accepted), 1)
                self.assertEqual(adjacent_rows, [])
                self.assertEqual(accepted[0]["openalex_id"], "https://openalex.org/W1")
                self.assertTrue(accepted[0]["artifact_id"])
                self.assertEqual(accepted[0]["paper_relationship"], "canonical_base")

    def test_paper_reconciliation_is_order_independent(self) -> None:
        """Verify base-paper selection is independent of input row order."""
        settings = load_settings(ROOT / "config/default.toml")
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
                })
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

    def test_published_record_wins_over_duplicate_preprint_manifestation(self) -> None:
        """Verify a published record is canonical over its duplicate preprint."""
        settings = load_settings(ROOT / "config/default.toml")
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
                })
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
