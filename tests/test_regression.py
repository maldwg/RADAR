from __future__ import annotations

import unittest

from ids_meta.common.ensemble import consensus_classification
from ids_meta.common.regression import evaluate_github_regression, evaluate_openalex_regression
from ids_meta.openalex.collect import is_survey_work, sample_query_results


class RegressionEvaluationTests(unittest.TestCase):
    """Exercise source-neutral metrics and conservative repeat voting."""

    def test_github_dict_regression_metrics(self) -> None:
        """Compare GitHub decisions and roles with a hand-rated dictionary."""
        expected = {"1": {"decision": "accept", "role": "native_detector"}, "2": {"decision": "exclude", "role": "demo"}}
        actual = {"1": {"decision": "accept", "role": "native_detector"}, "2": {"decision": "uncertain", "role": "demo"}}
        result = evaluate_github_regression(expected, actual)
        self.assertEqual(result.metrics["accuracy"], .5)
        self.assertEqual(result.metrics["detection_rate"], 1.0)
        self.assertEqual(result.metrics["classification_accuracy"], 1.0)

    def test_openalex_dict_regression_metrics(self) -> None:
        """Compare OpenAlex adjacent inclusion and paper-role classification."""
        expected = {"W1": {"decision": "accept_adjacent", "paper_role": "generator_release"}, "W2": {"decision": "accept", "paper_role": "original_release"}}
        actual = {"W1": {"decision": "accept", "paper_role": "generator_release"}, "W2": {"decision": "exclude", "paper_role": "dataset_analysis"}}
        result = evaluate_openalex_regression(expected, actual)
        self.assertEqual(result.metrics["accuracy"], .5)
        self.assertEqual(result.metrics["detection_rate"], .5)
        self.assertEqual(result.metrics["classification_accuracy"], .5)

    def test_consecutive_vote_reduces_boundary_decisions(self) -> None:
        """Require threshold agreement and turn a tied boundary into uncertain."""
        majority = consensus_classification("github", {"run-1": {"decision": "accept"}, "run-2": {"decision": "accept"}, "run-3": {"decision": "exclude"}}, .66)
        tied = consensus_classification("openalex", {"run-1": {"decision": "accept"}, "run-2": {"decision": "exclude"}}, .5)
        self.assertEqual(majority["decision"], "accept")
        self.assertEqual(tied["decision"], "uncertain")
        self.assertEqual(tied["ensemble_vote_status"], "tie")

    def test_openalex_field_consensus_falls_back_to_a_coherent_run(self) -> None:
        """Reject a synthetic role/type conflict and expose the audited fallback run."""
        base = {"decision": "accept", "relevance_tier": "core", "direct_release_evidence": "true", "artifact_name": "Fixture", "evidence_quote_verified": "true", "domain": "iot_iiot", "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled"}
        votes = {
            "run-1": {**base, "paper_role": "generator_release", "artifact_type": "generator_framework"},
            "run-2": {**base, "paper_role": "generator_release", "artifact_type": "dataset_and_generator"},
            "run-3": {**base, "paper_role": "original_release", "artifact_type": "dataset"},
        }
        result = consensus_classification("openalex", votes, .66)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["paper_role"], "generator_release")
        self.assertEqual(result["artifact_type"], "generator_framework")
        self.assertEqual(result["consensus_fallback_run"], "run-1")
        self.assertIn("field_consensus_fallback_to_representative", result["consensus_validation_flags"])

    def test_github_multilabel_consensus_recomputes_system_types(self) -> None:
        """Vote GitHub labels independently and derive system types from the result."""
        base = {"decision": "accept", "role": "native_detector", "independent_fork": "false", "direct_self_description": "true", "runtime_observation": "true", "reproducible_execution": "true", "offline_dataset_only": "false", "implements_detection_engine": "true", "requires_external_detector": "false", "integrates_multiple_detectors": "false", "cohesive_framework_architecture": "false", "research_basis": "false", "evidence_quote_verified": "true", "fork": "false", "upstream_full_name": ""}
        votes = {
            "run-1": {**base, "observation_scopes": "network", "capabilities": "detection;response", "detection_methods": "packet_analysis"},
            "run-2": {**base, "observation_scopes": "network", "capabilities": "detection;response", "detection_methods": "packet_analysis"},
            "run-3": {**base, "role": "support_tooling", "observation_scopes": "host_endpoint", "capabilities": "detection", "detection_methods": "log_analysis"},
        }
        result = consensus_classification("github", votes, .66)
        self.assertEqual(result["role"], "native_detector")
        self.assertEqual(result["observation_scopes"], "network")
        self.assertEqual(result["capabilities"], "detection;response")
        self.assertEqual(result["primary_system_type"], "NIDS")
        self.assertIn("NDR", result["system_types"].split(";"))
        self.assertEqual(result["consensus_fallback_run"], "")

    def test_query_sampling_takes_x_from_each_query(self) -> None:
        """Sample exactly x native-ranked works per query without oversampling."""
        class Client:
            def search(self, query: str, _per_page: int, _pages: int):
                """Yield a deterministic five-result query fixture."""
                yield from ({"id": f"{query}-{index}"} for index in range(5))
        rows, scanned, _exhausted = sample_query_results(Client(), ["a", "b"], 2, 100, 10)  # type: ignore[arg-type]
        self.assertEqual(len(rows), 4)
        self.assertEqual(scanned, {"a": 2, "b": 2})

    def test_survey_identification_terms_are_configurable(self) -> None:
        """Use caller-defined survey and release vocabularies."""
        work = {"title": "A landscape paper", "abstract": "We catalogue prior work.", "work_type": "article"}
        self.assertTrue(is_survey_work(work, ["landscape"], ["new corpus"]))
        self.assertFalse(is_survey_work(work, ["landscape"], ["catalogue prior work"]))


if __name__ == "__main__":
    unittest.main()
