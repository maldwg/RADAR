from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from tests.fixtures import SYSTEM_TYPE_PROFILE
from unittest.mock import patch

from radar.common.cache import PipelineCache
from radar.common.csvio import read_csv
from tests.fixtures import write_csv as atomic_write_csv
from radar.common.nli import (
    EvidenceChunk, NLIField, NLIProfile, NLIRuntimeConfig, NLIScore,
    classify_nli_baseline, infer_nli_fields, nli_profile_hash, rescore_nli_rows,
)
from radar.common.nli_diagnostics import NLIThresholdGrid, build_nli_threshold_diagnostics
from radar.common.ollama import EvidenceSection
from radar.common.evaluation import load_rating_profile
from radar.github.schema import CANDIDATE_COLUMNS


class FakeNLIBackend:
    """Deterministic test backend that avoids model downloads and GPU dependencies."""

    model_id = "fixture/nli"
    revision = "revision-1"

    def chunk_sections(
        self, sections: list[EvidenceSection], _premise_tokens: int, _overlap: int,
    ) -> list[EvidenceChunk]:
        """Expose each fixture evidence section as one chunk."""
        return [EvidenceChunk(index, section.source, section.text) for index, section in enumerate(sections)]

    def score(self, pairs: list[tuple[str, str]], _batch_size: int) -> list[NLIScore]:
        """Entail hypotheses marked MATCH and contradict hypotheses marked FALSE."""
        scores = []
        for _premise, hypothesis in pairs:
            if "MATCH" in hypothesis:
                scores.append(NLIScore(.88, .08, .04))
            elif "FALSE" in hypothesis:
                scores.append(NLIScore(.04, .08, .88))
            else:
                scores.append(NLIScore(.43, .52, .05))
        return scores


def _config() -> NLIRuntimeConfig:
    """Return a compact deterministic NLI runtime for unit tests."""
    return NLIRuntimeConfig(
        model="fixture/nli", revision="revision-1", device="cpu", batch_size=4,
        max_length=128, premise_tokens=64, chunk_overlap=8, max_chunks=2,
        top_k=1, entailment_threshold=.5, label_margin=.1,
    )


class NLITests(unittest.TestCase):
    """Exercise abstention-aware NLI inference, normalization, and resume caching."""

    def test_inference_supports_exclusive_multilabel_boolean_and_abstention(self) -> None:
        """Convert independent three-way scores into the three configured field modes."""
        profile = NLIProfile("github", (
            NLIField("role", "exclusive", "uncertain", {
                "winner": "MATCH role", "other": "other role",
            }),
            NLIField("scopes", "multilabel", "unknown", {
                "network": "MATCH scope", "host": "neutral scope",
            }),
            NLIField("supported", "boolean", "", {"true": "FALSE proposition"}),
            NLIField("ambiguous", "exclusive", "uncertain", {
                "one": "neutral one", "two": "neutral two",
            }),
        ), {}, Path("fixture.toml"))
        predictions, audit, support = infer_nli_fields(
            [EvidenceSection("README", "fixture evidence")], profile, FakeNLIBackend(), _config(),
        )
        self.assertEqual(predictions["role"], "winner")
        self.assertEqual(predictions["scopes"], ["network"])
        self.assertEqual(predictions["supported"], "false")
        self.assertEqual(predictions["ambiguous"], "uncertain")
        self.assertEqual(audit["_evidence"]["used_chunks"], 1)
        self.assertEqual(support["role"]["winner"][0]["source"], "README")

    def test_runner_writes_auditable_output_and_reuses_compatible_cache(self) -> None:
        """Run a source adapter and preserve the timestamp on a compatible resume."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.csv"
            output = root / "nli.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update({
                "github_repo_id": "g1", "full_name": "fixture/repo",
                "evidence_hash": "candidate-hash", "fork": "false",
            })
            atomic_write_csv(candidates, [row], CANDIDATE_COLUMNS)
            profile = NLIProfile("github", (
                NLIField("role", "exclusive", "uncertain", {
                    "out_of_scope": "MATCH role", "native_detector": "other role",
                }),
            ), {"fixture": True}, root / "profile.toml")
            original = {"repository": {"full_name": "fixture/repo"}, "readme_text": "Original fixture evidence"}
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_collected_record("github.repository", "g1", original, "now")
                first = classify_nli_baseline(
                    "github", candidates, output, cache, profile, _config(),
                    backend=FakeNLIBackend(), progress_every=1,
                 adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
                resumed = classify_nli_baseline(
                    "github", candidates, output, cache, profile, _config(),
                    backend=FakeNLIBackend(), progress_every=1,
                 adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
            self.assertEqual(first[0]["decision"], "exclude")
            self.assertEqual(first[0]["classifier_method"], "nli")
            self.assertEqual(first[0]["classified_at"], resumed[0]["classified_at"])
            self.assertEqual(read_csv(output)[0]["nli_model_revision"], "revision-1")

            with PipelineCache(root / "empty-cache.sqlite3") as empty_cache, patch(
                "radar.common.nli.TransformersNLIBackend",
                side_effect=AssertionError("a compatible output CSV must not initialize the model"),
            ):
                empty_cache.put_collected_record("github.repository", "g1", original, "now")
                reused_from_csv = classify_nli_baseline(
                    "github", candidates, output, empty_cache, profile, _config(),
                    progress_every=1,
                 adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
            self.assertEqual(reused_from_csv[0]["classified_at"], first[0]["classified_at"])

    def test_saved_scores_can_be_rescored_without_a_backend(self) -> None:
        """Change the abstention threshold by replaying aggregates rather than inference."""
        profile = NLIProfile("github", (
            NLIField("role", "exclusive", "uncertain", {
                "out_of_scope": "MATCH role", "native_detector": "other role",
            }),
        ), {}, Path("fixture.toml"))
        _predictions, audit, _support = infer_nli_fields(
            [EvidenceSection("README", "fixture evidence")], profile, FakeNLIBackend(), _config(),
        )
        row = {
            "github_repo_id": "g1", "fork": "false", "nli_scores_json": json.dumps(audit),
        }
        relaxed = rescore_nli_rows("github", [row], profile, .5, .1, adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})[0]
        strict = rescore_nli_rows("github", [row], profile, .9, .1, adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})[0]
        self.assertEqual(relaxed["decision"], "exclude")
        self.assertEqual(strict["decision"], "uncertain")

    def test_threshold_diagnostic_replays_saved_scores_only(self) -> None:
        """Render a finite threshold sweep from a classification audit without a model."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = NLIProfile("github", (
                NLIField("role", "exclusive", "uncertain", {
                    "out_of_scope": "MATCH role", "native_detector": "other role",
                }),
            ), {}, root / "nli.toml")
            _predictions, audit, _support = infer_nli_fields(
                [EvidenceSection("README", "fixture")], profile, FakeNLIBackend(), _config(),
            )
            classifications = root / "classifications.csv"
            atomic_write_csv(classifications, [{
                "github_repo_id": "g1", "full_name": "fixture/repo", "fork": "false",
                "decision": "exclude", "classifier_method": "nli",
                "classifier_profile_hash": nli_profile_hash(profile, _config()),
                "nli_entailment_threshold": ".5", "nli_label_margin": ".1",
                "nli_scores_json": json.dumps(audit),
             'classifier_supported_fields': '*'}], [
                "github_repo_id", "full_name", "fork", "decision", "classifier_method",
                "classifier_profile_hash", "nli_entailment_threshold", "nli_label_margin",
                "nli_scores_json",
            ])
            candidates = root / "candidates.csv"
            atomic_write_csv(candidates, [{"github_repo_id": "g1"}], ["github_repo_id"])
            ratings = root / "ratings.toml"
            ratings.write_text(
                'schema_version = "1.0"\nsource = "github"\nclassification_fields = ["role"]\n\n'
                '[[entities]]\nid = "g1"\ndecision = "exclude"\nrole = "out_of_scope"\n',
                encoding="utf-8",
            )
            outputs = build_nli_threshold_diagnostics(
                "github", classifications, candidates, load_rating_profile(ratings, "github"),
                profile, _config(), NLIThresholdGrid(True, (.5, .9), (.1,)),
                root / "diagnostics",
                {"formats": ["svg"], "dpi": 72}, confidence_level=.9,
            )
            self.assertEqual(len(read_csv(
                root / "diagnostics/data/nli_threshold_sensitivity.csv"
            )), 2)
            self.assertTrue(any(path.name == "00_threshold_sensitivity.svg" for path in outputs))
            self.assertTrue((root / "diagnostics/README.md").is_file())


if __name__ == "__main__":
    unittest.main()
