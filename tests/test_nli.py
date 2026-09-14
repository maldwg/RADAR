from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ids_meta.common.cache import PipelineCache
from ids_meta.common.csvio import atomic_write_csv, read_csv
from ids_meta.common.nli import (
    EvidenceChunk, NLIField, NLIProfile, NLIRuntimeConfig, NLIScore,
    classify_nli_baseline, infer_nli_fields,
)
from ids_meta.common.ollama import EvidenceSection
from ids_meta.github.schema import CANDIDATE_COLUMNS


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
            with PipelineCache(root / "cache.sqlite3") as cache:
                first = classify_nli_baseline(
                    "github", candidates, output, cache, profile, _config(),
                    backend=FakeNLIBackend(), progress_every=1,
                )
                resumed = classify_nli_baseline(
                    "github", candidates, output, cache, profile, _config(),
                    backend=FakeNLIBackend(), progress_every=1,
                )
            self.assertEqual(first[0]["decision"], "exclude")
            self.assertEqual(first[0]["classifier_method"], "nli")
            self.assertEqual(first[0]["classified_at"], resumed[0]["classified_at"])
            self.assertEqual(read_csv(output)[0]["nli_model_revision"], "revision-1")


if __name__ == "__main__":
    unittest.main()
