# Rated classification evaluation

The evaluation stage compares saved pipeline classifications with a small, versioned human-rated set. It is offline: it does not call GitHub, OpenAlex, or Ollama, and it does not change classification decisions.

## Central rating profiles

The default profiles are:

- `config/evaluation/github.toml`
- `config/evaluation/openalex.toml`

Each `[[entities]]` table identifies one rated entity with the exact stable ID stored in the pipeline CSV. `id` maps to `github_repo_id` for GitHub and `openalex_id` for OpenAlex. IDs must be unique.

```toml
schema_version = "1.0"
source = "github"
classification_fields = ["role", "observation_scopes", "capabilities", "detection_methods"]

[[entities]]
id = "123456789"
name = "Optional readable label"
decision = "accept"
role = "standalone_detector"
observation_scopes = ["network"]
capabilities = ["detection", "response"]
detection_methods = ["signature_based"]

[[entities]]
id = "987654321"
decision = "exclude"
```

The supported decisions are `accept`, OpenAlex `accept_adjacent`, `exclude`, and `uncertain`. Binary metrics use only human-rated include and exclude entities; a human-rated `uncertain` entry remains in exact decision and field reports but is outside the binary confusion-matrix denominator.

Only field values present on a rated entity are compared. This permits decision-only ratings and incrementally richer taxonomy ratings in the same file. Scalar comparisons trim whitespace and ignore case. Array and semicolon-separated multi-label values are compared as unordered sets. Empty and `unknown` remain distinct.

Profiles are validated before evaluation. A missing profile, wrong source/schema, duplicate/blank ID, malformed field list, or unsupported decision fails the command instead of emitting misleading scores.

## Running evaluation

Evaluate existing default classification files:

```bash
ids-meta github evaluate
ids-meta openalex evaluate
```

Use other classification and candidate files or another artifact directory:

```bash
ids-meta github evaluate \
  --input path/to/classifications.csv \
  --candidates path/to/candidates.csv \
  --output artifacts/evaluation/experiment-a
```

The candidate CSV is optional. When supplied, it lets the evaluator distinguish a rated entity that was retrieved but not classified (`not_classified`) from one that was never retrieved (`not_retrieved`). Without it, missing classifications are `unavailable` and retrieval coverage is `null`, because retrieval cannot honestly be inferred.

To make evaluation an automatic stage of each complete source run, set:

```toml
[evaluation]
enabled = true
```

Automatic evaluation runs immediately after classification and consensus, before reconciliation. The explicit `evaluate` commands run regardless of this switch.

For one model, artifacts use `artifacts/evaluation/<source>/`. In multi-model mode, every result is evaluated independently under `artifacts/evaluation/<source>/models/<model-slug>/`; the ensemble uses `artifacts/evaluation/<source>/ensemble/`. Ratings and candidates are shared, which makes comparisons use the same reference set.

## Metric views

A human `accept` or `accept_adjacent` is a positive; `exclude` is a negative. An actual `accept` or `accept_adjacent` predicts positive.

The report provides three views so uncertainty and missing results cannot disappear behind one denominator:

- **Operational:** every actual result other than inclusion predicts negative. This is the end-to-end screening view: an uncertain, failed, missing, or unretrieved expected inclusion is an FN; those results on an expected exclusion are TN.
- **Strict:** every non-definitive result is an error. On expected inclusions it is FN; on expected exclusions it is FP. Only explicit correct inclusion/exclusion decisions count as correct.
- **Selective:** uses only actual definitive include/exclude decisions. It measures correctness where the system committed to a decision; coverage and abstention must be reported beside it.

Each view reports TP, FP, TN, FN and:

- DR / recall / sensitivity = TP / (TP + FN)
- FPR = FP / (FP + TN)
- FNR = FN / (TP + FN)
- PREC / precision = TP / (TP + FP)
- ACC / accuracy = (TP + TN) / the view's denominator
- specificity, negative predictive value, F1, and balanced accuracy

The report also records exact decision accuracy, classification-field exact accuracy, per-field multi-label precision/recall/F1/Jaccard, retrieval coverage, classification coverage, definitive coverage, abstention rate, status counts, and the number of actual entities that were not in the rated set. Unrated actual rows are reported but do not enter rated-set metrics.

## Artifacts

Each evaluation directory contains:

- `metrics.json`: machine-readable summary and all three metric views.
- `instances.csv`: one rated entity per row, progression status, expected/actual decision, exact field results, and full JSON snapshots.
- `decision_confusion.csv`: exact expected-to-actual disposition counts, including uncertain and unavailable.
- `field_metrics.csv`: exact and multi-label metrics for every configured field.
- `README.md`: compact human-readable headline values and metric semantics.

The run manifest additionally records the operational TP, FP, TN, FN, rated entity count, and artifact count. This makes recurring runs comparable without treating the rating profiles as inputs to the classifier itself.

## Regression tests

`tests/test_evaluation.py` uses fixed GitHub and OpenAlex ratings and saved CSV rows. It checks exact confusion matrices, requested metric aliases, abstention and missing-result behavior, field comparisons, validation failures, artifact creation, both CLI stages, and manifest counts. These tests are deterministic and network-free.

The lower-level `ids_meta.common.regression.evaluate_regression(...)` dictionary helper remains useful for small in-code unit fixtures. The evaluation stage is the persisted, centrally configured run-level counterpart.
