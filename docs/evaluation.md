# Rated classification evaluation

The evaluation stage compares saved pipeline classifications with a small, versioned human-rated set. It is offline: it does not call GitHub, OpenAlex, or Ollama, and it does not change classification decisions.

## Central rating profiles

The only checked-in profiles are:

- `config/evaluation/github.toml`
- `config/evaluation/openalex_ashraf_2026.toml`

Source-record profiles use exactly one matching selector per `[[entities]]` table:

- GitHub: `repository = "owner/name"`. Matching uses the `full_name` saved in classifications and candidates.
- OpenAlex: `openalex_url = "https://openalex.org/W..."` is recommended. `title = "Exact paper title"` is also supported when no OpenAlex link is available.
- Legacy `id` selectors remain supported for existing custom profiles, but the default profiles no longer use them.

The checked-in OpenAlex profile instead uses `evaluation_unit = "artifact"`: every entity names a dataset with `artifact`, optional `aliases`, and survey-reference provenance. Evaluation searches the saved OpenAlex results for an accepted release representing that dataset.

Repository names and titles are matched exactly after trimming repeated whitespace and ignoring letter case. A trailing slash on an OpenAlex URL is ignored. If a selector matches multiple rows--most plausibly a duplicated paper title--evaluation stops with an ambiguity error instead of guessing. The resolved stable source ID remains visible in `instances.csv`.

```toml
schema_version = "1.0"
source = "github"
classification_fields = ["role", "observation_scopes", "capabilities", "detection_methods"]

[[entities]]
repository = "OISF/suricata"
decision = "accept"
role = "native_detector"
observation_scopes = ["network"]
capabilities = ["detection", "prevention", "monitoring"]
detection_methods = ["signature_based", "packet_analysis"]

[[entities]]
repository = "sbilly/awesome-security"
decision = "exclude"
```

```toml
schema_version = "1.0"
source = "openalex"
evaluation_unit = "artifact"
classification_fields = ["artifact_type", "domain", "data_form", "capture_environment", "labeling"]

[[entities]]
artifact = "UNSW-NB15"
aliases = ["UNSW NB15"]
name = "UNSW-NB15"
decision = "accept"
artifact_type = "dataset"
domain = "general_network"
survey_reference_openalex_url = "https://openalex.org/W2296509296"
```

The supported decisions are `accept`, OpenAlex `accept_adjacent`, `exclude`, and `uncertain`. Binary metrics use only human-rated include and exclude entities; a human-rated `uncertain` entry remains in exact decision and field reports but is outside the binary confusion-matrix denominator.

Only field values present on a rated entity are compared. This permits decision-only ratings and incrementally richer taxonomy ratings in the same file. Scalar comparisons trim whitespace and ignore case. Array and semicolon-separated multi-label values are compared as unordered sets. Empty and `unknown` remain distinct.

Classification files declare their output contract through `classifier_supported_fields`. `*` means the classifier emits the complete source taxonomy. Fields outside a custom classifier's contract remain visible in `field_metrics.csv` with `supported=false`, zero comparisons, and N/A scores; they are not silently counted as incorrect taxonomy predictions. The checked-in Keyword, NLI, LLM, and Ensemble approaches all emit the rated taxonomy fields.

Profiles are validated before evaluation. A missing profile, wrong source/schema, missing or multiple selectors, duplicate selectors, ambiguous result matches, malformed field list, or unsupported decision fails the command instead of emitting misleading scores.

### Benchmark composition

The GitHub profile is a deliberately **stratified test set**, not a random sample and not an estimate of how often each class occurs in the full discovery population.

For GitHub, the positive half contains two repositories for each eligible role: standalone detector, platform/framework, classifier/model, rules/content, support tooling, and research demo. Together those positives cover every concrete observation scope and detection method in the configured taxonomy. The negative half contains three examples from each of four common boundary families: generic monitors, offensive/scanning tools, curated/tutorial content, and unrelated repositories whose names collide with IDS terminology.

OpenAlex uses only the 105-dataset positive-reference fixture extracted from Table 2 of Ashraf and Masoodi (2026). It measures whether each named survey dataset is represented by an accepted release result. Because this retained benchmark has no negative examples, false-positive rate, specificity, accuracy, precision, and F1 are not estimable and are reported as N/A.

Regression tests lock the GitHub strata and the source counts, exclusions, aliases, requested domain taxonomy, and provenance fields of the OpenAlex dataset fixture.

## Running evaluation

Evaluate existing default classification files:

```bash
ids-meta github evaluate
ids-meta openalex evaluate
```

The baseline command creates keyword and/or local zero-shot NLI classification files and immediately evaluates them against the same profiles without calling an LLM:

```bash
ids-meta github baseline
ids-meta openalex baseline
ids-meta github baseline --methods nli
ids-meta openalex baseline --methods keyword,nli
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

Automatic LLM evaluation runs immediately after classification and consensus, before reconciliation. When `[baselines].enabled = true`, a complete run subsequently executes the configured baseline methods, evaluates them, and rebuilds the all-classifier plots. Explicit `evaluate` and `baseline` commands run regardless of their automatic-stage switches.

For one model, artifacts use `artifacts/evaluation/<source>/`. In multi-model mode, every result is evaluated independently under `artifacts/evaluation/<source>/models/<model-slug>/`; the ensemble uses `artifacts/evaluation/<source>/ensemble/`. Ratings and candidates are shared, which makes comparisons use the same reference set.

Cross-approach plots are written to `artifacts/plots/comparisons/<source>/classifiers/`. The OpenAlex comparison uses the 105-dataset artifact benchmark. Detection rate, false-positive rate, accuracy, precision, decision-F1, definitive coverage, abstention, aggregate taxonomy exact accuracy, and mean taxonomy-field macro-F1 each have a dedicated figure with every available Keyword, NLI, LLM, and Ensemble result on the same axis. Further heatmaps compare every approach by rated field using exact accuracy and macro-F1, then by every individual taxonomy category using one-vs-rest F1. Bar and cell values are deliberately not printed on the figures; companion CSVs and each evaluation's `field_label_metrics.csv` retain exact values, per-label TP/FP/FN, classifier IDs, rating paths, and Wilson intervals. The interval level is configured as `evaluation.confidence_level`; the evaluator derives its normal quantile from that value. Decision-count, pairwise-agreement, and disagreement diagnostics sit below `decision_agreement/`. Metrics that a positive-only artifact fixture cannot estimate are shown as N/A, not as zero.

NLI evaluation also writes `threshold_diagnostics/`. It replays the aggregate probabilities already stored in `nli_scores_json` over the finite grid configured in `[baselines.nli.sensitivity]`, so it performs no inference and needs no network access. Every metric has a separate plot, the configured threshold/margin pair is marked, and the complete grid is retained as CSV. The sweep is descriptive; it must not silently tune the operating point on the same fixture used for final reporting.

Use `ids-meta <source> baseline --evaluate-only` to rebuild baseline evaluations, threshold diagnostics, and cross-approach plots from existing baseline classification CSVs. This mode does not open the source cache or initialize the keyword or NLI classifier.

## Metric views

A human `accept` or `accept_adjacent` is a positive; `exclude` is a negative. An actual `accept` or `accept_adjacent` predicts positive.

The report provides three views so uncertainty and missing results cannot disappear behind one denominator:

- **Operational:** every actual result other than inclusion predicts negative. This is the end-to-end screening view: an uncertain, failed, missing, or unretrieved expected inclusion is an FN; those results on an expected exclusion are TN.
- **Strict:** every non-definitive result is an error. On expected inclusions it is FN; on expected exclusions it is FP. Only explicit correct inclusion/exclusion decisions count as correct.
- **Selective:** uses only actual definitive include/exclude decisions. It measures correctness where the system committed to a decision; coverage and abstention must be reported beside it.

In plain language, the views answer different questions:

| View | Question answered | Uncertain/missing when inclusion was expected | Uncertain/missing when exclusion was expected |
|---|---|---:|---:|
| Operational | What does the complete pipeline actually deliver in the final included set? | False negative | True negative |
| Strict | What happens if every unresolved boundary case is considered wrong? | False negative | False positive |
| Selective | How accurate is the system only where it made a definite decision? | Omitted | Omitted |

For example, assume a repository should be excluded but the pipeline returns `uncertain`. Operational evaluation counts it as a true negative because it was not included. Strict evaluation counts it as a false positive because the required explicit exclusion was not produced. Selective evaluation removes it from its denominator. If an entity should instead be included, an uncertain or missing result is a false negative in both operational and strict evaluation and is omitted selectively.

Selective results must always be read together with `definitive_coverage` and `abstention_rate`. Otherwise, a system could appear highly accurate simply by leaving difficult cases unresolved. Operational is the primary end-to-end screening view; strict is the conservative boundary-quality view; selective describes committed decisions only.

Each view reports TP, FP, TN, FN and:

- DR / recall / sensitivity = TP / (TP + FN)
- FPR = FP / (FP + TN)
- FNR = FN / (TP + FN)
- PREC / precision = TP / (TP + FP)
- ACC / accuracy = (TP + TN) / the view's denominator
- specificity, negative predictive value, F1, and balanced accuracy

The report also records exact decision accuracy, classification-field exact accuracy, per-field multi-label F1, retrieval coverage, classification coverage, definitive coverage, abstention rate, status counts, and the number of actual entities that were not in the rated set. Unrated actual rows are reported but do not enter rated-set metrics.

## Artifacts

Each evaluation directory contains:

- `metrics.json`: machine-readable summary and all three metric views.
- `instances.csv`: one rated entity per row, configured selector and resolved stable ID, progression status, expected/actual decision, exact field results, and full JSON snapshots.
- `decision_confusion.csv`: exact expected-to-actual disposition counts, including uncertain and unavailable.
- `field_metrics.csv`: support status plus exact and multi-label metrics for every configured field; unsupported baseline fields are N/A.
- `README.md`: compact human-readable headline values, metadata accuracy, metric semantics, and figure links.
- `01_decision_performance.svg`/`.pdf`: operational, strict, and selective DR, precision, accuracy, F1, FPR, and FNR.
- `02_decision_confusion.svg`/`.pdf`: exact expected-to-actual dispositions, retaining uncertain and unavailable outcomes.
- `03_field_accuracy.svg`/`.pdf`: exact metadata accuracy per configured field plus directly labeled multi-label F1 bars where applicable.
- `04_coverage_and_status.svg`/`.pdf`: retrieval, classification, definitive-decision, abstention, and final-status summaries.
- `plot_data/`: the exact aggregate CSVs behind the four figures.

Figure formats and resolution use the shared `[plots]` `formats` and `dpi` settings. The run manifest additionally records the operational TP, FP, TN, FN, rated entity count, and artifact count. This makes recurring runs comparable without treating the rating profiles as inputs to the classifier itself.

## Regression tests

`tests/test_evaluation.py` uses the fixed GitHub repository ratings, the survey-derived OpenAlex dataset fixture, and temporary saved CSV rows. It checks exact confusion matrices, requested metric aliases, abstention and missing-result behavior, field comparisons, validation failures, artifact creation, both CLI stages, and manifest counts. These tests are deterministic and network-free.

The lower-level `ids_meta.common.regression.evaluate_regression(...)` dictionary helper remains useful for small in-code unit fixtures. The evaluation stage is the persisted, centrally configured run-level counterpart.
