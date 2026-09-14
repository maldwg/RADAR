# Rated classification evaluation

The evaluation stage compares saved pipeline classifications with a small, versioned human-rated set. It is offline: it does not call GitHub, OpenAlex, or Ollama, and it does not change classification decisions.

## Central rating profiles

The default profiles are:

- `config/evaluation/github.toml`
- `config/evaluation/openalex.toml`

Each `[[entities]]` table uses exactly one matching selector:

- GitHub: `repository = "owner/name"`. Matching uses the `full_name` saved in classifications and candidates.
- OpenAlex: `openalex_url = "https://openalex.org/W..."` is recommended. `title = "Exact paper title"` is also supported when no OpenAlex link is available.
- Legacy `id` selectors remain supported for existing custom profiles, but the default profiles no longer use them.

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

[[entities]]
openalex_url = "https://openalex.org/W2296509296"
name = "UNSW-NB15"
decision = "accept"
paper_role = "original_release"
artifact_type = "dataset"
```

The supported decisions are `accept`, OpenAlex `accept_adjacent`, `exclude`, and `uncertain`. Binary metrics use only human-rated include and exclude entities; a human-rated `uncertain` entry remains in exact decision and field reports but is outside the binary confusion-matrix denominator.

Only field values present on a rated entity are compared. This permits decision-only ratings and incrementally richer taxonomy ratings in the same file. Scalar comparisons trim whitespace and ignore case. Array and semicolon-separated multi-label values are compared as unordered sets. Empty and `unknown` remain distinct.

Classification files can declare their output contract through `classifier_supported_fields`. `*` means the classifier emits the complete source taxonomy; a decision-only baseline uses `decision`. Fields outside that contract remain visible in `field_metrics.csv` with `supported=false`, zero comparisons, and N/A scores. They are not silently counted as incorrect taxonomy predictions.

Profiles are validated before evaluation. A missing profile, wrong source/schema, missing or multiple selectors, duplicate selectors, ambiguous result matches, malformed field list, or unsupported decision fails the command instead of emitting misleading scores.

### Balanced default benchmark

The checked-in profiles are deliberately **stratified test sets**, not random samples and not estimates of how often each class occurs in the full discovery population. Both contain 24 rated entities split equally between 12 expected inclusions and 12 expected exclusions. This prevents accuracy from being dominated by whichever decision is most common.

For GitHub, the positive half contains two repositories for each eligible role: standalone detector, platform/framework, classifier/model, rules/content, support tooling, and research demo. Together those positives cover every concrete observation scope and detection method in the configured taxonomy. The negative half contains three examples from each of four common boundary families: generic monitors, offensive/scanning tools, curated/tutorial content, and unrelated repositories whose names collide with IDS terminology.

For OpenAlex, the positive half contains four original releases, four versioned releases, and four generator releases. The negative half contains three surveys, three downstream applications, three dataset analyses, and three unrelated high-overlap papers. Papers are chosen to vary domains, data forms, capture environments, and labeling, but those secondary fields are not forced to identical counts: some combinations are intrinsically rare or invalid, and fabricating symmetry there would make the ground truth less credible.

`test_default_profiles_preserve_the_stratified_benchmark_design` locks the decision and role quotas and the intended taxonomy coverage. A profile edit that silently reintroduces class imbalance therefore fails the regression suite.

## Running evaluation

Evaluate existing default classification files:

```bash
ids-meta github evaluate
ids-meta openalex evaluate
```

The keyword baseline command creates its classification file and immediately evaluates it against the same profiles without calling an LLM:

```bash
ids-meta github baseline
ids-meta openalex baseline
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

`tests/test_evaluation.py` uses fixed GitHub and OpenAlex ratings and saved CSV rows. It checks exact confusion matrices, requested metric aliases, abstention and missing-result behavior, field comparisons, validation failures, artifact creation, both CLI stages, and manifest counts. These tests are deterministic and network-free.

The lower-level `ids_meta.common.regression.evaluate_regression(...)` dictionary helper remains useful for small in-code unit fixtures. The evaluation stage is the persisted, centrally configured run-level counterpart.
