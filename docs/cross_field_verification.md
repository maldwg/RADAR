# Cross-field verification with the Memmesheimer review

The implemented adapter uses Pascal Memmesheimer, Vincent Heuveline and Jürgen
Hesser's *A Systematic Review of Conformal Inference Procedures for Treatment
Effect Estimation: Methods and Challenges*. The accessible
[arXiv v1](https://arxiv.org/html/2509.21660v1) supplies the reference inventory
and assumption labels. The requested published edition is
[IEEE Access, DOI 10.1109/ACCESS.2026.3683260](https://doi.org/10.1109/ACCESS.2026.3683260).
Its screening rules were also checked against the
[author's public full text](https://www.researchgate.net/publication/403783179_A_Systematic_Review_of_Conformal_Inference_Procedures_for_Treatment_Effect_Estimation_Methods_and_Challenges).
The benchmark is explicitly versioned as arXiv v1; it is not a reconstruction
of the published edition's entire retrieval population.

## Verification approach

Treat this as a replication experiment in a second field. Discover and screen
papers about conformal treatment-effect methods, then evaluate saved predictions
against eleven reviewed inclusions. This tests portability of the review
workflow; it does not establish accuracy for IDS research or statistical
coverage guarantees for the classifier.

The paper's screening methodology transfers directly: all five eligibility
criteria must pass, and at least five of nine quality criteria must pass
(Section 3.3 of the preprint; Section III.C of the publication). The new
`StudyScreeningAdapter` replays those rules on supplied criterion judgments,
keeps supporting quotes, and exports the same portable classification format
used by `ids-meta report`.

| File | Purpose |
| --- | --- |
| `config/studies/conformal_treatment_effects/taxonomy.toml` | Paper identity, title selector, and assumption taxonomy |
| `config/studies/conformal_treatment_effects/study.toml` | Study entry point, models, discovery source and isolated outputs |
| `config/studies/conformal_treatment_effects/prompts/classification.toml` | Blind assessment prompt with explicit scientific inputs |
| `config/studies/conformal_treatment_effects/screening.toml` | Five eligibility rules, nine quality rules, and threshold |
| `config/studies/conformal_treatment_effects/pipeline.toml` | Collection limits, full-text policy and classification budget |
| `config/studies/conformal_treatment_effects/search/openalex.toml` | OpenAlex query adaptations and publication-date window |
| `config/studies/conformal_treatment_effects/evaluation/references.toml` | Eleven positive papers and three rated assumption fields |
| `src/ids_meta/common/study_screening.py` | Evidence validation, rule replay, and CSV export |
| `src/ids_meta/common/study_pipeline.py` | Independent discovery, PDF extraction, and blind model assessments |
| `examples/verification/memmesheimer/assessments.json` | Synthetic input illustrating accept, exclude, and uncertain |

The configured workflow can be selected directly:

```bash
ids-meta --study conformal_treatment_effects validate
ids-meta --study conformal_treatment_effects collect --collect-limit 200
ids-meta --study conformal_treatment_effects classify --classify-limit 5
ids-meta --study conformal_treatment_effects evaluate
```

These commands use study-local queries, criteria, extraction definitions and
prompts. The standalone commands below remain useful for explicit input/output
files and supplied assessment replay.

The reference crosswalk converts Table 3 checkmarks to `required` and crosses
to `not_required` for SUTVA, strong ignorability, and overlap. A cross does not
mean an assumption is false in a dataset. Table 2 and the bibliography provide
paper identities and provenance. The Jonkers selector uses the full bibliography
title because Table 2 abbreviates its ending as ITEs. Source years are preserved,
including differences from reference dates; they are not used as matching keys.

## Input and screening

Each JSON record needs a unique `paper_id`, a `title`, supplied `full_text`, and
criterion assessments. IDs can be reviewed DOIs, arXiv identifiers, or stable
local keys. Use the same identity convention in candidates and classifications.
Resolve duplicate manifestations before import; title collisions fail evaluation
instead of being merged automatically.

An assessment has this form:

```json
"F1": {
  "value": "yes",
  "quote": "An exact supporting passage from the supplied paper text",
  "source": "full_text"
}
```

All values are `yes`, `no`, or `unknown`. `yes` means the criterion is satisfied:
for example, F3=yes means that the paper is not a review. Missing criteria default
to `unknown`. Every definite assessment requires a non-empty quote occurring in
the supplied full text after whitespace and Latin-ligature normalization
(for example, `ﬀ` and `ff`). Words, punctuation and mathematical characters are
otherwise compared literally, and raw quotes/text are retained. Do not quote the review
as evidence about a candidate paper. Provide taxonomy predictions separately as
`sutva`, `strong_ignorability`, and `overlap`, using `required`, `not_required`,
or `unknown`; absent fields are not filled from the reference labels.

The decision is `exclude` if any eligibility rule fails, or if the number of
quality passes plus unknown quality judgments cannot reach five. It is `accept`
when all eligibility rules pass and at least five quality rules pass. Otherwise
it is `uncertain`. Unassessed quality rules may remain unknown once five passes
are established. Failed and unknown filter IDs, quality counts, original
assessments, and input/protocol/profile hashes remain in the classification CSV.

`study-screen` validates quote occurrence and rule arithmetic. Human review is
still needed to decide whether a quote supports its criterion. It does not call
a model, retrieve papers, infer assumptions, or automatically decide that two
analogous methods represent the same contribution. Record that final
reconciliation judgment separately.

Try the synthetic demonstration:

```bash
ids-meta study-screen \
  --protocol config/studies/conformal_treatment_effects/screening.toml \
  --input examples/verification/memmesheimer/assessments.json \
  --output /tmp/memmesheimer-demo/classifications.csv

ids-meta report \
  --profile config/studies/conformal_treatment_effects/taxonomy.toml \
  --input /tmp/memmesheimer-demo/classifications.csv \
  --output /tmp/memmesheimer-demo/report

ids-meta verify --input /tmp/memmesheimer-demo/report
ids-meta verify --database /tmp/memmesheimer-demo/report/reporting.sqlite3
```

These examples are synthetic software checks, not measured research results.

## Evaluate a real run

Collect a bounded cohort and screen its available PDF text with the configured
Ollama model. Collection never reads the reference profile. Model prompts contain
the paper title, abstract, supplied text and protocol rules; they omit expected
labels and separate author, citation-count, or discovery-query metadata.
Author names and references can still occur within the paper text itself.

```bash
ids-meta study-collect \
  --search-profile config/studies/conformal_treatment_effects/search/openalex.toml \
  --collect-limit 100 --classify-all --query-limit-factor 3 \
  --output data/verification/memmesheimer/candidates.json

ids-meta study-classify \
  --protocol config/studies/conformal_treatment_effects/screening.toml \
  --input data/verification/memmesheimer/candidates.json \
  --output data/verification/memmesheimer/classifications.csv \
  --fetch-full-text --resume --consecutive-runs 5 --consensus-threshold 0.8
```

`study-classify --limit 2` limits inference to a pilot; `--dry-run` performs no
PDF downloads or model calls. Missing text produces an uncertain record. Invalid
assessments get one correction request listing all schema/quotation problems,
without reference labels. Both responses and their validation errors are retained.
If the correction still fails, or the backend fails, the record is an error.

The configured study commands use the same limits, five repeated assessments and
ensemble policy as the IDS workflows:

```bash
ids-meta --study conformal_treatment_effects run
ids-meta --study conformal_treatment_effects collect --collect-all --classify-all
```

The default retains 100 candidates after retrieving up to 300 hits per query.
The full run classifies each retained candidate five times per selected model,
votes with threshold 0.8, then combines the selected models at threshold 1.0.
Winning judgments retain source quotes, and disposition conflicts abstain.
With unlimited collection and classification, searches paginate to exhaustion
within the configured publication-date window. The standalone commands above
write explicit files; the study runner additionally writes the ensemble and
model-specific reports under `artifacts/studies/conformal_treatment_effects/`.

Individual invalid assessments remain recorded as errors. Failed assessments
count toward the voting denominator; enough valid, agreeing assessments can
still produce a usable consensus, while all failed assessments produce an error.
A PDF download is
capped at 30 MiB and supplied text at 120,000 characters by default; truncation is
recorded. The model gets at least 4,000 output tokens for fourteen criteria and
three assumption fields. Quote checking establishes textual presence, while the
meaning and sufficiency of each quotation still require review.

Each classification is checkpointed in a neighboring `.assessments.json` file.
Successful assessments can be resumed when their input content, protocol,
source profile, prompt version, model settings and CSV checkpoint hashes match.
Missing-text/error records are retried when resumed. Candidate text and
assessments remain in the run outputs for inspection; model identifiers are
recorded, but model weights are not independently pinned. Reference labels are
joined only when reporting.

Alternatively, replay independently reviewed assessments with `study-screen`
or import an existing classifier's decisions directly. Supply the entire
candidate export to distinguish known papers that were not retrieved from
retrieved papers that were not classified:

```bash
ids-meta report \
  --profile config/studies/conformal_treatment_effects/taxonomy.toml \
  --input data/verification/memmesheimer/classifications.csv \
  --candidates data/verification/memmesheimer/candidates.json \
  --ratings config/studies/conformal_treatment_effects/evaluation/references.toml \
  --output artifacts/verification/memmesheimer

ids-meta verify --input artifacts/verification/memmesheimer
ids-meta verify --database artifacts/verification/memmesheimer/reporting.sqlite3
```

Read retrieval coverage, end-to-end recovery, missing/uncertain decisions,
assumption accuracy, and accuracy among recovered positives separately. The
reference list contains only inclusions: precision, false-positive rate,
false-discovery rate, specificity, binary accuracy and decision-F1 are N/A.
The evaluator now applies this policy to positive-only paper benchmarks as well
as artifact inventories. The verifier rejects finite scores for those metrics
when the bundle declares a positive-only benchmark. Decision accuracy remains a
descriptive match fraction on the rated list, not population binary accuracy.

Reference matches, frozen reporting inputs, metric audits and plot tables can be
checked independently with `verify`. Keep assessment JSON and the protocol with
the run as well: the screening CSV retains their hashes, while the reporting
bundle freezes its normalized CSVs, source profile and rating inputs.

## Completed development pilot

The saved run in `data/verification/memmesheimer/pilot_scoped/` collected at most
25 results per query and retained five of 49 distinct OpenAlex records. The
five retained candidates contain two of the eleven reviewed references. Only
those first two candidates were submitted to `gemma4:31b`, with a 60,000-character
text cap, a 90-second request timeout, no network retries, and one permitted
evidence correction.

| Paper | Observed outcome |
| --- | --- |
| Conformal Inference of Counterfactuals and Individual Treatment Effects | Accepted: all five eligibility rules and six quality rules pass; three assumption labels match the reference |
| Conformal Off-policy Prediction | Evidence-validation error: quotations for Q9, strong ignorability and overlap remain absent after correction |

The adapter fixed a real PDF typography issue by normalizing Latin ligatures,
while retaining the original text and quotes. It still rejects reworded passages
and changed mathematical notation. The earlier runs and responses are preserved,
so the development history is inspectable rather than replaced by hand labels.

The report is in `artifacts/verification/memmesheimer/pilot_scoped/`. Its observed
reference coverage is 2/11 (18.2%) and end-to-end inclusion recovery is 1/11
(9.1%). Nine references are absent from the retained cohort; one retrieved
reference has an assessment error. Both figure/table verification and reporting
database verification pass. The reporting database additionally freezes the
collection audit, assessment checkpoint, inference settings, protocol and
adapter implementations.

This small, capped run was used to develop the adapter. Its results describe
that run; they do not estimate general classifier accuracy, final search recall,
or performance on IDS research. A larger frozen collection and an independently
reviewed sample of negatives are needed for a mixed screening benchmark.

## Search replication limits

The preprint reports 1,250 initial hits; the published text reports 2,255.
The preprint also inconsistently names 2005 and 2015 as the start year, and its
Google Scholar cap is 199 results, whereas the published limitations describe
1,000. These are version/protocol discrepancies to audit, not interchangeable
observations. This adapter does not manufacture intermediate stage memberships
from those aggregate numbers.

For a discovery experiment, freeze the chosen edition's search vocabulary,
source-specific query syntax, dates and caps before collecting data. Use the
paper's PICOC-style concept groups as the documented starting point. A search
through OpenAlex alone is a provider substitution, not an exact recreation of
the authors' publisher searches. `search.toml` records the adapted concept groups
and a 2015-01-01 through 2025-09-25 date window. The cutoff is the preprint's
submission date, an adapter choice because the original search date is unavailable.

The collector uses exact searches over titles and abstracts, rather than the
provider's general full-text search, following the
[OpenAlex search API](https://help.openalex.org/api/searching/). It retains papers
by their best per-query result rank, then query declaration order and stable ID.
That is provider relevance ranking, not random sampling. Both per-query caps
and the final retained-cohort cap are explicit. The `.collection.json` audit
preserves raw result pages, query membership, reported hit counts, selected IDs,
profile content and hashes; `complete` describes exhausting a query, not proving
that the cohort contains every relevant study. Reported hit counts are not
deduplicated counts of eligible papers.

A separately reviewed sample of excluded candidates can support a second,
mixed screening benchmark. Describe its sampling population and keep its
interpretation distinct from recovery of the eleven published inclusions.
If this case is used to tune queries or prompts, reserve independent reviewed
material for claims about generalization.

## Validation

```bash
MPLCONFIGDIR=/tmp/ids-meta-mpl python -m unittest tests.test_study_screening tests.test_study_pipeline tests.test_reporting -v
```

Tests exercise threshold boundaries, missing evidence, definite exclusions,
fabricated quotes, identity collisions, CLI export, blind model prompts,
resumption, query caps, the reference crosswalk, and independently verified
evaluation/database bundles. Handwritten test
predictions establish software behavior, not classifier performance.
