# IDS Dataset and Repository Meta-Study

This repository contains two reproducible research pipelines. GitHub discovery builds a role-labelled catalogue of meaningful IDS/NDR repositories. OpenAlex discovery finds papers that introduce IDS/attack datasets or data-generation frameworks; papers that merely apply an existing dataset are retained only in the audit trail.

Both pipelines use the same stages:

```text
collection -> recall-oriented keyword prefilter -> row-level Ollama adjudication -> consensus -> reconciliation -> export -> plots
                         \-> keyword baseline ----\
                         \-> zero-shot NLI baseline +-> rated evaluation -> unified classifier plots
```

The keyword score only controls which records reach the LLM. It is not a final relevance label, and keyword scores/signals are deliberately omitted from model prompts.

## Environment

The project targets Python 3.11. The existing thesis environment already contains the runtime dependencies:

```bash
/home/max/venv/phd/bin/python -m pip install -e .
# Include the local zero-shot NLI baseline:
/home/max/venv/phd/bin/python -m pip install -e '.[nli]'
```

Credentials and model selection are environment-only:

```bash
export GITHUB_TOKEN=...
export OPENALEX_MAILTO=researcher@example.org
export OPENALEX_API_KEY=...          # optional
export SEMANTIC_SCHOLAR_API_KEY=... # optional, raises missing-evidence lookup limits
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=your-model
```

Do not put secrets into `config/default.toml`.

## Commands

Each stage can be run separately and resumed:

```bash
ids-meta github collect
ids-meta github prefilter
ids-meta github classify
ids-meta github baseline
ids-meta github evaluate
ids-meta github reconcile
ids-meta github export
ids-meta github plot

ids-meta openalex collect
ids-meta openalex prefilter
ids-meta openalex classify
ids-meta openalex baseline
ids-meta openalex evaluate
ids-meta openalex reconcile
ids-meta openalex export
ids-meta openalex plot
```

Evaluation uses external reference sets declared by `[[evaluation.reference_sets]]`: GHML23 for GitHub repository screening and the survey-derived 105-dataset SHIELD profile for OpenAlex artifacts. `ids-meta <source> evaluate` rescans saved classification CSVs only; it does not collect data or call a classifier.

The primary survey profile retains the 105 datasets from Table 2 of Ashraf and Masoodi (2026) that belong to the requested types: general network, host/endpoint, IoT/IIoT, cloud, mobile/Android, electrical/smart-grid, UAV, and VPN. It preserves the survey’s original attributes and crosswalks category, format, traffic origin, and label availability into this project’s taxonomy. The other 18 published rows—Botnet, IOV, MANET-network, and satellite categories—are explicitly recorded as excluded profile categories rather than forced into an incorrect type. The article states 124 datasets, but its published table contains 123 rows (1–124 with row 122 absent), which the fixture records explicitly.

The profile is a positive-reference dataset-recall benchmark. Artifact names and aliases are matched first; an accepted release paper cited by the survey is used as a fallback identity match. Because several survey citations are analyses or documentation rather than original releases, all other citation outcomes remain separate diagnostics: classified non-release, classification uncertain/error, retrieved but not classified, not retrieved, or unresolved OpenAlex reference. Consequently, the report separates actual classifier abstention from the unresolved-artifact rate.

### Baselines

Run the deterministic keyword and local zero-shot NLI baselines, evaluate both, and rebuild the unified comparison plots without reclassifying any saved LLM result:

```bash
ids-meta github baseline
ids-meta openalex baseline
# Or select one method:
ids-meta github baseline --methods nli
ids-meta openalex baseline --methods keyword
# Re-evaluate and replot already saved baseline outputs; no classifier is initialized:
ids-meta github baseline --evaluate-only
ids-meta openalex baseline --methods nli --evaluate-only
```

The command processes every row in the candidate audit, including discovery-band `reject` rows. Keyword relevance thresholds are configured under `[baselines.keyword.github]` and `[baselines.keyword.openalex]`. A result is accepted only when its broad evidence score reaches `accept_threshold` and its configured role is eligible; the middle band abstains, explicit exclusion roles are excluded, and a low-score result without an eligible role is excluded. This keeps the screening gate separate from taxonomy assignment.

Keyword taxonomy rules live in `config/baselines/github_keyword.toml` and `config/baselines/openalex_keyword.toml`. Each source-complete profile defines exclusive or multilabel fields, normalized exact phrases, optional label weights, a minimum score, and an exclusive-label margin. Ties and insufficient evidence use the configured fallback instead of declaration-order guessing. Every label score and matched phrase is retained in the classification CSV, and changing either the screening vocabulary or taxonomy profile invalidates only incompatible cached keyword decisions. These rules are fixed independently of the evaluation fixtures and require no labeled training data.

NLI uses the revision-pinned model and inference settings under `[baselines.nli]`. Source-specific candidate-label hypotheses live in `config/baselines/github_nli.toml` and `config/baselines/openalex_nli.toml`. The strongest two representative evidence chunks are averaged; a label is emitted only when its entailment threshold and winning-label margin pass, otherwise the field abstains. GitHub role and OpenAlex paper role map directly to the decision, while the remaining hypotheses independently produce taxonomy fields. Scores, supporting chunks, model revision, thresholds, evidence source, and inference time are retained in every CSV row. Profile/model/threshold/chunk changes invalidate incompatible NLI cache entries. On resume, compatible decisions may be restored from the decision cache or the existing classification CSV before the transformer backend is constructed.

The predeclared `[baselines.nli.sensitivity]` grid is evaluated by replaying those saved aggregate scores. It never calls the model and never changes the configured operating point. One six-panel heatmap under `classifiers/nli/<model>/sensitivity/` shows what happens as the minimum entailment score and minimum winner gap become stricter; the black outline marks the configured cutoffs. This is a robustness and abstention trade-off diagnostic for the same NLI classifier, not another classifier result and not automatic threshold selection.

Keyword, NLI, each LLM, and the ensemble emit the same classification schema. Evaluation compares only fields that the active external reference set actually supplies; GHML23 directly rates screening decisions and only assigns `out_of_scope` to its negative controls rather than inventing fine-grained labels for its broad `secumon` positives. All rated results now live under one source-oriented root: `artifacts/evaluation/<source>/`. Start with `comparison/00_metric_overview`, which places DR, FNR, FPR, FDR, precision, accuracy, decision-F1, coverage, abstention, and taxonomy scores against every available approach. Larger per-metric figures are under `comparison/metrics/`, taxonomy comparisons under `comparison/taxonomy/`, agreement diagnostics under `comparison/agreement/`, and one-classifier audit details under `classifiers/`. The OpenAlex comparison uses only the survey-derived dataset benchmark. Positive-only fixtures show non-estimable false-positive, false-discovery, accuracy, precision, and decision-F1 results explicitly as N/A rather than drawing misleading values. Exact values, confusion counts, per-label TP/FP/FN, classifier IDs, rating paths, and Wilson intervals are in companion CSVs. Their interval level is set by `evaluation.confidence_level` (0.95 by default), and the normal quantile is derived from that setting rather than hardcoded.

With `[baselines].enabled = true`, complete `run` commands execute the configured baseline methods after the main pipeline and regenerate these comparisons. Explicit `baseline` commands always run regardless of that switch. `ids-meta compare github` or `ids-meta compare openalex` rebuilds comparisons from all evaluated classifier outputs currently present on disk.

`ids-meta run github` and `ids-meta run openalex` execute all stages. Collection and LLM adjudication have separate limits. The audit-oriented defaults retain at most 5,000 records per source and classify every retained row; keyword scores prioritize the work but do not reject an active record:

```bash
ids-meta github run --collect-limit 5000 --classify-all
ids-meta openalex run --collect-limit 5000 --classify-all
```

To keep discovery bounded but classify every row that passed the prefilter rejection threshold, use `--classify-all`:

```bash
ids-meta github run --collect-limit 1000 --classify-all
ids-meta openalex run --collect-limit 1000 --classify-all
```

“All” is still threshold-bound: it includes every `candidate` and `borderline` row, which means scores at or above the source's configured `borderline_threshold`; `reject` rows remain in the candidate audit and are never sent to the LLM. Set `classification_limit = "all"` under `[github]` or `[openalex]` for the same behavior by default. A numeric `classification_limit` or `--classify-limit N` retains the deterministic top-N shortlist.

The shortlist is deterministic: `candidate` rows precede `borderline` rows, higher prefilter scores come first within each band, and the stable source ID breaks ties. Cached decisions count toward the classification limit, so resume cannot silently expand a 100-row shortlist. Configure persistent defaults as `github.collection_limit`, `github.classification_limit`, `openalex.collection_limit`, and `openalex.classification_limit` in TOML.

For a single `collect` or `classify` command, `--limit N` remains a shorthand for that stage. On `run`, legacy `--limit N` sets both limits unless `--classify-all` explicitly removes the classification cap; prefer the explicit options when the limits differ. `--classify-all` and `--classify-limit` are mutually exclusive. `--dry-run` avoids Ollama calls during classification; collection dry-runs inspect the existing cache without network access.

Commands log timestamped progress to stdout at `INFO` level by default, including searches, date-shard splitting, API retries/rate limits, enrichment, cache reuse, prefilter totals, each LLM decision, checkpoints, exports, and generated figures. Redirect stdout to retain a complete run log:

```bash
ids-meta github collect > artifacts/github-collect.log 2>&1
ids-meta --log-level DEBUG openalex classify
```

Set `logging.level` and `logging.progress_every` in the TOML configuration for persistent defaults. `--log-level DEBUG|INFO|WARNING|ERROR` overrides the configured level for one command.

API pacing is enforced inside the HTTP clients before every request, including retries and enrichment calls. The defaults are conservative and configurable:

```toml
[github]
request_pause_seconds = 2.0
search_pause_seconds = 3.0

[openalex]
request_pause_seconds = 1.0
```

GitHub search uses its own stricter interval because it has a smaller rate-limit bucket than ordinary repository API calls. Startup logs report the active intervals; DEBUG logs show individual pacing sleeps.

GitHub pacing also adapts to the `X-RateLimit-*` response headers and spreads remaining calls over the quota window. A rate-limit wait therefore does not consume ordinary retry attempts. If startup reports `authentication=anonymous`, export `GITHUB_TOKEN` or `GH_TOKEN`; anonymous collection is technically supported but repository enrichment will be extremely slow.

GitHub discovery appends the configured minimum-star qualifier (zero by default) to every query. `github.search_sort` accepts `best_match`, `stars`, `forks`, `help-wanted-issues`, or `updated`, and `github.search_order` accepts `asc` or `desc`; order is ignored for best match. Best match is the default because popularity and recency sorting introduce different sampling biases. Each compact Boolean synonym query contributes at most x = `github.query_sample_size` repositories (1,000 by default). After cross-query deduplication, the collector selects the best y = `github.collection_limit` by the configured API ordering and records that explicit active collection. Search caches are fingerprinted by the ordered query profile and sorting strategy, preventing a changed protocol from reusing candidates found only by an older query set. Prefiltering reads only the active collection, while older cached repositories remain available for audit.

Classification is intentionally row-level, but canonical selection is not. The `reconcile` stage runs after all available decisions are present:

- GitHub repositories are grouped into project families using immutable fork/parent metadata, generic name variants, evidence-grounded upstream repositories, explicit repository links, and shared project metadata. The selected base is `canonical` for both multi-member and singleton families; singleton selection remains explicit as `canonical_resolution=singleton_project`. Derivative-only families use `base_repository_missing` instead of promoting packaging or tooling.
- OpenAlex papers are grouped by the generically extracted artifact name and aliases. Original, versioned, and generator releases remain visible, while one paper is labelled `canonical_base`; duplicate publication/preprint records are labelled `alternate_manifestation`.
- Structural ranking handles clear families without another model call and gives no repository role an inherent advantage; it uses fork status, base-like naming, explicit inbound links, direct self-description, and commit evidence. Only close cases can receive one group-level Ollama adjudication over the complete family. Stars and citation counts are never supplied as selection criteria.

This makes the result independent of classification order: seeing a packaging repository or version paper first cannot permanently make it the canonical base. Re-running reconciliation after more candidates are collected updates the family decision.

Classifier audit plots are separate from thesis-facing catalogue plots:

```bash
ids-meta compare github
ids-meta compare openalex
```

Generate both paper-ready architecture views independently from source plotting. The command writes `pipeline_architecture.{svg,pdf}` for the detailed five-stage method and `system_architecture.{svg,pdf}` for the additional high-level component view:

```bash
ids-meta architecture
# Alternate directory:
ids-meta architecture --output artifacts/architecture
# Select one vector format:
ids-meta architecture --formats pdf
# Direct module entry point:
python -m ids_meta.architecture
```

The detailed pipeline figure labels the five paper stages `(i)`–`(v)` and keeps GitHub and OpenAlex discovery, reconciliation, and export paths visually distinct. The separate system figure shows the GitHub and OpenAlex APIs, crawlers, evidence database, pipeline client, Ollama server, configured LLMs, and outputs.

All catalogue, evaluation, comparison, and funnel builders use `plots.formats`.
The default `formats = ["svg", "pdf"]` therefore writes a matching PDF beside
every SVG figure.

Generate the source-to-catalogue reverse-pyramid independently from collection or inference:

```bash
ids-meta funnel github
ids-meta funnel openalex
# Alternate directory:
ids-meta funnel github --output artifacts/paper/funnel
# Direct module entry point:
python -m ids_meta.funnel github
```

The funnel reads the active source cache and compatible candidate, classification, and catalogue CSVs when they exist. If no database snapshot exists, it falls back to acquisition and collection capacities from the selected TOML configuration and marks those segments as configured capacity rather than observed results. Query counts are displayed as acquisition context instead of being compared geometrically with record counts. Every normal `classify`, `baseline`, or complete `run` refreshes the default bundle under `artifacts/plots/<source>/pipeline_funnel/`; custom-input/output experiments remain opt-in through the standalone command. Multi-model and baseline outputs receive separate funnels with a shared discovery denominator. Each transition labels the exact number and percentage removed, while companion CSVs preserve the counts, provenance, and denominators.

### Multiple models and ensembles

Set `ollama.models` to run the model-dependent stages once per model. Collection and prefiltering remain shared, so every model receives the same ranked candidates. Leave the array empty to retain the scalar `ollama.model` setting and all legacy paths. When a non-empty array is configured, it takes precedence over `ollama.model`.

```toml
[ollama]
model = "qwen3.6:27b"
models = ["qwen3.6:27b", "llama3.3:70b", "gemma3:27b"]
context_length = 262144
keep_alive = "30m"

[ensemble]
enabled = true
# Empty means every ollama.models entry; otherwise list only participating models.
models = ["qwen3.6:27b", "llama3.3:70b"]
threshold = 0.5
```

`ollama.context_length` is sent as `options.num_ctx` on every classification and
reconciliation request, overriding Ollama's server-selected context for this
pipeline. Use `262144` for a 256 Ki-token context window. Set it to `0` to omit
the request-level override and use the Ollama server default. `ollama.keep_alive`
is sent with every request so Ollama keeps that model loaded between consecutive
items when memory permits. Context capacity and keep-alive are runtime controls,
not semantic decision settings, so changing them does not invalidate compatible
cached classifications. Resume remains enabled unless `--no-resume` is passed.
A compatible decision produced while context capacity was briefly part of the
cache key is migrated lazily on its first cache hit; this does not call Ollama.
New processes log Ollama-reported `load`, `prompt_eval`, and `eval` durations for
every request; a large `load` duration is direct evidence of a model reload.

Each source can also repeat the same classification consecutively with `<source>.consecutive_runs` and vote using `<source>.consensus_threshold`. Consensus covers the include/exclude/uncertain disposition and normalized classification fields. Single-value fields use threshold voting; GitHub scopes, capabilities, and methods use per-label voting; derived system types are recomputed. The assembled tuple is checked against deterministic cross-field rules. Invalid mixtures fall back to an audited coherent supporting run, or remain uncertain when no valid fallback exists. Set `consecutive_runs = 1` for the former single-run behavior. See [classification consensus](docs/classification_consensus.md) for the complete rules, independence limitations, and audit columns.

The ensemble first maps `accept` and OpenAlex `accept_adjacent` to include, while `exclude` and `uncertain` remain separate dispositions. A unique disposition must reach the threshold among all selected ensemble members. Missing/error results remain in the denominator, and tied or below-threshold votes become `uncertain`. The ensemble CSV preserves the representative classification plus member names, each model decision, exact decision counts, all disposition scores, agreement, and vote status. Its reconciliation is deterministic and does not ask another model to override the vote.

With two or more configured models, outputs are isolated under `data/interim/<source>/models/<model-slug>/`, `data/processed/<source>/models/<model-slug>/`, and `artifacts/plots/<source>/models/<model-slug>/`. Ensemble equivalents use an `ensemble/` directory. Unrated cross-model decision diagnostics from catalogue plotting are written under `artifacts/plots/<source>/comparison/models/`; rated correctness and agreement comparisons are kept separately under `artifacts/evaluation/<source>/comparison/`. The shared cache remains safe because decision fingerprints already include the model name. `ids-meta compare <source>` rebuilds the rated cross-classifier bundle. In multi-model or ensemble mode, model-dependent `--output` paths and stage-specific `--input` paths for reconcile, export, or plot are rejected because one file cannot represent every variant; a shared candidate `--input` remains supported for classify and run.

Use `--config path/to/config.toml`, `--input`, and `--output` for alternate runs. The default generic discovery queries and weighted concept groups live in `config/terms/`. OpenAlex uses parenthesized `AND`/`OR` synonym blocks across title, abstract, and indexed full text. GitHub uses smaller unparenthesized `OR` groups because repository search does not reliably share OpenAlex's grouping behavior. Production term profiles must contain domain concepts only, never named products, repositories, datasets, or benchmarks.

The compact `config/default.toml` contains runtime settings and paths. Classifier prompts and taxonomy explanations live separately in `config/prompts/github.toml` and `config/prompts/openalex.toml`; the `[paths]` table selects those profiles, so a custom main config can point at alternatives. Python contains no fallback prompt text. Both `template` and `reconciliation_template`, every required taxonomy table, and every normalized enum description are mandatory; missing content produces a configuration error before an LLM request. GitHub classification supports `${roles}`, `${scopes}`, `${capabilities}`, `${methods}`, and `${evidence}`; OpenAlex classification supports `${roles}`, `${tiers}`, `${artifact_types}`, `${domains}`, `${data_forms}`, `${environments}`, `${labeling}`, and `${evidence}`. Reconciliation templates use `${evidence}` for the complete candidate family. `${evidence}` is mandatory, and unknown placeholders also fail before an LLM request. Unlike keyword concept groups, classifier taxonomy keys are schema-controlled: descriptions may be edited, but adding a new output type also requires extending its response schema. Relevant prompt configuration is included in the decision fingerprint, so prompt edits invalidate only incompatible cached decisions.

The LLM never supplies a confidence score. GitHub classification uses the roles `native_detector`, `classifier_engine`, `detection_framework`, `detection_platform`, `research_project`, `demo`, `rules_content`, and `support_tooling`, plus audit-only `out_of_scope` and `uncertain`. Deterministic gates require native detectors to implement their own runtime engine; classifier engines to implement detection logic with a reproducible execution path; detection frameworks to additionally expose a cohesive agent/manager or inseparable multi-subsystem architecture; detection platforms to reproducibly integrate multiple independent detectors; and research projects to combine a research basis with a reproducible train, inference, evaluation, or detection path. Failed gates are exported in `classification_validation_flags`. For GitHub reporting, `native_detector`, `classifier_engine`, and `detection_framework` are the operational-system segment; all other accepted roles remain reported separately. Extracted upstream repositories must occur in verified evidence. `canonical_confidence` is a coarse code-derived indicator for multi-member structural ranking only; singleton and missing-base families leave confidence and margin blank.

GitHub system types are defined independently in `config/taxonomy/github_system_types.toml`. Each `[types.CODE]` table has a readable name, description, priority, and generic selectors over scopes, capabilities, methods, and roles. Add a new table to introduce another derived type without changing Python. Specific matches suppress generic fallback types (`IDS` and `IPS`); multiple equally applicable specific types remain visible. `PIDS` explicitly means protocol-based IDS, while provenance-based IDS uses `ProvIDS`. The catalogue exports `primary_system_type`, `system_types`, and `system_type_names`. Type rules are included in classification fingerprints.

### External reference-set evaluation

The `[evaluation].reference_sets` array in `config/default.toml` selects `config/evaluation/github_ghml23.toml` for GHML23 and `config/evaluation/openalex_ashraf_2026.toml` for SHIELD. The GHML23 TOML profile declaratively maps the untouched upstream semicolon-delimited CSV into this project's decisions (`secumon` → accept; `pentest` and `imageocr` → exclude/out-of-scope). It intentionally does not manufacture role, scope, method, or system-type labels for positive records. `ids-meta <source> evaluate` compares saved classifications with the configured source reference set. Each classifier keeps machine-readable audit tables at `artifacts/evaluation/<source>/classifiers/<family>/...`, with its diagnostic figures and companion data isolated below `figures/`. Cross-classifier reporting is in the sibling `comparison/` directory, and a source-level README points to the correct starting figure. Set `[evaluation].enabled = true` to execute this stage automatically after classification in a full run. Reference profiles and their imported CSVs are evaluation-only: they are absent from LLM decision fingerprints, so editing or replacing a reference set cannot invalidate cached classifications. Use the explicit `evaluate` command when only rescoring/replotting is wanted. See [rated classification evaluation](docs/evaluation.md) for profile details, exact denominators, missing-result behavior, output files, and tests.

`ids_meta.common.regression.evaluate_regression(source, expected, rated_entities, classification_fields)` remains available for small plain-dictionary unit fixtures without persisted run artifacts.

## Data products

- `data/raw/<source>/cache.sqlite3`: resumable API evidence, coverage records, and fingerprinted LLM decisions.
- `data/interim/github/candidates.csv`: every GitHub prefilter result.
- `data/interim/github/classifications.csv`: every evaluated GitHub accept/exclude/uncertain/error decision.
- `data/interim/github/baselines/keyword/classifications.csv`: deterministic keyword role, scope, capability, method, system-type, and decision baseline over the complete candidate audit, including per-label score and matched-term audit.
- `data/interim/github/baselines/nli/<model-slug>/classifications.csv`: zero-shot NLI role/taxonomy decisions with complete score and evidence-chunk audit.
- `data/interim/github/reconciled.csv`: complete GitHub audit plus global project-family and canonical fields.
- `data/processed/github/repositories.csv`: one combined accepted, role-labelled repository catalogue with canonical relationships and explicit `primary_system_type`, `system_types`, and `system_type_names` fields.
- `data/interim/openalex/paper_candidates.csv`: work-level OpenAlex prefilter results.
- `data/interim/openalex/survey_sources.csv`: auxiliary survey records used only for reference discovery; these do not consume collection or classification capacity.
- `data/interim/openalex/classifications.csv`: every evaluated paper decision. `classification_validation_flags` explains why a plausible release remained uncertain, `model_evidence_json` preserves the quotes returned by the model, and `evidence_json` contains only quotes verified against the supplied source section.
- `data/interim/openalex/baselines/keyword/classifications.csv`: deterministic keyword paper-role, relevance, artifact, domain, representation, capture, labeling, and decision baseline over the complete paper-candidate audit, including per-label score and matched-term audit.
- `data/interim/openalex/baselines/nli/<model-slug>/classifications.csv`: zero-shot NLI paper-role/taxonomy decisions with complete score and evidence-chunk audit.
- `data/interim/openalex/reconciled.csv`: complete paper audit plus artifact-family, manifestation, and canonical-base fields.
- `data/processed/openalex/base_papers.csv`: security-intended dataset and generator base papers. `title` is the publication title, `artifact_name` is the canonical LLM-extracted resource identity, and `artifact_display_name`/`resource_name` is the explicit concise label used in tables and plots.
- `data/processed/openalex/adjacent_base_papers.csv`: neutral foundational data releases.
- `artifacts/plots/`: source-specific SVG/PDF catalogue figures plus the exact aggregate CSV behind each plot.
- `artifacts/plots/<source>/pipeline_funnel/`: database-aware, paper-ready reverse pyramids for every current classifier variant, including exact transition-loss CSVs and configuration-only fallbacks.
- `artifacts/evaluation/<source>/comparison/`: the primary rated cross-approach overview, metric, taxonomy, and agreement figures.
- `artifacts/evaluation/<source>/classifiers/`: per-classifier metrics, auditable instance/confusion/field tables, diagnostic figures, and NLI sensitivity analysis.
- `pipeline_architecture.{svg,pdf}`: the detailed five-stage paper pipeline.
- `system_architecture.{svg,pdf}`: the additional high-level component architecture. Both are generated with `ids-meta architecture`.
- `artifacts/runs/`: configuration fingerprints, source coverage, counts, and the explicit API-coverage limitation for each command.

Popularity fields never determine eligibility or relevance. The default GitHub star and OpenAlex citation floors are both zero. Within an already verified OpenAlex artifact family, raw citation counts provide only a configurable, log-scaled, capped secondary signal for canonical-paper selection.

The GitHub plot bundle starts with one tall, scrollable feature matrix of every accepted repository and a lossless `data/00_repository_overview.csv`. It groups repositories by normalized system family and shows role dots, system labels, methods, activity, stars, forks, canonical/variant status, and latest commit time. New collections obtain the exact latest commit timestamp; older cached records transparently use `pushed_at` and mark that fallback in the CSV.

The OpenAlex bundle starts with a scrollable publication timeline: publication year is on x, citations use zero-safe logarithmic spacing on y, color identifies the dataset domain, and marker shape distinguishes datasets, generators, and combined dataset/generator releases. Same-year points receive small deterministic horizontal offsets. Every resource name is placed in a compact bordered callout connected to its marker by a domain-colored leader line; callouts are collision-spread within each year, and the vector canvas scales moderately with both result count and maximum same-year density. Point and label coordinates are recorded in the companion CSV. The classifier extracts `artifact_display_name` separately from the publication title: it prefers an explicit short name or acronym, never invents one, and otherwise supplies a concise evidence-grounded descriptor. Existing pre-`openalex-3` rows fall back to their shortest recorded canonical name or alias until they are reclassified.

The remaining figures deliberately avoid overloaded scatterplots. Citation context is an ordered dot plot whose logarithmically spaced axis is labelled with actual citation counts. Discovery provenance is an exact row matrix showing direct search, survey snowballing, and the number of distinct surveys citing each accepted resource; auxiliary survey papers remain in their separate survey-source figure. Domain/data-form and capture/label relationships are annotated count heatmaps; growth uses cumulative counts with integer axes; and classifier outcomes show counts plus within-prefilter-band percentages. Survey sources use three aligned panels for retrieval score, citations, and referenced-work count. Labels occupy dedicated rows rather than being placed on top of data points. Exact citation, provenance, score, and identifier tables are exported beside every figure. Large SVG/PDF canvases are intended for scrolling and zooming.

OpenAlex discovery samples x = `openalex.query_sample_size` native results independently from each Boolean synonym-block query, deduplicates the combined pool, and retains the best y = `openalex.collection_limit`. Survey queries use their own `survey_query_sample_size`; snowball references use `reference_sample_size`. Survey detection and release-exclusion phrases live in `[survey_identification]` in `config/terms/openalex.toml`. Title evidence has a configurable additional weight (`openalex.retrieval_title_weight`) so an abstract that merely mentions many relevant concepts cannot completely hide an obviously weak title. Blank or structurally uninformative titles and results below the configured generic retrieval floor are not retained. Retrieval, title, combined rank, and survey-source scores are exported in the audit CSVs. Citations are logged and exported but never enter retrieval ranking. The selected work IDs are recorded as the active collection: later prefiltering reads this pool rather than mixing in stale works from older collection strategies, while those historical raw records remain in SQLite for audit and reuse.

Survey discovery runs before direct discovery so it cannot be starved by the collection limit. Surveys themselves are ranked using generic positive evidence, with each selected survey's score, citations, and reference-list size logged. They are auxiliary records and are excluded from both the collection limit and LLM shortlist. For a 250-paper collection with the default `survey_reference_fraction = 0.5`, the target allocation is 125 direct candidates plus 125 new snowballed candidates, in addition to the separately configured survey sources. Unused snowball capacity is backfilled with the next strongest direct papers. Reference IDs mentioned by multiple selected surveys are resolved first. References shared by at least `survey_consensus_threshold` selected surveys (two by default) may bypass the generic text-score floor, and survey-source count takes priority when filling the snowball budget. This preserves opaque dataset papers whose OpenAlex metadata does not expose their release contribution. Each retained paper records both `survey_reference` and every originating `survey_reference:<OpenAlex survey ID>` relation. Run manifests and coverage records expose scanned/selected result counts and bounded query/reference lists.

## CSV conventions

- Schema version is `1.0`.
- Lists are semicolon-separated.
- Booleans are lowercase `true` or `false`.
- Timestamps use UTC ISO 8601.
- `unknown` means the evidence did not support a taxonomy value; an empty cell means the source did not provide metadata.
- Processed OpenAlex metadata always comes from the accepted work itself, not from an earlier keyword catalogue.
- When an OpenAlex work has no abstract, classification resolves it by DOI or exact normalized title through Semantic Scholar. Successful abstracts and optional open-access PDF text are cached, written back to the candidate audit, and included in the evidence hash; missing full text never blocks classification.
- OpenAlex release roles and artifact types must agree: dataset releases use `dataset` or `dataset_and_generator`; generator releases use `generator_framework` or `dataset_and_generator`.
- Accepted OpenAlex rows always expose `resource_name`; legacy rows without a dedicated display name are backfilled from an extracted alias or `artifact_name`, never from a hardcoded dataset registry.

## Testing

Tests are offline and use small source/LLM fixtures. They do not run GitHub, OpenAlex, or Ollama:

```bash
MPLCONFIGDIR=/tmp/ids-meta-matplotlib \
  PYTHONPATH=src \
  /home/max/venv/phd/bin/python -m unittest discover -s tests -v
```

The catalogues are bounded by source indexing and the recorded query coverage. They must not be described as exhaustive beyond that observable coverage.
