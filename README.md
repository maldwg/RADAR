# IDS Dataset and Repository Meta-Study

This repository contains two reproducible research pipelines. GitHub discovery builds a role-labelled catalogue of meaningful IDS/NDR repositories. OpenAlex discovery finds papers that introduce IDS/attack datasets or data-generation frameworks; papers that merely apply an existing dataset are retained only in the audit trail.

Both pipelines use the same stages:

```text
collection -> recall-oriented keyword prefilter -> row-level Ollama adjudication -> consensus -> reconciliation -> export -> plots
                                                                                 \-> rated evaluation artifacts
```

The keyword score only controls which records reach the LLM. It is not a final relevance label, and keyword scores/signals are deliberately omitted from model prompts.

## Environment

The project targets Python 3.11. The existing thesis environment already contains the runtime dependencies:

```bash
/home/max/venv/phd/bin/python -m pip install -e .
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
ids-meta github evaluate
ids-meta github reconcile
ids-meta github export
ids-meta github plot

ids-meta openalex collect
ids-meta openalex prefilter
ids-meta openalex classify
ids-meta openalex evaluate
ids-meta openalex reconcile
ids-meta openalex export
ids-meta openalex plot
```

`ids-meta run github` and `ids-meta run openalex` execute all stages. Collection and LLM adjudication have separate limits. The defaults for both sources retrieve at most 1,000 records and classify the 100 strongest non-rejected prefilter results:

```bash
ids-meta github run --collect-limit 1000 --classify-limit 100
ids-meta openalex run --collect-limit 1000 --classify-limit 100
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

GitHub discovery appends the configured minimum-star qualifier (100 by default) to every query while retaining the API native best-match order; it does not sort by stars, updated time, or creation date. Each query contributes at most x = `github.query_sample_size` repositories (100 by default). After cross-query deduplication, the collector selects the best y = `github.collection_limit` by native best-match rank and records that explicit active collection. Prefiltering reads only that collection, while older cached repositories remain available for audit.

Classification is intentionally row-level, but canonical selection is not. The `reconcile` stage runs after all available decisions are present:

- GitHub repositories are grouped into project families using immutable fork/parent metadata, generic name variants, evidence-grounded upstream repositories, explicit repository links, and shared project metadata. Multi-member families use `canonical` for the selected base; one-member projects use `standalone_project`, and derivative-only families use `base_repository_missing` instead of promoting packaging or tooling.
- OpenAlex papers are grouped by the generically extracted artifact name and aliases. Original, versioned, and generator releases remain visible, while one paper is labelled `canonical_base`; duplicate publication/preprint records are labelled `alternate_manifestation`.
- Structural ranking handles clear families without another model call. Only close cases can receive one group-level Ollama adjudication over the complete family. Stars and citation counts are never supplied as selection criteria.

This makes the result independent of classification order: seeing a packaging repository or version paper first cannot permanently make it the canonical base. Re-running reconciliation after more candidates are collected updates the family decision.

Classifier audit plots are separate from thesis-facing catalogue plots:

```bash
ids-meta compare github
ids-meta compare openalex
```

Generate the combined architecture figure independently from source plotting. By default it writes `pipeline_architecture.svg` in the repository root:

```bash
ids-meta architecture
# Alternate directory:
ids-meta architecture --output artifacts/architecture
# Direct module entry point:
python -m ids_meta.architecture
```

### Multiple models and ensembles

Set `ollama.models` to run the model-dependent stages once per model. Collection and prefiltering remain shared, so every model receives the same ranked candidates. Leave the array empty to retain the scalar `ollama.model` setting and all legacy paths. When a non-empty array is configured, it takes precedence over `ollama.model`.

```toml
[ollama]
model = "qwen3.6:27b"
models = ["qwen3.6:27b", "llama3.3:70b", "gemma3:27b"]

[ensemble]
enabled = true
# Empty means every ollama.models entry; otherwise list only participating models.
models = ["qwen3.6:27b", "llama3.3:70b"]
threshold = 0.5
```

Each source can also repeat the same classification consecutively with `<source>.consecutive_runs` and vote using `<source>.consensus_threshold`. Consensus covers the include/exclude/uncertain disposition and normalized classification fields. Single-value fields use threshold voting; GitHub scopes, capabilities, and methods use per-label voting; derived system types are recomputed. The assembled tuple is checked against deterministic cross-field rules. Invalid mixtures fall back to an audited coherent supporting run, or remain uncertain when no valid fallback exists. Set `consecutive_runs = 1` for the former single-run behavior. See [classification consensus](docs/classification_consensus.md) for the complete rules, independence limitations, and audit columns.

The ensemble first maps `accept` and OpenAlex `accept_adjacent` to include, while `exclude` and `uncertain` remain separate dispositions. A unique disposition must reach the threshold among all selected ensemble members. Missing/error results remain in the denominator, and tied or below-threshold votes become `uncertain`. The ensemble CSV preserves the representative classification plus member names, each model decision, exact decision counts, all disposition scores, agreement, and vote status. Its reconciliation is deterministic and does not ask another model to override the vote.

With two or more configured models, outputs are isolated under `data/interim/<source>/models/<model-slug>/`, `data/processed/<source>/models/<model-slug>/`, and `artifacts/plots/<source>/models/<model-slug>/`. Ensemble equivalents use an `ensemble/` directory. Cross-model decision counts, pairwise agreement, and record-level disagreement tables/plots are written to `artifacts/plots/comparisons/<source>/models/`. The shared cache remains safe because decision fingerprints already include the model name. `ids-meta compare <source>` detects this configuration and rebuilds the cross-model bundle. In multi-model or ensemble mode, model-dependent `--output` paths and stage-specific `--input` paths for reconcile, export, or plot are rejected because one file cannot represent every variant; a shared candidate `--input` remains supported for classify and run.

Use `--config path/to/config.toml`, `--input`, and `--output` for alternate runs. The default generic discovery queries and weighted concept groups live in `config/terms/`. Production term profiles must contain domain concepts only, never named products, repositories, datasets, or benchmarks.

The compact `config/default.toml` contains runtime settings and paths. Classifier prompts and taxonomy explanations live separately in `config/prompts/github.toml` and `config/prompts/openalex.toml`; the `[paths]` table selects those profiles, so a custom main config can point at alternatives. Python contains no fallback prompt text. Both `template` and `reconciliation_template`, every required taxonomy table, and every normalized enum description are mandatory; missing content produces a configuration error before an LLM request. GitHub classification supports `${roles}`, `${scopes}`, `${capabilities}`, `${methods}`, and `${evidence}`; OpenAlex classification supports `${roles}`, `${tiers}`, `${artifact_types}`, `${domains}`, `${data_forms}`, `${environments}`, `${labeling}`, and `${evidence}`. Reconciliation templates use `${evidence}` for the complete candidate family. `${evidence}` is mandatory, and unknown placeholders also fail before an LLM request. Unlike keyword concept groups, classifier taxonomy keys are schema-controlled: descriptions may be edited, but adding a new output type also requires extending its response schema. Relevant prompt configuration is included in the decision fingerprint, so prompt edits invalidate only incompatible cached decisions.

The LLM never supplies a confidence score. Classification decisions use categorical roles plus deterministic validation gates. A GitHub `standalone_detector` additionally requires runtime-observation evidence, a runnable detector path, and must not be limited to offline datasets; failed gates are exported in `classification_validation_flags`. Extracted upstream repositories must occur in verified evidence. `canonical_confidence` is a coarse code-derived indicator for multi-member structural ranking only; singleton and missing-base families leave confidence and margin blank.

GitHub system types are defined independently in `config/taxonomy/github_system_types.toml`. Each `[types.CODE]` table has a readable name, description, priority, and generic selectors over scopes, capabilities, methods, and roles. Add a new table to introduce another derived type without changing Python. Specific matches suppress generic fallback types (`IDS` and `IPS`); multiple equally applicable specific types remain visible. `PIDS` explicitly means protocol-based IDS, while provenance-based IDS uses `ProvIDS`. The catalogue exports `primary_system_type`, `system_types`, and `system_type_names`; `derived_labels` remains as a compatibility alias. Type rules are included in classification fingerprints.

### Rated evaluation and regression fixtures

Central human ratings live in `config/evaluation/github.toml` and `config/evaluation/openalex.toml`. `ids-meta <source> evaluate` compares saved classifications with those ratings and writes operational, strict, and selective DR/FPR/ACC/PREC/F1 metrics plus entity, confusion, and field-level audit tables under `artifacts/evaluation/`. Set `[evaluation].enabled = true` to execute this stage automatically after classification in a full run. Multi-model and ensemble results receive separate evaluation directories. See [rated classification evaluation](docs/evaluation.md) for profile examples, exact denominators, missing-result behavior, output files, and tests.

`ids_meta.common.regression.evaluate_regression(source, expected, rated_entities, classification_fields)` remains available for small plain-dictionary unit fixtures without persisted run artifacts.

## Data products

- `data/raw/<source>/cache.sqlite3`: resumable API evidence, coverage records, and fingerprinted LLM decisions.
- `data/interim/github/candidates.csv`: every GitHub prefilter result.
- `data/interim/github/classifications.csv`: every evaluated GitHub accept/exclude/uncertain/error decision.
- `data/interim/github/reconciled.csv`: complete GitHub audit plus global project-family and canonical fields.
- `data/processed/github/repositories.csv`: one combined accepted, role-labelled repository catalogue with canonical relationships and explicit `primary_system_type`, `system_types`, and `system_type_names` fields.
- `data/interim/openalex/paper_candidates.csv`: work-level OpenAlex prefilter results.
- `data/interim/openalex/survey_sources.csv`: auxiliary survey records used only for reference discovery; these do not consume collection or classification capacity.
- `data/interim/openalex/classifications.csv`: every evaluated paper decision. `classification_validation_flags` explains why a plausible release remained uncertain, `model_evidence_json` preserves the quotes returned by the model, and `evidence_json` contains only quotes verified against the supplied source section.
- `data/interim/openalex/reconciled.csv`: complete paper audit plus artifact-family, manifestation, and canonical-base fields.
- `data/processed/openalex/base_papers.csv`: security-intended dataset and generator base papers. `title` is the publication title, `artifact_name` is the canonical LLM-extracted resource identity, and `artifact_display_name`/`resource_name` is the explicit concise label used in tables and plots.
- `data/processed/openalex/adjacent_base_papers.csv`: neutral foundational data releases.
- `artifacts/plots/`: source-specific SVG/PDF catalogue figures plus the exact aggregate CSV behind each plot.
- `artifacts/evaluation/`: rated-set metrics and auditable instance, confusion, and field tables for each source/model/ensemble.
- `pipeline_architecture.svg`: the combined repository-level data-flow architecture, generated independently with `ids-meta architecture`. Each component states its input data, transformation, and output data.
- `artifacts/runs/`: configuration fingerprints, source coverage, counts, and the explicit API-coverage limitation for each command.

Popularity fields are descriptive metadata only. OpenAlex candidate papers must meet the configured `minimum_citations` eligibility floor (50 by default); among eligible papers, citations never determine retrieval ranking, relevance, or canonical-family selection. Auxiliary survey sources are exempt from the floor.

The GitHub plot bundle starts with one tall, scrollable feature matrix of every accepted repository and a lossless `data/00_repository_overview.csv`. It groups repositories by normalized system family and shows role dots, system labels, methods, activity, stars, forks, canonical/variant status, and latest commit time. New collections obtain the exact latest commit timestamp; older cached records transparently use `pushed_at` and mark that fallback in the CSV.

The OpenAlex bundle starts with a scrollable, one-row-per-resource publication timeline. The classifier extracts `artifact_display_name` separately from the publication title: it prefers an explicit short name or acronym, never invents one, and otherwise supplies a concise evidence-grounded descriptor. Existing pre-`openalex-3` rows fall back to their shortest recorded canonical name or alias until they are reclassified.

The remaining figures deliberately avoid overloaded scatterplots. Citation context is an ordered dot plot whose logarithmically spaced axis is labelled with actual citation counts. Discovery provenance is an exact row matrix; domain/data-form and capture/label relationships are annotated count heatmaps; growth uses cumulative counts with integer axes; and classifier outcomes show counts plus within-prefilter-band percentages. Survey sources use three aligned panels for retrieval score, citations, and referenced-work count. Labels occupy dedicated rows rather than being placed on top of data points. Exact citation, provenance, score, and identifier tables are exported beside every figure. Large SVG/PDF canvases are intended for scrolling and zooming.

OpenAlex discovery samples x = `openalex.query_sample_size` native results independently from every direct query, deduplicates the combined pool, and retains the best y = `openalex.collection_limit`. Survey queries use their own `survey_query_sample_size`; snowball references use `reference_sample_size`. Survey detection and release-exclusion phrases live in `[survey_identification]` in `config/terms/openalex.toml`. Title evidence has a configurable additional weight (`openalex.retrieval_title_weight`) so an abstract that merely mentions many relevant concepts cannot completely hide an obviously weak title. Blank or structurally uninformative titles and results below the generic retrieval floor are not retained. Retrieval, title, combined rank, and survey-source scores are exported in the audit CSVs. Citations are logged and exported but never enter retrieval ranking. The selected work IDs are recorded as the active collection: later prefiltering reads this pool rather than mixing in stale works from older collection strategies, while those historical raw records remain in SQLite for audit and reuse.

Survey discovery runs before direct discovery so it cannot be starved by the collection limit. Surveys themselves are ranked using generic positive evidence, with each selected survey's score, citations, and reference-list size logged. They are auxiliary records and are excluded from both the collection limit and LLM shortlist. For a 250-paper collection with `survey_reference_fraction = 0.35`, the target allocation is 163 direct candidates plus 87 new snowballed candidates, in addition to the separately configured survey sources. Unused snowball capacity is backfilled with the next strongest direct papers. Reference IDs mentioned by multiple selected surveys are resolved first. References shared by at least `survey_consensus_threshold` selected surveys (two by default) may bypass the generic text-score floor, and survey-source count takes priority when filling the snowball budget. This preserves opaque dataset papers whose OpenAlex metadata does not expose their release contribution. Each retained paper records both `survey_reference` and every originating `survey_reference:<OpenAlex survey ID>` relation. Run manifests and coverage records expose scanned/selected result counts and bounded query/reference lists.

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
