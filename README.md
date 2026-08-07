# IDS Dataset and Repository Meta-Study

This repository contains two reproducible research pipelines. GitHub discovery builds a role-labelled catalogue of meaningful IDS/NDR repositories. OpenAlex discovery finds papers that introduce IDS/attack datasets or data-generation frameworks; papers that merely apply an existing dataset are retained only in the audit trail.

Both pipelines use the same stages:

```text
collection -> recall-oriented keyword prefilter -> row-level Ollama adjudication -> global reconciliation -> export -> plots
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
ids-meta github reconcile
ids-meta github export
ids-meta github plot

ids-meta openalex collect
ids-meta openalex prefilter
ids-meta openalex classify
ids-meta openalex reconcile
ids-meta openalex export
ids-meta openalex plot
```

`ids-meta run github` and `ids-meta run openalex` execute all stages. Collection and LLM adjudication have separate limits. The defaults for both sources retrieve at most 1,000 records and classify the 100 strongest non-rejected prefilter results:

```bash
ids-meta github run --collect-limit 1000 --classify-limit 100
ids-meta openalex run --collect-limit 1000 --classify-limit 100
```

The shortlist is deterministic: `candidate` rows precede `borderline` rows, higher prefilter scores come first within each band, and the stable source ID breaks ties. Cached decisions count toward the classification limit, so resume cannot silently expand a 100-row shortlist. Configure persistent defaults as `github.collection_limit`, `github.classification_limit`, `openalex.collection_limit`, and `openalex.classification_limit` in TOML.

For a single `collect` or `classify` command, `--limit N` remains a shorthand for that stage. On `run`, legacy `--limit N` sets both limits; prefer the explicit options when the limits differ. `--dry-run` avoids Ollama calls during classification; collection dry-runs inspect the existing cache without network access.

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

Classification is intentionally row-level, but canonical selection is not. The `reconcile` stage runs after all available decisions are present:

- GitHub repositories are grouped into project families using immutable fork/parent metadata, generic name variants, explicit repository links, and shared project metadata. Each accepted row remains in the catalogue and receives a relationship such as `canonical`, `deployment_variant`, `rules_content`, `integration`, or `mirror`.
- OpenAlex papers are grouped by the generically extracted artifact name and aliases. Original, versioned, and generator releases remain visible, while one paper is labelled `canonical_base`; duplicate publication/preprint records are labelled `alternate_manifestation`.
- Structural ranking handles clear families without another model call. Only close cases can receive one group-level Ollama adjudication over the complete family. Stars and citation counts are never supplied as selection criteria.

This makes the result independent of classification order: seeing a packaging repository or version paper first cannot permanently make it the canonical base. Re-running reconciliation after more candidates are collected updates the family decision.

Classifier audit plots are separate from thesis-facing catalogue plots:

```bash
ids-meta compare github
ids-meta compare openalex
```

Use `--config path/to/config.toml`, `--input`, and `--output` for alternate runs. The default generic discovery queries and weighted concept groups live in `config/terms/`. Production term profiles must contain domain concepts only, never named products, repositories, datasets, or benchmarks.

## Data products

- `data/raw/<source>/cache.sqlite3`: resumable API evidence, coverage records, and fingerprinted LLM decisions.
- `data/interim/github/candidates.csv`: every GitHub prefilter result.
- `data/interim/github/classifications.csv`: every evaluated GitHub accept/exclude/uncertain/error decision.
- `data/interim/github/reconciled.csv`: complete GitHub audit plus global project-family and canonical fields.
- `data/processed/github/repositories.csv`: one combined accepted, role-labelled repository catalogue with canonical relationships.
- `data/interim/openalex/paper_candidates.csv`: work-level OpenAlex prefilter results.
- `data/interim/openalex/classifications.csv`: every evaluated paper decision.
- `data/interim/openalex/reconciled.csv`: complete paper audit plus artifact-family, manifestation, and canonical-base fields.
- `data/processed/openalex/base_papers.csv`: security-intended dataset and generator base papers.
- `data/processed/openalex/adjacent_base_papers.csv`: neutral foundational data releases.
- `artifacts/plots/`: SVG/PDF figures plus the exact aggregate CSV behind each plot.
- `artifacts/runs/`: configuration fingerprints, source coverage, counts, and the explicit API-coverage limitation for each command.

Popularity and citation fields are descriptive metadata only. They never determine relevance or inclusion.

The GitHub plot bundle starts with one tall, scrollable feature matrix of every accepted repository and a lossless `data/00_repository_overview.csv`. It groups repositories by normalized system family and shows role dots, system labels, methods, activity, stars, forks, canonical/variant status, and latest commit time. New collections obtain the exact latest commit timestamp; older cached records transparently use `pushed_at` and mark that fallback in the CSV.

The OpenAlex bundle starts with one tall chronological table of every canonical artifact family and `data/00_artifact_timeline_overview.csv`. Its publication timeline is stacked by `dataset`, `generator_framework`, and `dataset_and_generator` rather than merging those contributions into one count. Overview figures are never split into pages; large SVG/PDF canvases are intended for scrolling and zooming.

Survey discovery runs before direct discovery so it cannot be starved by `--limit`. Limited runs reserve `openalex.survey_reference_fraction` of their candidate capacity for papers cited by detected surveys. References are resolved in API batches, and each cached paper records both `survey_reference` and the originating `survey_reference:<OpenAlex survey ID>` provenance. Run manifests record survey-source, reference-work, and reference-edge counts; coverage records expose truncated reference lists and unresolved references.

## CSV conventions

- Schema version is `1.0`.
- Lists are semicolon-separated.
- Booleans are lowercase `true` or `false`.
- Timestamps use UTC ISO 8601.
- `unknown` means the evidence did not support a taxonomy value; an empty cell means the source did not provide metadata.
- Processed OpenAlex metadata always comes from the accepted work itself, not from an earlier keyword catalogue.

## Testing

Tests are offline and use small source/LLM fixtures. They do not run GitHub, OpenAlex, or Ollama:

```bash
MPLCONFIGDIR=/tmp/ids-meta-matplotlib \
  PYTHONPATH=src \
  /home/max/venv/phd/bin/python -m unittest discover -s tests -v
```

The catalogues are bounded by source indexing and the recorded query coverage. They must not be described as exhaustive beyond that observable coverage.
