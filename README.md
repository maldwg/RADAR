# Data Gathering

This folder contains scripts for collecting metadata used in the IDS/dataset analysis.

## OpenAlex Dataset Discovery

`openalex_dataset_discovery.py` searches for IDS/security datasets and related dataset-generation frameworks using OpenAlex only. It is meant to find canonical dataset publications, not papers that merely train a detector on an existing dataset.

Example run:

```bash
python3 data-gathering/openalex_dataset_discovery.py \
  --dataset-search-per-page 100 \
  --max-reference-works 300 \
  --candidate-per-page 20 \
  --out data-gathering/ids_datasets_openalex.csv
```

### Workflow

The script combines several discovery sources:

1. Trusted survey/base papers supplied with `--trusted-paper` or `--trusted-papers-file`.
2. Automatically discovered IDS dataset surveys, unless `--no-auto-surveys` is used.
3. Dataset names extracted from OpenAlex titles and, optionally, table-like contexts in open-access PDFs.
4. References from trusted survey papers, unless `--no-reference-snowball` is used.
5. Direct OpenAlex searches for dataset-publication and generator/framework papers.

The output entity is the dataset or dataset-generation artifact, not every paper that mentions it.

### Rating And Filtering

Candidate papers are scored with a heuristic. Positive signals include:

- dataset or benchmark wording in the title
- introduction/release wording such as `introduced`, `presented`, `created`, `publicly available`
- security or traffic scope such as intrusion detection, attacks, botnets, DNS, ICS/SCADA, IoT, CAN, logs, or network traffic
- the dataset/artifact name appearing in the title
- DOI, citation count, and mentions from multiple trusted sources
- generator/framework wording such as traffic generation, synthetic attacks, attack injection, dataset toolkit, or anomaly modeling engine

OpenAlex abstracts, keywords, and topics are not fetched or used to decide whether a paper proposes a dataset. Candidate/base-paper decisions are title-first because abstract phrase matching was too imprecise for distinguishing proposed datasets from downstream usage.

Negative filters reject likely false positives:

- surveys and reviews
- detector/classifier/model papers
- papers that only use, benchmark, compare, or analyze an existing dataset
- generic names such as `ICS`, `CAN`, `AI/ML`, or `CICIDS`

The `artifact_type` column separates classic datasets from related generation tools:

- `Dataset`
- `Generator/framework`
- `Dataset + generator/framework`

### Outputs

For an output path like `ids_datasets_openalex.csv`, the script writes:

- `ids_datasets_openalex.csv`: accepted dataset/artifact entities
- `ids_datasets_openalex_candidates.csv`: candidate canonical papers and scores
- `ids_datasets_openalex_rejected.csv`: rejected candidates with rejection reasons

The candidate and rejected files are important for manual auditing. They show where recall or precision can still be tuned.

### Useful Parameters

- `--trusted-paper`, `--trusted-papers-file`: manually provide trusted seed papers.
- `--no-auto-surveys`: use only manual trusted papers.
- `--direct-dataset-search` / `--no-direct-dataset-search`: enable or disable direct OpenAlex searches for dataset/framework papers.
- `--dataset-query`: add a custom dataset/framework search query.
- `--reference-snowball` / `--no-reference-snowball`: inspect or skip references from trusted papers.
- `--max-reference-works`: maximum trusted-paper references to inspect.
- `--candidate-pages`, `--candidate-per-page`: how deeply to search for canonical papers per dataset/artifact.
- `--candidate-query-profile compact|full`, `--max-aliases-per-entity`, `--max-candidate-queries-per-entity`: control candidate-resolution fanout. `compact` is the default; `full` restores the older broad query templates when you also raise the caps.
- `--mailto` or `OPENALEX_MAILTO`: identify the crawler to OpenAlex's polite pool; the crawler logs when this is missing.
- `--include-low-confidence`: include weak/unresolved entities in the accepted CSV.

## GitHub IDS Repository Crawler

`github-crawler.py` searches GitHub for open-source IDS/IPS/security-monitoring repositories, scores them, filters out support/research/false-positive repositories, and writes a compact CSV for plotting.

Useful cache-only rebuild:

```bash
python3 data-gathering/github-crawler.py \
  --evaluated-cache-in data-gathering/ids_repos.evaluated.csv \
  --cache-only \
  --out data-gathering/ids_repos.csv
```

Useful interrupted-run continuation:

```bash
python3 data-gathering/github-crawler.py \
  --evaluated-cache-in data-gathering/ids_repos.evaluated.csv \
  --evaluated-cache-out data-gathering/ids_repos.evaluated.csv \
  --out data-gathering/ids_repos.csv
```

Backfill missing latest commit SHAs from an existing evaluated cache:

```bash
python3 data-gathering/github-crawler.py \
  --evaluated-cache-in data-gathering/ids_repos.evaluated.csv \
  --evaluated-cache-out data-gathering/ids_repos.evaluated.csv \
  --cache-only \
  --refresh-commit-sha \
  --out data-gathering/ids_repos.csv
```

### Crawler Features In Execution Order

Keep this list in sync when changing `github-crawler.py`.

- Raises Python's CSV field-size limit so wide evaluated-cache rows with README/docs text can be read.
- Resolves GitHub authentication from `--token`, `GITHUB_TOKEN`, or `GH_TOKEN`.
- Selects request pacing: faster authenticated mode, or slow `--anonymous-friendly` public-API pacing.
- Loads the evaluated-repository cache from `--evaluated-cache-in` when provided.
- If `--cache-only` is used, skips GitHub search and uses only repositories from the evaluated cache.
- Otherwise searches GitHub with the default seed queries plus any `--extra-seed` queries.
- Deduplicates search results by repository full name while preserving all matched search seeds.
- Does not apply final relevance, role, or main-repository decisions during collection.
- Performs English/Latin parseability analysis on the available evidence text.
- Counts English words, unique English words, Latin letters, CJK characters, and English-vs-CJK share.
- Matches evidence phrases with exact normalized text first, then fuzzy token/concept matching for reordered words and close domain variants.
- Assigns preliminary core IDS/IPS hits such as IDS, IPS, HIDS, NIDS, WIDS, NDR, and NSM.
- Ignores curated product-signal priors for scoring, canonical/main-repository inference, and acceptance.
- Applies category evidence for solution families such as IDS, IPS, HIDS, NIDS, WIDS, PIDS, NIPS, HIPS, DBIDS, DBIPS, WIPS, and UNKNOWN, using the strongest matched phrase per family to avoid synonym overcounting.
- Applies negative/research/support evidence such as `survey`, `benchmark`, `awesome`, `tutorial`, `notebook`, `course project`, `deprecated`, or `unmaintained`.
- Infers repository kind from metadata, e.g. system, ruleset, tooling, framework, classifier, dataset, paper, list, or tutorial.
- Infers detection method/type, e.g. signature-based, rule-based, Log-based, packet-analysis, query-based, ML-based, DNN-based, ANN-based, anomaly-based, or UNKNOWN.
- Infers repository role as full IDS, classifier-only, or framework-only.
- Infers support descriptors for derivative/support repositories, e.g. rulesets, dashboards, integrations, deployment tooling, distributions, or wrappers.
- Demotes generic libraries, APIs, or SDKs when their IDS evidence is only phrased as a possible thing users could build with them.
- Infers a base-detector identity used for deduplication.
- Builds a same-owner repository-family identity from lexical tool stems, stripping side-variant terms such as `legacy`, `docker`, `packages`, `ansible`, `helm`, `flutter`, or `android`.
- Infers base-detector status as `base` or `support`, with an audit reason and score.
- Ranks collected candidates only for progress order, using stars and number of matched queries.
- Enriches every collected candidate before any final relevance, role, or main-repository decision is made.
- For every collected repository, reuses cached repository JSON/README/docs when available.
- If no usable cache exists, fetches repository detail JSON from GitHub.
- For fresh repository evaluations, fetches the latest visible default-branch commit SHA and stores it as `last_commit_sha`.
- If `--refresh-commit-sha` is used, also fetches missing `last_commit_sha` values for cached repositories.
- Optionally fetches README text according to `--readme-mode`.
- Discovers documentation/homepage URLs from repository metadata and README text.
- Optionally fetches documentation text according to `--docs-mode` and `--max-docs-pages`.
- Recomputes the final score from repository metadata, topics, README text, and docs text.
- Infers a generic main-repository signal from repository shape, role, base-detector evidence, activity, and side-variant naming.
- Classifies all enriched repositories before applying acceptance filters.
- Rejects repositories below `--min-stars`.
- Rejects forks unless `--include-forks` is set.
- Rejects archived repositories unless `--include-archived` is set.
- Rejects repositories older than `--pushed-within-days` when that option is set.
- Rejects research/list/tutorial/dataset-style repositories unless allowed by `--include-research`.
- Rejects repositories below `--min-score`.
- Rejects repositories without significant parseable English evidence.
- Rejects support/derivative repositories unless they are base detectors or `--include-support-tools` is set.
- Rejects repositories with no core IDS/IPS hits.
- Rejects adjacent-method-only rows where the only inferred method is packet-analysis, threat-intel, or an unclear `* tool (method unclear)` label.
- Rejects RASP-like rows unless they also have strong standalone IDS wording and a non-risky detector method.
- Rejects integrity-only rows unless they also have strong standalone IDS wording plus base-detector evidence.
- Rejects full-role repositories whose solution family remains UNKNOWN.
- Rejects tooling repositories without strong direct or product-relation evidence.
- Marks accepted rows as fully fledged IDS, classifier-only, or framework-only.
- Writes a wide evaluated cache row for every evaluated repository, accepted or rejected.
- Stores repository JSON, README text, docs text, docs URLs, matched queries, score card JSON, errors, `last_commit_sha`, and key metadata in the evaluated cache.
- Writes the partial output CSV after batch classification according to `--checkpoint-every`.
- Periodically writes the evaluated cache according to `--evaluated-cache-out`.
- Deduplicates accepted rows by base-detector identity and same-owner repository-family identity.
- Keeps the highest-quality row per identity, preferring likely main repositories over side variants, fully fledged IDS rows, stronger base-detector evidence, more stars, and higher score.
- Sorts final rows by stars and score.
- Truncates final rows to `--max-repos`.
- Assigns final rank numbers.
- Writes the final compact CSV to `--out`.
- Prints a short top-match summary.

### GitHub Crawler Outputs

For an output path like `ids_repos.csv`, the crawler can write:

- `ids_repos.csv`: compact accepted repository list used by the plotting scripts.
- `ids_repos.partial.csv`: checkpointed compact accepted list during long runs.
- `ids_repos.evaluated.csv`: wide cache/audit file for every evaluated repository.

The evaluated cache is the most important debugging artifact. It records why a repository was accepted or rejected and lets later runs avoid re-fetching the same repositories.

## LLM Repository Classification Outputs

`github-llm-classifier.py` reads the keyword crawler's evaluated cache and keeps three deliberately separate outputs:

- `ids_repos.llm_classified.csv`: wide, resumable intermediate/audit output. It includes accepted, excluded, uncertain, and errored model decisions. Do not use this as a final plotting input.
- `ids_repos.llm_accepted.csv`: wide accepted output for `figure_creation/github-classifier-comparison/compare_keyword_llm_classifiers.py`. It contains every accepted catalogue role, including deployable IDS, support tooling, frameworks, rules/content, classifier models, and research demos.
- `ids_repos.llm.csv`: compact accepted output compatible with `figure_creation/ids_github_repo_plots_improved.py`.

The two final outputs can be regenerated from an existing intermediate file without running Ollama again:

```bash
python3 data-gathering/github-llm-classifier.py \
  --export-catalogue-from data-gathering/ids_repos.llm_classified.csv
```

Create the standard repository figures from the compact LLM catalogue:

```bash
python3 figure_creation/ids_github_repo_plots_improved.py \
  --input data-gathering/ids_repos.llm.csv \
  --out-dir figure_creation/github-plots-llm
```
