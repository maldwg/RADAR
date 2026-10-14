# Shared provider adapters

Provider implementations live in `src/ids_meta/adapters/`. The discovery runner
in `common/discovery.py` owns query budgets, raw-response audits, deduplication,
selection and the frozen candidate file. Providers do not choose eligibility
rules, quality thresholds, extraction labels or prompts; those remain study files.

| Contract | Responsibility |
| --- | --- |
| `ProviderDescriptor` | Supported entity kinds and optional operations |
| `ProviderQuery` | Named, explicit provider-native text and parameters |
| `SearchPage` | Raw response, records, opaque continuation, coverage status |
| `SourceRecord` | Provider/native identity, strong identifiers, metadata, evidence, native extras |
| `DiscoveryAdapter` | Validate profiles and queries, retrieve pages and normalize individual records |
| Optional lookup, evidence, references and catalogue protocols | Expose supported operations without mandatory no-op methods |

`adapters/registry.py` is the registration point. Configuration validates registered
capabilities and entity kinds without network calls. GitHub and OpenAlex wrap
their paced clients. GitHub keeps its optional catalogue strategy for date
sharding and repository evidence. Both paper studies use the same common paper
collector for direct retrieval, survey traversal, backward snowballing and
ranking; their configurations select different values. OpenAlex requests and Semantic
Scholar discovery and evidence lookup use the shared JSON retry/pacing transport. All scholarly PDF
extraction uses the same bounded downloader in `common/pdf.py`.

## Multiple providers in one review

The checked-in conformal review still selects OpenAlex by default. Add this source
table to its `study.toml`, before `[evaluation]`, to enable the supplied Semantic
Scholar search:

```toml
[[sources]]
name = "semantic_scholar"
profile = "../../shared/sources/semantic_scholar.toml"
search = "search/semantic_scholar.toml"
```

```bash
ids-meta --study conformal_treatment_effects validate
ids-meta --study conformal_treatment_effects run
ids-meta --study conformal_treatment_effects collect --source semantic_scholar
```

Unqualified review stages use all configured providers. The combined candidate
file is `artifacts/studies/<study>/interim/cohort/candidates.json`. The `prefilter`
stage acquires shared evidence and writes `prefiltered.json`, which is the default
classification input. Run `collect`, `prefilter`, `classify` in that order, or
use `run`. Explicit classification input is prepared through the same stage.
Model outputs
use `interim/cohort/models/<model>/` and `interim/cohort/ensemble/`. Reports use
`plots/cohort/`. Explicit `--source` stages use `cohort_<provider>/` throughout,
so each selected provider has separate candidates, predictions and reports.

Each provider/query receives the shared three-times discovery budget before the
single union collection limit is applied. Each provider uses the study's
`collection_mode` and survey allocation; the union orders these selections by
their position, provider declaration order and native ID. Retrieval positions
reflect each API's ordering; they are not calibrated relevance scores. Semantic
Scholar bulk retrieval orders by native paper ID rather than relevance.

Queries are deliberately separate files: OpenAlex Boolean syntax is not silently
rewritten into Semantic Scholar syntax. The supplied Semantic Scholar query is
an additional retrieval strategy, not an assertion that the original review used
that source. Its bulk endpoint and token pagination follow the
[official API documentation](https://api.semanticscholar.org/api-docs/).
Authentication reads `SEMANTIC_SCHOLAR_API_KEY`; OpenAlex and GitHub retain their
existing environment variables.

The `.collection.json` audit retains every response page, profile fingerprint,
effective query cap, source recovery counts and candidate-file hash. Candidates
retain all provider/query memberships, native records and evidence variants.
Matching uses normalized DOI/arXiv or the same provider/native identity, always
within the same entity kind. Conflicting strong identifiers are flagged and kept
separate. Equal titles only produce possible-duplicate flags. A paper, repository
and artifact are separate entities. No title match automatically merges them.

Candidate identities use `doi:<normalized DOI>`, `arxiv:<unversioned ID>`, or
`<provider>:<native ID>` when no strong identifier is available. Native IDs remain
available separately. The study's reporting contract interprets the scientific
output schema; discovery providers use `SourceRecord` and do not own that schema.

Paper evidence policies are shared by the dataset and conformal studies. PDF
responses are reused within a cohort. With review ensembles and uncertain-evidence
enrichment enabled, the prefilter freezes available expanded evidence before the
models run, so both models assess the same text.

The same screening protocol assesses the combined cohort once per configured
model/repetition. Evaluation reports the union in `evaluation/`, and each
provider's retained cohort against the same references in
`evaluation/sources/<provider>/`. These measurements describe recovery within
the retained cohort, including its collection cap. References do not enter
discovery or assessment prompts. Provider membership is retained through CSV
export and is not substituted by the study's scientific reporting identity.

## Local files

To discover papers from a frozen CSV/JSON export, add `local` to the study's
sources using `../../shared/sources/local.toml` and a study-owned search profile:

```toml
schema_version = "1.0"
provider = "local"
path = "papers.csv"
identity = "export_id"
entity_kind = "paper"
named_entity_seeding = false

[mapping]
title = "paper_title"
doi = "DOI"
full_text = "text"
full_text_complete = "complete"
```

`path` resolves relative to that search file. Pagination follows file order and
records the content hash. A changed file aborts collection. Mappings accept
metadata and evidence fields only. Scientific judgments present in the export
are kept in the raw audit and are excluded from candidate assessment inputs.
`full_text_complete` must be boolean or the CSV string `true`/`false`; omitted
completeness defaults to false.

## Adding a provider or strategy

1. Implement `queries`, `validate_query`, `search` and `normalize` with the shared contracts.
   Return an opaque continuation and report API truncation through
   `SearchPage.complete`. Preserve native extras in `SourceRecord.native`.
2. Add a registration in `adapters/registry.py` with the factory and descriptor.
   Implement optional protocols only for operations that actually work.
3. Add a shared provider profile and a study-owned native search file. Select
   them through another `[[sources]]` table. Review configuration and the common
   runner need no provider-name dispatch changes.
4. Test pagination, normalization and an offline union cohort with an injected
   client. New provider transport can reuse `JsonTransport`; provider-specific
   rate-header handling belongs in its transport subclass.

The criterion-assessment review engine accepts any registered paper discovery
provider. The two catalogue engines still use their specialized repository-role
and dataset-release schemas, caches, baselines and reconciliation. A discovery
adapter alone does not implement a catalogue workflow: its descriptor must expose
an implemented catalogue strategy, and a new catalogue engine needs a scientific
workflow integration. Configuration rejects unsupported combinations up front.
There are no output migrations, automatic stale-result repairs or schema aliases.
