.. _adapters-shared-provider-adapters:

Shared provider adapters
========================

Provider implementations live in ``src/radar/adapters/``. The discovery runner in
``common/discovery.py`` owns query budgets, raw-response audits, deduplication, selection and the
frozen candidate file. Providers do not choose eligibility rules, quality thresholds, extraction
labels or prompts; those remain study files.

.. list-table::
   :header-rows: 1
   :class: radar-table

   * - Contract
     - Responsibility
   * - ``ProviderDescriptor``
     - Supported entity kinds and optional operations
   * - ``ProviderQuery``
     - Named, explicit provider-native text and parameters
   * - ``SearchPage``
     - Raw response, records, opaque continuation, coverage status
   * - ``SourceRecord``
     - Provider/native identity, strong identifiers, metadata, evidence, native extras
   * - ``DiscoveryAdapter``
     - Validate profiles and queries, retrieve pages and normalize individual records
   * - Optional lookup, evidence, references and catalogue protocols
     - Expose supported operations without mandatory no-op methods

``adapters/registry.py`` is the registration point. Configuration validates registered capabilities
and entity kinds without network calls. GitHub and OpenAlex wrap their paced clients. GitHub keeps
its optional catalogue strategy for date sharding and repository evidence. Both paper studies use
the same common paper collector for direct retrieval, survey traversal, backward snowballing and
ranking; their configurations select different values. OpenAlex requests and Semantic Scholar
discovery and evidence lookup use the shared JSON retry/pacing transport. All scholarly PDF
extraction uses the same bounded downloader in ``common/pdf.py``.

Provider contracts and optional capabilities
--------------------------------------------

``DiscoveryAdapter`` is the common required Python protocol. Implementations
provide ``queries``, ``validate_query``, ``search``, and ``normalize``, together
with a ``ProviderDescriptor``. A provider can satisfy the protocol without
inheriting a shared base class. Registry checks validate the required members
and advertised capabilities; they do not replace signature checks or behavioral
tests. See :doc:`api` for the definitions.

Additional protocols describe operations supported by the same provider object:

.. list-table::
   :header-rows: 1
   :class: radar-table

   * - Capability
     - Operation
     - Current implementation
   * - ``LookupAdapter``
     - Retrieve a known record by native or supported external identifier
     - OpenAlex and Semantic Scholar
   * - ``EvidenceAdapter``
     - Obtain additional evidence for an existing candidate
     - Semantic Scholar abstract and PDF enrichment
   * - ``ReferenceAdapter``
     - Retrieve cited-work identifiers for configured snowballing
     - OpenAlex
   * - ``CatalogueAdapter``
     - Invoke the provider's supported catalogue collection strategy
     - GitHub repositories and OpenAlex dataset papers

The current catalogue capability includes workflow orchestration. Catalogue
support therefore requires more than implementing discovery, and remains a
boundary of the current implementation.

``JsonTransport`` is a shared HTTP helper used by OpenAlex and Semantic Scholar,
not another discovery provider. It handles request pacing, timeouts, retries,
headers, and JSON decoding. Adapters own endpoints, query syntax, pagination,
and response normalization. GitHub retains its separate rate-aware client.

``LocalDiscoveryAdapter`` imports candidate metadata and evidence for screening
from a frozen file. ``LocalSourceAdapter`` in ``common/sources.py`` instead
imports existing records and scientific results for reporting and evaluation.
These are distinct entry points; candidate discovery does not reuse imported
classification decisions as evidence.

.. _adapters-multiple-providers-in-one-review:

Multiple providers in one review
--------------------------------

The checked-in conformal review still selects OpenAlex by default. Add this source table to its
``study.toml``, before ``[evaluation]``, to enable the supplied Semantic Scholar search:

.. code:: toml

   [[sources]]
   name = "semantic_scholar"
   profile = "../../shared/sources/semantic_scholar.toml"
   search = "search/semantic_scholar.toml"

.. code:: bash

   radar --study conformal_treatment_effects validate
   radar --study conformal_treatment_effects run
   radar --study conformal_treatment_effects collect --source semantic_scholar

Unqualified review stages use all configured providers. The combined candidate file is
``artifacts/studies/<study>/interim/cohort/candidates.json``. The ``prefilter`` stage acquires
shared evidence and writes ``prefiltered.json``, which is the default classification input. Run
``collect``, ``prefilter``, ``classify`` in that order, or use ``run``. Explicit classification
input is prepared through the same stage. Model outputs use ``interim/cohort/models/<model>/`` and
``interim/cohort/ensemble/``. Reports use ``plots/cohort/``. Explicit ``--source`` stages use
``cohort_<provider>/`` throughout, so each selected provider has separate candidates, predictions
and reports.

Each provider/query receives the shared three-times discovery budget before the single union
collection limit is applied. Each provider uses the study’s ``collection_mode`` and survey
allocation; the union orders these selections by their position, provider declaration order and
native ID. Retrieval positions reflect each API’s ordering; they are not calibrated relevance
scores. Semantic Scholar bulk retrieval orders by native paper ID rather than relevance.

Queries are deliberately separate files: OpenAlex Boolean syntax is not silently rewritten into
Semantic Scholar syntax. The supplied Semantic Scholar query is an additional retrieval strategy,
not an assertion that the original review used that source. Its bulk endpoint and token pagination
follow the `official API documentation <https://api.semanticscholar.org/api-docs/>`__.
Authentication reads ``SEMANTIC_SCHOLAR_API_KEY``; OpenAlex and GitHub retain their existing
environment variables.

The ``.collection.json`` audit retains every response page, profile fingerprint, effective query
cap, source recovery counts and candidate-file hash. Candidates retain all provider/query
memberships, native records and evidence variants. Matching uses normalized DOI/arXiv or the same
provider/native identity, always within the same entity kind. Conflicting strong identifiers are
flagged and kept separate. Equal titles only produce possible-duplicate flags. A paper, repository
and artifact are separate entities. No title match automatically merges them.

Candidate identities use ``doi:<normalized DOI>``, ``arxiv:<unversioned ID>``, or
``<provider>:<native ID>`` when no strong identifier is available. Native IDs remain available
separately. The study’s reporting contract interprets the scientific output schema; discovery
providers use ``SourceRecord`` and do not own that schema.

Paper evidence policies are shared by the dataset and conformal studies. PDF responses are reused
within a cohort. With review ensembles and uncertain-evidence enrichment enabled, the prefilter
freezes available expanded evidence before the models run, so both models assess the same text.

The same screening protocol assesses the combined cohort once per configured model/repetition.
Evaluation reports the union in ``evaluation/``, and each provider’s retained cohort against the
same references in ``evaluation/sources/<provider>/``. These measurements describe recovery within
the retained cohort, including its collection cap. References do not enter discovery or assessment
prompts. Provider membership is retained through CSV export and is not substituted by the study’s
scientific reporting identity.

.. _adapters-local-files:

Local files
-----------

To discover papers from a frozen CSV/JSON export, add ``local`` to the study’s sources using
``../../shared/sources/local.toml`` and a study-owned search profile:

.. code:: toml

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

``path`` resolves relative to that search file. Pagination follows file order and records the
content hash. A changed file aborts collection. Mappings accept metadata and evidence fields only.
Scientific judgments present in the export are kept in the raw audit and are excluded from candidate
assessment inputs. ``full_text_complete`` must be boolean or the CSV string ``true``/``false``;
omitted completeness defaults to false.

.. _adapters-adding-a-provider-or-strategy:

Adding a provider or strategy
-----------------------------

1. Implement ``queries``, ``validate_query``, ``search`` and ``normalize`` with the shared
   contracts. Return an opaque continuation and report API truncation through
   ``SearchPage.complete``. Preserve native extras in ``SourceRecord.native``.

2. Add a registration in ``adapters/registry.py`` with the factory and descriptor. Implement
   optional protocols only for operations that actually work.

3. Add a shared provider profile and a study-owned native search file. Select them through another
   ``[[sources]]`` table. Review configuration and the common runner need no provider-name dispatch
   changes.

4. Test pagination, normalization and an offline union cohort with an injected client. New provider
   transport can reuse ``JsonTransport``; provider-specific rate-header handling belongs in its
   transport subclass.

The criterion-assessment review engine accepts any registered paper discovery provider. The two
catalogue engines still use their specialized repository-role and dataset-release schemas, caches,
baselines and reconciliation. A discovery adapter alone does not implement a catalogue workflow: its
descriptor must expose an implemented catalogue strategy, and a new catalogue engine needs a
scientific workflow integration. Configuration rejects unsupported combinations up front. There are
no output migrations, automatic stale-result repairs or schema aliases.
