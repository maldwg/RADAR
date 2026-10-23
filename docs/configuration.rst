.. _configuration-study-configuration:

Study configuration
===================

Select a study before running a stage:

.. code:: bash

   radar --study ids_repositories validate
   radar --study ids_repositories run --collect-limit 100 --classify-all
   radar --study ids_datasets baseline --methods keyword
   radar --study conformal_treatment_effects collect --collect-limit 200
   radar --study conformal_treatment_effects prefilter
   radar --study conformal_treatment_effects classify --classify-limit 5

``validate`` reads the configuration, prompts, taxonomy, baselines and references without making API
or inference calls. ``--config /path/to/study.toml`` selects an explicit manifest, including a study
outside the checked-in configuration tree. The default study is ``ids_repositories``. Standalone
reporting and verification commands use their explicit inputs and do not require a study manifest.

.. _configuration-ownership:

Ownership
---------

.. code:: text

   config/
     shared/
       runtime.toml                    # logging and backend connection
       models.toml                     # model names, revisions and inference settings
       quality.toml                    # shared quality prompts, evidence limits and NLI defaults
       sources/*.toml                  # GitHub, OpenAlex and Semantic Scholar
     studies/
       ids_repositories/
         study.toml                    # entry point and explicit file references
         pipeline.toml                 # collection, evidence, classification and reconciliation
         screening.toml                # scientific eligibility and optional quality criteria
         taxonomy.toml                 # system types and label descriptions
         search/github.toml            # discovery queries and scoring vocabulary
         prompts/classification.toml   # classification and reconciliation templates
         baselines/
           settings.toml               # method selection and baseline thresholds
           keyword.toml                # weighted taxonomy phrases
           nli.toml                    # taxonomy hypotheses
           llm_nli.toml                # three-way NLI prompt
         evaluation/
           references.toml             # evaluation-only reference labels
           github_ghml23.csv            # immutable upstream snapshot
         reporting.toml                # figure settings
       ids_datasets/                   # same ownership, with OpenAlex search
       conformal_treatment_effects/     # paper eligibility/quality protocol and extraction

Shared profiles contain no study search terms or reference labels. Scientific settings belong to the
study that uses them. To select different models or transport settings for one study, reference
another explicit profile from its manifest. There is no implicit profile inheritance or override
chain: declaring the same execution setting in two referenced profiles is an error.

All three studies use the same file responsibilities. ``pipeline.toml`` has ``[collection]``,
``[evidence]``, ``[classification]`` and, where needed, ``[reconciliation]`` tables. Classification
repetitions and ensemble policy belong under ``[classification]``, with ensemble settings in
``[classification.ensemble]``. API transport settings remain in the shared source profile. Unknown
or misplaced pipeline options are rejected.

Both paper studies use ``OpenAlexAdapter`` and the same collector in ``common/paper_collection.py``.
Their ``pipeline.toml`` files declare identical keys in the same order; the loader requires the
shared paper schema. Their OpenAlex search profiles also share the same structure: named
``[[queries]]``, explicit search parameter and publication window, survey queries, survey
identification, publisher targets and retrieval weights. Topic vocabulary and scoring groups vary
with the study.

.. list-table::
   :header-rows: 1
   :class: radar-table

   * - Setting
     - IDS datasets
     - Conformal treatment effects
   * - ``collection_mode``
     - ``ranked`` by generic evidence
     - ``query_position``
   * - ``max_surveys`` / ``snowball_max_depth``
     - 100 / 10
     - 0 / 0
   * - ``survey_reference_fraction``
     - 0.5
     - 0.0
   * - ``fetch_full_text``
     - false
     - true
   * - Abstract / uncertain-text enrichment
     - true / true
     - false / false
   * - ``max_full_text_chars``
     - 60,000
     - 120,000
   * - ``reconciliation.strategy``
     - ``dataset_release``
     - ``none``

Survey and backward-reference collection can be enabled for conformal work by changing these values
and supplying survey queries. Providers without reference capabilities must keep both traversal
controls at zero. Direct, survey and reference budgets each have an explicit factor; all three
default to 3.0. Both studies use the same evidence service, prefilter scores and borderline gate.
``enrich_missing_full_text`` is declared in both paper studies. When full-text
fetching is active, it checks structured-text providers before parsing a PDF,
even when the candidate already supplies a working PDF link. Both studies use
``full_text_lookup_providers = ["openalex", "semantic_scholar"]``, independently
of which provider discovered the paper.

The shared policy is:

#. Try any TEI URL supplied with the candidate.
#. Resolve configured providers advertising ``tei_xml`` (currently OpenAlex),
   and try their identity-matched TEI documents.
#. If no usable TEI was obtained, parse supplied or already discovered PDFs with
   pypdf, including OpenAlex cached PDFs, repository locations and the candidate's
   own arXiv identity.
#. Only if those sources fail, resolve the remaining PDF-only providers
   (currently Semantic Scholar) and try their alternatives.

Provider order applies within these format tiers. A Semantic Scholar discovery
record can therefore use OpenAlex's cached GROBID TEI through its own DOI or an
unambiguous exact title match. Semantic Scholar's Academic Graph adapter supplies
PDF links and identifiers; it does not itself supply TEI. Conflicting identities
remain rejected. Failed, invalid or inaccessible TEI, including a missing content
API key, permits PDF fallback. No local GROBID server is required.

An empty provider list disables external lookups while retaining candidate TEI,
PDFs and the candidate's own arXiv identity. With enrichment disabled, supplied
TEI is still preferred, but no additional provider requests are introduced.
Already supplied full text stays frozen unless configured uncertainty expansion
requests new evidence. Having an abstract does not suppress full-text acquisition.
The ``structured-tei-before-pdf-v1`` policy is recorded in review checkpoints,
collection audits and classifier fingerprints. Re-run acquisition/prefiltering to
apply it to a previously frozen PDF corpus.
``enrich_missing_abstracts`` remains independent. The IDS dataset study can continue to fetch full
text only on uncertainty through ``enrich_uncertain_with_full_text``; the conformal study fetches
it before assessment. Provider lookup and document results are cached within the shared evidence
service. OpenAlex cached content uses ``OPENALEX_API_KEY``; the key is sent only to the content
service and is never added to stored source URLs. Without a key, the audit records
``api_key_required`` and the service continues to other available sources. Cached downloads count
against OpenAlex usage; see `OpenAlex full-text access <https://help.openalex.org/access/fulltext/>`__.
TEI extraction supports modern and older archive shapes, retains article text, appendices and inline
mathematical characters, and rejects ordinary HTML pages and abstract-only XML. The audit records
``full_text_format`` and ``full_text_source_sha256`` for the fetched document.
``full_text_attempts_json`` records lookup and download attempts, including HTTP failure details.
Failed-attempt audits alone do not cause a new LLM assessment.

At INFO level, ``full-text lookup complete`` separates ``links_found``,
``record_found_no_document``, ``no_match`` and ``lookup_failed``. A failed
identifier route remains in the audit even if title lookup subsequently succeeds;
HTTP 404 is an absent record, whereas HTTP 429, access failures and network errors
are request failures. Failed requests also produce a contextual WARNING.

``full-text download complete`` names the paper, provider, requested URL and
redirect destination, HTTP status, content type, extraction format and failure
reason. ``provider=candidate`` means a supplied candidate link; other provider
names identify the metadata service that supplied the alternative. In particular,
``reason=non_pdf_content`` can accompany HTTP 200 and ``text/html``: the request
returned a page rather than a PDF. No response body is logged. Displayed URLs
exclude credentials, query strings and fragments.

A final ``full-text retrieval complete`` line reports ``retrieved``,
``already_available``, ``existing_text_retained``, ``no_document_link``,
``lookup_incomplete``, ``download_failed``, ``no_extractable_text`` or
``api_key_required`` with provider/request counts, failure counts, text length
and elapsed time. Metadata-only lookup uses ``not_requested`` for full text.
``lookup_failures`` counts unresolved provider lookups; ``lookup_request_failures``
counts failed identifier/title routes even when a later route succeeds.
Lookup/download completion lines identify cache hits; elapsed time and cache
flags are runtime logs, not changes to frozen evidence. HTTP, redirect and media
diagnostics are retained in ``full_text_attempts_json`` and available
``full_text_http_status``, ``full_text_content_type``, ``full_text_final_url`` and
``full_text_failure_reason`` fields. Restart the process to load logging changes.

``fetch_full_text`` works in either study. ``classify_borderline = false`` excludes borderline
candidates from either scientific classifier.

Dataset release-family weights such as ``canonical_role_scores`` are scientific definitions in the
dataset ``taxonomy.toml`` under ``[reconciliation]``. They are separate from the shared operational
pipeline. Criterion assessment and dataset release validation remain distinct scientific strategies,
with their own labels, prompts and optional baseline profiles.

``screening.toml`` contains a protocol ID, relative ``taxonomy`` and ``prompt`` references,
``[eligibility]`` descriptions, optional ``[quality]`` criteria, ``[decision]`` and optional
``[provenance]`` attribution. Operational limits and evidence budgets are rejected in this file. The
screening profile is the single place selecting the scientific prompt and taxonomy for all selected
sources.

Classification label descriptions belong in ``taxonomy.toml``, under ``[prompt_tables.<table>]`` for
catalogue workflows. Prompt templates reference those tables rather than maintaining another copy of
their descriptions. The review taxonomy defines extraction descriptions and allowed values under
``[assessment_fields.<field>]``. Its screening file owns the eligibility and quality criteria and
explicitly references its taxonomy and prompt. Catalogue prompts expand ``${eligibility_E1}``, etc.
from their scientific profiles; ``${eligibility}`` can instead expand the complete eligibility
block. Every declared catalogue criterion must appear in the prompt. Review prompts use
``${eligibility}`` and ``${quality}`` to obtain the criteria as structured JSON.

.. list-table::
   :header-rows: 1
   :class: radar-table

   * - Study
     - Decision strategy
     - Quality assessment
   * - IDS repositories
     - ``repository_role_validation``
     - None
   * - IDS datasets
     - ``dataset_release_validation``
     - None
   * - Conformal treatment effects
     - ``criterion_assessment``
     - At least five of nine criteria

The catalogue strategies retain their existing structural role/release validators. Their scientific
eligibility instructions now come from the screening profile. Review criterion assessment supports
optional quality scoring; when quality is absent, no quality threshold is applied. Adding quality
criteria to a catalogue strategy is rejected because that engine does not produce those assessments.

.. _configuration-shared-collection-and-classification-controls:

Shared collection and classification controls
---------------------------------------------

Provider interfaces and multi-provider review configuration are described in
:doc:`shared adapters <adapters>`. Review stages collect all configured providers into
``interim/cohort/``; ``--source`` selects a separate provider cohort. Study criteria, prompts and
taxonomy apply to the resulting cohort regardless of its discovery providers.

All three checked-in studies retain at most 100 candidates and use ``query_limit_factor = 3.0``.
Each query retrieves up to three times the largest finite collection/classification limit before
deduplication and final selection. With ``classification_limit = "all"``, the default cap is
therefore 300 per query. Changing either numeric limit, including a CLI override, changes the
derived cap. The saved collection audit records the actual derived ``per_query_limit``; this is an
audit value rather than a configurable fixed limit.

``collection_limit = "all"`` and ``classification_limit = "all"`` are supported by all three
workflows. When both are unlimited, query pagination continues until the source reports exhaustion.
A finite classification limit still bounds query retrieval when collection is unlimited. The same
rules apply to ``--collect-all``, ``--classify-all``, ``--collect-limit`` and ``--classify-limit``.

All three defaults also use five consecutive assessments per model, a 0.8 consensus threshold and an
enabled ensemble with threshold 1.0. Review consensus votes on eligibility/quality judgments and
extraction fields, retaining selected source block IDs and exact source passages. Failed or unavailable assessments remain in
the denominator. A tied vote or a conflict between voted criteria and the voted disposition produces
uncertainty. Review ensembles require completed model checkpoints with the same selected cohort,
protocol, prompt and full-text evidence. These repeated votes are not independent statistical
confidence estimates.

All workflows select shared quality-assessment configuration through ``study.toml``
and define their own criteria and pass thresholds in ``screening.toml``; see
:doc:`quality_assessment` for thresholds, backend support and separate evaluation views.
The review retains its five-of-nine quality protocol. GitHub documentation controls and dataset
release-family reconciliation remain specific to their entities. Review release-family
reconciliation and keyword/NLI baselines are rejected because its scientific classifier does not
implement those operations. The single-pass ``plain_llm`` baseline is supported for all three
workflows; see :doc:`plain_llm` for model settings and evidence policies. Shared paper acquisition
and prefilter settings are implemented for both paper studies.

.. _configuration-manifest:

Manifest
--------

The manifest schema is ``2.0``; the individual profile schemas remain ``1.0``. Every file or
directory reference resolves relative to the file declaring it, including the reference sets and
``output_root``. No project-root or working-directory heuristics are used.

Every manifest declares ``pipeline = "pipeline.toml"``, ``screening = "screening.toml"``
and an explicit ``quality = "../../shared/quality.toml"`` reference. Source
entries contain only ``name``, ``profile`` and ``search``. Keyword/NLI profiles are declared at
study level alongside ``methods``; scientific prompts and taxonomy are selected by
``screening.toml``. Separate workflow-specific ``collection`` or ``assessment`` manifest tables are
unsupported.

The :download:`repository manifest <../config/studies/ids_repositories/study.toml>` selects the
shared runtime and models, local baseline/reporting settings, source profile, scientific profiles
and reference set. The :download:`dataset manifest <../config/studies/ids_datasets/study.toml>` owns
its separate OpenAlex configuration. The
:download:`review manifest <../config/studies/conformal_treatment_effects/study.toml>` selects a
generic eligibility/quality screening engine.

.. list-table::
   :header-rows: 1
   :class: radar-table

   * - Workflow
     - Discovery source
     - Research outputs
   * - ``repository_catalogue``
     - GitHub
     - Repository classifications and project catalogue
   * - ``dataset_catalogue``
     - OpenAlex
     - Dataset/generator release classifications and artifact catalogue
   * - ``systematic_review``
     - OpenAlex, Semantic Scholar
     - Evidence-backed criterion judgments and configured extraction fields

These are execution engines, not study-name branches. A new study can reuse an engine under an
arbitrary identifier. Source/workflow combinations are checked before execution. Semantic Scholar
supplies both paper discovery and optional scholarly evidence enrichment for reviews. Catalogue
workflows retain their specialized source restrictions. Unknown APIs cannot be enabled merely by adding their name to a manifest.

.. _configuration-adding-a-study:

Adding a study
--------------

1. Copy the directory of the closest existing study.
2. Set a unique ``id`` and ``output_root`` in ``study.toml``.
3. Edit the queries, taxonomy, scientific screening rules, templates and baseline settings.
   Configure limits, evidence acquisition and consensus in ``pipeline.toml``.

4. Replace the evaluation references independently of the scientific prompts.
5. Run ``radar --study <directory-name> validate``.

A review can define different eligibility rules, quality thresholds and extraction fields without
adding a named-study branch to the CLI. The repository and dataset engines retain their specialized
catalogue semantics; a fundamentally different catalogue schema requires changes to that engine.

.. _configuration-outputs-and-clean-state:

Outputs and clean state
-----------------------

The checked-in manifests isolate outputs under ``artifacts/studies/<study>/``:

.. code:: text

   raw/<source>/cache.sqlite3
   interim/<source>/                         # selected candidates
   interim/<source>/models/<model>/          # classifications and reconciliation
   interim/<source>/baselines/<method>/      # baseline results
   processed/<source>/models/<model>/        # exported catalogues
   plots/<source>/models/<model>/            # figures and their audit bundles
   evaluation/<source>/classifiers/          # catalogue classifier evaluations
   runs/                                    # stage manifests

The review workflow writes its candidate JSON under ``interim/<source>/`` and uses the same
model-specific classification paths. Its generic report contains its evaluation and reporting
database beneath the model report directory.

Use a clean output root for each independent experiment. ``--resume`` explicitly continues
compatible current-format work; it is disabled by default. Stage manifests record the study
identity, resolved configuration paths and profile hashes. Reference labels remain outside
classification prompts and decision fingerprints.

The former flat configuration layout is removed. There are no old-path aliases, automatic migrations
or stale-report repairs.
