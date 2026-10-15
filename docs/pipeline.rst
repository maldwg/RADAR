.. _pipeline-pipeline-operation-and-scientific-methods:

Pipeline operation and scientific methods
=========================================

.. _pipeline-commands:

Commands
--------

Each stage can be run separately and resumed:

.. code:: bash

   radar --study ids_repositories collect
   radar --study ids_repositories prefilter
   radar --study ids_repositories classify
   radar --study ids_repositories baseline
   radar --study ids_repositories evaluate
   radar --study ids_repositories reconcile
   radar --study ids_repositories export
   radar --study ids_repositories plot

   radar --study ids_datasets collect
   radar --study ids_datasets prefilter
   radar --study ids_datasets classify
   radar --study ids_datasets baseline
   radar --study ids_datasets evaluate
   radar --study ids_datasets reconcile
   radar --study ids_datasets export
   radar --study ids_datasets plot

Evaluation uses external reference sets declared by ``[[evaluation.reference_sets]]``: GHML23 for
GitHub repository screening and the survey-derived 105-dataset SHIELD profile for OpenAlex
artifacts. ``radar <source> evaluate`` rescans saved classification CSVs only; it does not collect
data or call a classifier.

The primary survey profile retains the 105 datasets from Table 2 of Ashraf and Masoodi (2026) that
belong to the requested types: general network, host/endpoint, IoT/IIoT, cloud, mobile/Android,
electrical/smart-grid, UAV, and VPN. It preserves the survey’s original attributes and crosswalks
category, format, traffic origin, and label availability into this project’s taxonomy. The other 18
published rows—Botnet, IOV, MANET-network, and satellite categories—are recorded as omitted source
categories rather than forced into an incorrect type. Figure 1 and SHIELD dataset descriptions are
used to audit the crosswalk; the survey fields remain unchanged, even where a source category is
wrong (notably LBNL). The article states 124 datasets, but its published table contains 123 rows
(1–124 with row 122 absent), which the fixture records explicitly.

The profile is a 105-row artifact benchmark with 96 core accepts, five adjacent accepts, and four
excluded audit controls. Artifact names and aliases are matched first; for positive artifacts only,
an accepted release paper cited by the survey is used as a fallback identity match. Because several
survey citations are analyses or documentation rather than original releases, all other citation
outcomes remain separate diagnostics: classified non-release, classification uncertain/error,
retrieved but not classified, not retrieved, or unresolved OpenAlex reference. Consequently, the
report separates actual classifier abstention from the unresolved-artifact rate.

.. _pipeline-baselines:

Baselines
~~~~~~~~~

Run the deterministic keyword and local zero-shot NLI baselines, evaluate both, and rebuild the
unified comparison plots without reclassifying any saved LLM result:

.. code:: bash

   radar --study ids_repositories baseline
   radar --study ids_datasets baseline
   # Or select one method:
   radar --study ids_repositories baseline --methods nli
   radar --study ids_datasets baseline --methods keyword
   # Re-evaluate and replot already saved baseline outputs; no classifier is initialized:
   radar --study ids_repositories baseline --evaluate-only
   radar --study ids_datasets baseline --methods nli --evaluate-only

The command processes every row in the candidate audit, including discovery-band ``reject`` rows.
Keyword relevance thresholds are configured under ``[baselines.keyword.github]`` and
``[baselines.keyword.openalex]``. A result is accepted only when its broad evidence score reaches
``accept_threshold`` and its configured role is eligible; the middle band abstains, explicit
exclusion roles are excluded, and a low-score result without an eligible role is excluded. This
keeps the screening gate separate from taxonomy assignment.

Keyword taxonomy rules live in ``config/studies/ids_repositories/baselines/keyword.toml`` and
``config/studies/ids_datasets/baselines/keyword.toml``. Each source-complete profile defines
exclusive or multilabel fields, normalized exact phrases, optional label weights, a minimum score,
and an exclusive-label margin. Ties and insufficient evidence use the configured fallback instead of
declaration-order guessing. Every label score and matched phrase is retained in the classification
CSV, and changing either the screening vocabulary or taxonomy profile invalidates only incompatible
cached keyword decisions. These rules are fixed independently of the evaluation fixtures and require
no labeled training data.

NLI uses the revision-pinned model and inference settings under ``[baselines.nli]``. Source-specific
candidate-label hypotheses live in ``config/studies/ids_repositories/baselines/nli.toml`` and
``config/studies/ids_datasets/baselines/nli.toml``. The strongest two representative evidence chunks
are averaged; a label is emitted only when its entailment threshold and winning-label margin pass,
otherwise the field abstains. GitHub role and OpenAlex paper role map directly to the decision,
while the remaining hypotheses independently produce taxonomy fields. Scores, supporting chunks,
model revision, thresholds, evidence source, and inference time are retained in every CSV row.
Profile/model/threshold/chunk changes invalidate incompatible NLI cache entries. On resume,
compatible decisions may be restored from the decision cache or the existing classification CSV
before the transformer backend is constructed.

The predeclared ``[baselines.nli.sensitivity]`` grid is evaluated by replaying those saved aggregate
scores. It never calls the model and never changes the configured operating point. One six-panel
heatmap under ``classifiers/nli/<model>/sensitivity/`` shows what happens as the minimum entailment
score and minimum winner gap become stricter; the black outline marks the configured cutoffs. This
is a robustness and abstention trade-off diagnostic for the same NLI classifier, not another
classifier result and not automatic threshold selection.

Keyword, NLI, each LLM, and the ensemble emit the same classification schema. Evaluation compares
only fields that the active external reference set actually supplies. The GHML23 snapshot retains
its published source categories as provenance, while its executable TOML profile explicitly
crosswalks every repository into the project’s role, observation-scope, observation-object,
capability, detection-method, and IDS-type vocabularies. All rated results now live under one
source-oriented root: ``artifacts/studies/<study>/evaluation/<source>/``. Start with
``comparison/00_metric_overview``, which places DR, FNR, FPR, FDR, precision, accuracy, decision-F1,
coverage, abstention, and taxonomy scores against every available approach. Larger per-metric
figures are under ``comparison/metrics/``, taxonomy comparisons under ``comparison/taxonomy/``,
agreement diagnostics under ``comparison/agreement/``, and one-classifier audit details under
``classifiers/``. The OpenAlex comparison uses only the survey-derived dataset benchmark.
Positive-only custom fixtures show non-estimable false-positive, false-discovery, accuracy,
precision, and decision-F1 results as N/A. The default mixed SHIELD fixture computes these for its
four selected negative controls, which are not representative of all excluded publications. Exact
values, confusion counts, per-label TP/FP/FN, classifier IDs, rating paths, and Wilson intervals are
in companion CSVs. Their interval level is set by ``evaluation.confidence_level`` (0.95 by default),
and the normal quantile is derived from that setting rather than hardcoded.

With ``[baselines].enabled = true``, complete ``run`` commands execute the configured baseline
methods after the main pipeline and regenerate these comparisons. Explicit ``baseline`` commands
always run regardless of that switch. ``radar --study ids_repositories compare github`` or
``radar --study ids_datasets compare openalex`` rebuilds comparisons from all evaluated classifier
outputs currently present on disk.

``radar --study ids_repositories run`` and ``radar --study ids_datasets run`` execute all stages.
Collection and LLM adjudication have separate limits. The audit-oriented defaults retain at most 100
records per source and classify every retained row; keyword scores prioritize the work but do not
reject an active record:

.. code:: bash

   radar --study ids_repositories run --collect-limit 5000 --classify-all
   radar --study ids_datasets run --collect-limit 5000 --classify-all

To keep discovery bounded but classify every row that passed the prefilter rejection threshold, use
``--classify-all``:

.. code:: bash

   radar --study ids_repositories run --collect-limit 1000 --classify-all
   radar --study ids_datasets run --collect-limit 1000 --classify-all

“All” is still threshold-bound: it includes every ``candidate`` and ``borderline`` row, which means
scores at or above the source’s configured ``borderline_threshold``; ``reject`` rows remain in the
candidate audit and are never sent to the LLM. Set ``classification_limit = "all"`` under
``[classification]`` in the selected study’s ``pipeline.toml`` for the same behavior by default. A
numeric ``classification_limit`` or ``--classify-limit N`` retains the deterministic top-N
shortlist.

Collection also supports ``collection_limit = "all"`` or ``--collect-all``. Omitted
collection/classification limits are unlimited. With a limit of 1,000 and
``query_limit_factor = 3.0``, each query can contribute up to 3,000 hits before deduplication and
final selection. If both limits are numeric, the larger controls this discovery cap. Logs and
collection snapshots record the effective caps.

The shortlist is deterministic: ``candidate`` rows precede ``borderline`` rows, higher prefilter
scores come first within each band, and the stable source ID breaks ties. Cached decisions count
toward the classification limit, so resume cannot silently expand a 100-row shortlist. Configure
``collection_limit`` under ``[collection]`` and ``classification_limit`` under ``[classification]``
in the study’s ``pipeline.toml``.

Use ``--collect-limit N`` and ``--classify-limit N`` to set the two stage budgets independently.
``--classify-all`` and ``--classify-limit`` are mutually exclusive. ``--dry-run`` avoids Ollama
calls during classification; collection dry-runs inspect the existing cache without network access.

Commands log timestamped progress to stdout at ``INFO`` level by default, including searches,
date-shard splitting, API retries/rate limits, enrichment, cache reuse, prefilter totals, each LLM
decision, checkpoints, exports, and generated figures. Redirect stdout to retain a complete run log:

.. code:: bash

   radar --study ids_repositories collect > artifacts/github-collect.log 2>&1
   radar --log-level DEBUG openalex classify

Set ``logging.level`` and ``logging.progress_every`` in the TOML configuration for persistent
defaults. ``--log-level DEBUG|INFO|WARNING|ERROR`` overrides the configured level for one command.

API pacing is enforced inside the HTTP clients before every request, including retries and
enrichment calls. Set ``request_pause_seconds`` and, for GitHub, ``search_pause_seconds`` under
``[settings]`` in ``config/shared/sources/<source>.toml``. The current request intervals are two
seconds for GitHub and one second for OpenAlex, with three seconds for GitHub search.

GitHub search uses its own stricter interval because it has a smaller rate-limit bucket than
ordinary repository API calls. Startup logs report the active intervals; DEBUG logs show individual
pacing sleeps.

GitHub pacing also adapts to the ``X-RateLimit-*`` response headers and spreads remaining calls over
the quota window. A rate-limit wait therefore does not consume ordinary retry attempts. If startup
reports ``authentication=anonymous``, export ``GITHUB_TOKEN`` or ``GH_TOKEN``; anonymous collection
is technically supported but repository enrichment will be extremely slow.

GitHub discovery appends the configured minimum-star qualifier (zero by default) to every query.
``collection.search_sort`` accepts ``best_match``, ``stars``, ``forks``, ``help-wanted-issues``, or
``updated``, and ``collection.search_order`` accepts ``asc`` or ``desc``; order is ignored for best
match. Best match is the default because popularity and recency sorting introduce different sampling
biases. Each compact Boolean synonym query scans at most ``collection.query_limit_factor`` (default
3) times the larger numeric collection/classification limit. With neither numeric limit, queries
scan until exhausted. Larger or unbounded searches use date sharding. After cross-query
deduplication, the collector selects the best y = ``collection.collection_limit`` by the configured
API ordering and records that explicit active collection. Search caches are fingerprinted by the
ordered query profile and sorting strategy, preventing a changed protocol from reusing candidates
found only by an older query set. Prefiltering reads only the active collection, while older cached
repositories remain available for audit.

Classification is intentionally row-level, but canonical selection is not. The ``reconcile`` stage
runs after all available decisions are present:

- GitHub repositories are grouped into project families using immutable fork/parent metadata,
  generic name variants, evidence-grounded upstream repositories, explicit repository links, and
  shared project metadata. The selected base is ``canonical`` for both multi-member and singleton
  families; singleton selection remains explicit as ``canonical_resolution=singleton_project``.
  Derivative-only families use ``base_repository_missing`` instead of promoting packaging or
  tooling.

- OpenAlex papers are grouped by the generically extracted artifact name and aliases. Original,
  versioned, and generator releases remain visible, while one paper is labelled ``canonical_base``;
  duplicate publication/preprint records are labelled ``alternate_manifestation``.

- Structural ranking handles clear families without another model call and gives no repository role
  an inherent advantage; it uses fork status, base-like naming, explicit inbound links, direct
  self-description, and commit evidence. Only close cases can receive one group-level Ollama
  adjudication over the complete family. Stars and citation counts are never supplied as selection
  criteria.

This makes the result independent of classification order: seeing a packaging repository or version
paper first cannot permanently make it the canonical base. Re-running reconciliation after more
candidates are collected updates the family decision.

Classifier audit plots are separate from thesis-facing catalogue plots:

.. code:: bash

   radar --study ids_repositories compare github
   radar --study ids_datasets compare openalex

Generate both paper-ready architecture views independently from source plotting. The command writes
``pipeline_architecture.{svg,pdf}`` for the detailed five-stage method and
``system_architecture.{svg,pdf}`` for the additional high-level component view:

.. code:: bash

   radar architecture
   # Alternate directory:
   radar architecture --output artifacts/architecture
   # Select one vector format:
   radar architecture --formats pdf
   # Direct module entry point:
   python -m radar.architecture

The detailed pipeline figure labels five stages ``(i)``–``(v)`` for the selected study: provider
discovery, evidence and prefiltering, scientific screening, consensus and reconciliation, and
outputs and evaluation. IDS datasets and conformal treatment effects share the paper collector,
evidence acquisition, prefilter, and operational configuration schema, with study-specific values
and scientific criteria. Repository screening and catalogue reconciliation retain their own
scientific rules. Reference profiles feed evaluation only.

The system figure shows the shared adapter contracts and provider registry for GitHub, OpenAlex,
Semantic Scholar, and local CSV/JSON input; collection and evidence modules; the Ollama client and
configured model families; and study outputs, reporting, and provenance. Semantic Scholar and
local-file discovery currently support review studies. Stored cohorts, assessments, and state remain
scoped to each study and model.

All catalogue, evaluation, comparison, and funnel builders use ``plots.formats``. The default
``formats = ["svg", "pdf"]`` therefore writes a matching PDF beside every SVG figure.

Set ``[plots] show_subcaptions = false`` to hide explanatory subtitles and footnotes across
catalogue, evaluation, comparison, funnel, NLI sensitivity, and presentation figures. Titles, axes,
legends, and record/percentage labels remain. Set it to ``true`` to restore them; omitted settings
default to ``true``. A command can override the configuration with
``radar --no-subcaptions <command>`` or ``radar --subcaptions <command>``.

Evaluation bundles include ``figures/07_recovered_taxonomy_accuracy.{svg,pdf}`` and source
comparisons include ``taxonomy/05_recovered_taxonomy_comparison.{svg,pdf}``. These measure exact
field accuracy among recovered reference positives: expected relevant items matched to accepted
(``accept`` or ``accept_adjacent``) outputs. Each field shows its rated recovered count; recovery
totals remain visible in the axis or legend. Missing field predictions are incorrect; unsupported
fields or zero recovered comparisons are N/A. Read alongside the full-inventory taxonomy and
reference-recovery plots, since approaches may recover different subsets. Companion CSVs retain the
numerator, denominator and recovery totals. These plots reuse saved entity audits and require no new
inference.

Generate the source-to-catalogue reverse-pyramid independently from collection or inference:

.. code:: bash

   radar funnel github
   radar funnel openalex
   # Alternate directory:
   radar funnel github --output artifacts/paper/funnel
   # Direct module entry point:
   python -m radar.funnel github

The funnel reads the active source cache and compatible candidate, classification, and catalogue
CSVs when they exist. GitHub and OpenAlex use the same sequence: **Crawler hits (overlap) →
Candidate selection → Enrichment / prefilter → Classification → Inclusion → Export**. All six layers
remain visible; unavailable counts are grey N/A, with no inferred loss across those stages or from
overlapping crawler hits to selected candidates. If no database snapshot exists, it falls back to
collection capacity from the selected TOML configuration and marks it as configured capacity rather
than an observed count. Every normal ``classify``, ``baseline``, or complete ``run`` refreshes the
default bundle under ``artifacts/studies/<study>/plots/<source>/pipeline_funnel/``;
custom-input/output experiments remain opt-in through the standalone command. Multi-model and
baseline outputs receive separate funnels with a shared selected-candidate denominator. Short labels
name the pipeline steps, with the number and percentage removed printed on the right at each
observed main-path transition. Companion CSVs and the bundle README preserve detailed explanations,
provenance, and denominators; the figures omit subcaptions.

The top funnel layer shows the total crawler hits, which can overlap across queries and discovery
routes. Detailed query matches, unique search matches, source-route counts and auxiliary surveys
remain in ``data/discovery_context.csv``; the figures have no separate discovery-input panel. An
unavailable unique discovery pool is recorded in that table rather than inferred from crawler hits.
Discovery caps derive from the retained-candidate ``collection_limit`` and the model-input
``classification_limit``, using ``query_limit_factor`` (default 3) times the larger numeric limit.
If neither limit is numeric, retrieval has no application cap. Auxiliary surveys are discovery
sources, not classification removals. This alignment changes reporting only; collection, ranking,
prefiltering and classification behavior are preserved.

.. _pipeline-multiple-models-and-ensembles:

Multiple models and ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the required, non-empty ``ollama.models`` array to run each model-dependent stage. Collection
and prefiltering are shared. One model and several models use the same ``models/<model-slug>/``
output layout.

.. code:: toml

   # config/shared/models.toml
   [ollama]
   models = ["gemma4:31b", "qwen3.8:27b"]
   context_length = 131072
   keep_alive = "30m"

   # config/studies/<study>/pipeline.toml
   [classification.ensemble]
   enabled = true
   # Empty means every ollama.models entry; otherwise list only participating models.
   models = ["gemma4:31b", "qwen3.8:27b"]
   threshold = 1.0

``ollama.context_length`` is sent as ``options.num_ctx`` on every classification and reconciliation
request, overriding Ollama’s server-selected context for this pipeline. The checked-in profile
requests a 128 Ki-token context window (``131072``). Set it to ``0`` to omit the request-level
override and use the Ollama server default. ``ollama.keep_alive`` is sent with every request so
Ollama keeps that model loaded between consecutive items when memory permits. Context capacity and
keep-alive are runtime controls, not semantic decision settings, so changing them does not
invalidate compatible cached classifications. Use ``--resume`` to explicitly continue compatible
current-format work. New processes log Ollama-reported ``load``, ``prompt_eval``, and ``eval``
durations for every request; a large ``load`` duration is direct evidence of a model reload.

Each source can also repeat the same classification consecutively with
``classification.consecutive_runs`` and vote using ``classification.consensus_threshold`` in its
``pipeline.toml``. Consensus covers the include/exclude/uncertain disposition and normalized
classification fields. Single-value fields use threshold voting; GitHub scopes, observation objects,
capabilities, detection methods, and system types use per-label voting. The primary system type is
voted as a single value. System types are preserved from model outputs and validated against the
configured vocabulary; they are not inferred from the other fields. The assembled tuple is checked
against deterministic cross-field rules. Invalid mixtures fall back to an audited coherent
supporting run, or remain uncertain when no valid fallback exists. Set ``consecutive_runs = 1`` for
one assessment per model. See :doc:`classification consensus <classification_consensus>` for the
complete rules, independence limitations, and audit columns.

The ensemble first maps ``accept`` and OpenAlex ``accept_adjacent`` to include, while ``exclude``
and ``uncertain`` remain separate dispositions. A unique disposition must reach the threshold among
all selected ensemble members. Missing/error results remain in the denominator, and tied or
below-threshold votes become ``uncertain``. The ensemble CSV preserves the representative
classification plus member names, each model decision, exact decision counts, all disposition
scores, agreement, and vote status. Its reconciliation is deterministic and does not ask another
model to override the vote.

With two or more configured models, outputs are isolated under
``artifacts/studies/<study>/interim/<source>/models/<model-slug>/``,
``artifacts/studies/<study>/processed/<source>/models/<model-slug>/``, and
``artifacts/studies/<study>/plots/<source>/models/<model-slug>/``. Ensemble equivalents use an
``ensemble/`` directory. Unrated cross-model decision diagnostics from catalogue plotting are
written under ``artifacts/studies/<study>/plots/<source>/comparison/models/``; rated correctness and
agreement comparisons are kept separately under
``artifacts/studies/<study>/evaluation/<source>/comparison/``. The shared cache remains safe because
decision fingerprints already include the model name. ``radar compare <source>`` rebuilds the rated
cross-classifier bundle. In multi-model or ensemble mode, model-dependent ``--output`` paths and
stage-specific ``--input`` paths for reconcile, export, or plot are rejected because one file cannot
represent every variant; a shared candidate ``--input`` remains supported for classify and run.

Use ``--config path/to/config.toml``, ``--input``, and ``--output`` for alternate runs. The default
generic discovery queries and weighted concept groups live in each study’s ``search/`` directory.
OpenAlex uses parenthesized ``AND``/``OR`` synonym blocks across title, abstract, and indexed full
text. GitHub uses smaller unparenthesized ``OR`` groups because repository search does not reliably
share OpenAlex’s grouping behavior. Production term profiles must contain domain concepts only,
never named products, repositories, datasets, or benchmarks.

The compact ``config/studies/ids_repositories/study.toml`` references shared runtime and study-owned
profiles. Classifier prompts and taxonomy explanations live separately in
``config/studies/ids_repositories/prompts/classification.toml`` and
``config/studies/ids_datasets/prompts/classification.toml``; ``screening.toml`` selects those
profiles, so a custom study can point at alternatives. Python contains no fallback prompt text. Both
``template`` and ``reconciliation_template``, every required taxonomy table, and every normalized
enum description are mandatory; missing content produces a configuration error before an LLM
request. GitHub classification supports ``${roles}``, ``${scopes}``, ``${observation_objects}``,
``${capabilities}``, ``${methods}``, ``${system_types}``, and ``${evidence}``; OpenAlex
classification supports ``${roles}``, ``${tiers}``, ``${artifact_types}``, ``${domains}``,
``${data_forms}``, ``${environments}``, ``${labeling}``, and ``${evidence}``. Reconciliation
templates use ``${evidence}`` for the complete candidate family. ``${evidence}`` is mandatory, and
unknown placeholders also fail before an LLM request. GitHub system-type enums are generated from
the selected taxonomy profile, so adding a configured type does not require changing Python.
Relevant prompt configuration is included in the decision fingerprint, so prompt edits invalidate
only incompatible cached decisions.

The LLM never supplies a confidence score. GitHub classification uses the roles ``native_detector``,
``classifier_engine``, ``detection_framework``, ``detection_platform``, ``traffic_analyzer``,
``antivirus``, ``research_project``, ``demo``, ``rules_content``, and ``support_tooling``, plus
audit-only ``out_of_scope`` and ``uncertain``. Deterministic gates require native detectors to
implement their own runtime engine; classifier engines to implement detection logic with a
reproducible execution path; detection frameworks to additionally expose a cohesive agent/manager or
inseparable multi-subsystem architecture; detection platforms to reproducibly integrate multiple
independent detectors; traffic analyzers to provide reproducible security traffic inspection or
forensic analysis (offline captures are permitted, and detection logic is optional); antivirus/file
scanners to provide reproducible file analysis and a detection capability (offline scans and
external engines are permitted); and research projects to combine a research basis with a
reproducible train, inference, evaluation, or detection path. Failed gates are exported in
``classification_validation_flags``. For GitHub reporting, ``native_detector``,
``classifier_engine``, and ``detection_framework`` are the operational-system segment; all other
accepted roles remain reported separately. Extracted upstream repositories must occur in verified
evidence. ``canonical_confidence`` is a coarse code-derived indicator for multi-member structural
ranking only; singleton and missing-base families leave confidence and margin blank.

Antivirus/file scanning has its own ``antivirus`` role, ``AV`` system type, and ``file_analysis``
observation object. This covers scanning uploads, binaries, archives, and cloud-storage files for
malware, including signature/hash/YARA scanners and ClamAV services. File scanning or quarantine
alone does not establish HIDS/HIPS/APIDS. File-integrity monitoring and behavioral intrusion
detection remain distinct. Rule databases and illustrative malware demos retain ``rules_content``
and ``demo``, while their target system may be ``AV``. Roles and types are selected independently.
All-role overview and role/type plots display antivirus separately; operational IDS timelines and
method/type plots retain their operational-role segment. The GHML23 reference crosswalk now labels
BinaryAlert, Hawk, and Loki as antivirus, and corrects signature-base and CyberDoc to AV
content/demo targets. Cached classifications need a classification rerun to acquire these labels;
rescoring alone only compares existing predictions with the updated reference.

GitHub system types are defined independently in ``config/studies/ids_repositories/taxonomy.toml``.
Each ``[types.CODE]`` table provides the code’s readable name and semantic description. The
descriptions are shown to the LLM, which directly returns ``primary_system_type`` and the
multi-label ``system_types``; Python validates those values and resolves ``system_type_names`` but
does not derive them from scope, capability, method, or role. Add a new table to introduce another
type without changing Python. ``PIDS`` explicitly means protocol-based IDS, while provenance-based
IDS uses ``ProvIDS``. The ``NSM`` type represents principal network security monitoring and traffic
analysis, including systems such as Zeek whose limited detection policies do not make their
principal role an operational detector. The ``SM`` type (Security monitoring / telemetry analysis
system) covers principal non-network or mixed-source security log aggregation, endpoint activity
monitoring, and telemetry analysis without requiring intrusion decisions or alerts. ``SM`` is a
catalogue code for this category. Pure forwarding, configuration, and generic observability alone do
not establish it. System types are inserted into the final prompt through ``${system_types}`` from
the selected taxonomy TOML; the same vocabulary controls response enums, validation, and readable
output labels. The separate file keeps type definitions reusable and independently configurable.
``flow_analysis`` requires aggregated flow/session records as analysis input; packet capture, stream
reconstruction, and flow export alone do not qualify. The type vocabulary is included in
classification fingerprints.

For NIDS/NIPS, the catalogue requires a material packet/interface intrusion-inspection path; NIPS
additionally enforces network prevention. HIDS monitors endpoint activity/state (including OS audit
logs), which can be forwarded to a central manager. HIPS additionally enforces endpoint protection,
such as process termination, execution restrictions, file quarantine, or account disabling. Local
service-log detection and firewall-only source-IP bans, such as Fail2ban, use general IDS/IPS.
Deployment location and NDR/EDR branding alone do not determine these types. See the
:ref:`boundary review and repository evidence <ghml23_repository_audit-packet-inspection-endpoint-activity-and-enforcement-boundaries-2026-10-02>`.

.. _pipeline-external-reference-set-evaluation:

External reference-set evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``[evaluation].reference_sets`` array in ``config/studies/ids_repositories/study.toml`` selects
``config/studies/ids_repositories/evaluation/references.toml`` for GHML23 and
``config/studies/ids_datasets/evaluation/references.toml`` for SHIELD. The GHML23 TOML profile keeps
the untouched upstream semicolon-delimited CSV as an immutable provenance snapshot and explicitly
crosswalks all 120 repositories into this project’s decisions and controlled taxonomy. The original
``secumon``, ``pentest``, and ``imageocr`` labels remain recorded as benchmark categories; they are
not used as a coarse runtime mapping. ``radar <source> evaluate`` compares saved classifications
with the configured source reference set. Each classifier keeps machine-readable audit tables at
``artifacts/studies/<study>/evaluation/<source>/classifiers/<family>/...``, with its diagnostic
figures and companion data isolated below ``figures/``. Cross-classifier reporting is in the sibling
``comparison/`` directory, and a source-level README points to the correct starting figure. Set
``[evaluation].enabled = true`` to execute this stage automatically after classification in a full
run. Reference profiles and their imported CSVs are evaluation-only: they are absent from LLM
decision fingerprints, so editing or replacing a reference set cannot invalidate cached
classifications. Use the explicit ``evaluate`` command when only rescoring/replotting is wanted. See
:doc:`rated classification evaluation <evaluation>` for profile details, exact denominators,
missing-result behavior, output files, and tests.

``radar.common.regression.evaluate_regression(source, expected, rated_entities, classification_fields)``
remains available for small plain-dictionary unit fixtures without persisted run artifacts.

.. _pipeline-ollama-nli-baseline:

Ollama NLI baseline
~~~~~~~~~~~~~~~~~~~

The optional ``llm_nli`` baseline asks an Ollama model one premise/hypothesis pair at a time and
obtains entailment, neutral and contradiction scores from output token log probabilities. It uses
the same hypotheses, tokenizer-based evidence chunking, chunk selection and aggregation as DeBERTa
NLI. Configure models in ``[baselines.llm_nli]``; the default is ``qwen3.8:27b``. Install the
``nli`` optional dependencies for the shared tokenizer. This baseline loads no DeBERTa classifier.

Check server capabilities, then run a small pilot:

.. code:: bash

   radar --study ids_repositories baseline --methods llm_nli --check-only
   radar --study ids_repositories baseline --methods llm_nli --fields role --limit 5
   radar --study ids_datasets baseline --methods llm_nli --fields paper_role --limit 5

The check requires all three class-token alternatives at the same output position. Missing log
probabilities cause an explicit error. Ollama provides output-token log probabilities rather than
full vocabulary logits; see its `chat API <https://docs.ollama.com/api/chat>`__. Scores are
normalized over the three class codes and are not calibrated probabilities of correctness.

After inspecting pilot scores and runtime, run the complete taxonomy:

.. code:: bash

   radar --study ids_repositories baseline --methods llm_nli
   radar --study ids_datasets baseline --methods llm_nli

This can require hundreds of Ollama requests per entity. Pair caching makes interrupted runs
resumable. Full results live under
``artifacts/studies/<study>/interim/<source>/baselines/llm_nli/<model-slug>/``; evaluations and
comparison plots identify them separately from structured LLM classification. Limited or
field-restricted pilots use a nested ``pilots/`` directory and stay out of full comparison plots.
Automatic execution is selected by the study baseline settings.

Use ``radar --study <study> baseline --methods llm_nli --evaluate-only`` to regenerate evaluations
and plots offline. Threshold-only changes rescore saved probabilities; changes to prompts,
hypotheses or chunking require inference again. See
:ref:`evaluation details <evaluation-ollama-nli>` for provenance and denominators.

.. _pipeline-data-products:

Data products
-------------

GitHub repositories also receive a directly classified, single-label ``centralization``:
``centralized``, ``hierarchical``, ``fully_distributed``, or ``unknown``. The GitHub prompt
additionally supports ``${centralization}``, with descriptions in ``[prompt_tables.centralization]``
in ``config/studies/ids_repositories/taxonomy.toml``. Keyword definitions and NLI hypotheses live in
the corresponding ``config/studies/ids_repositories/baselines/`` profiles. Centralization is
retained in caches/CSVs, voted across repeated runs and models, and included in taxonomy
evaluation/comparison plots. General catalogue plots include ``13_ids_centralization`` and
``14_centralization_role_matrix`` with CSV audit tables and SVG/PDF output. The GHML23 reference
profile records supported topology ratings with sources; unknown reference topology is not scored.
This dimension is GitHub-only and does not change inclusion criteria or OpenAlex classification.
Existing GitHub classifications need fresh model judgments to populate it; reevaluation/replotting
alone cannot supply missing topology evidence.

The checked-in study manifests place outputs under ``artifacts/studies/<study>/``. Within that root:

- ``raw/<source>/cache.sqlite3`` stores API evidence, selections and decision caches.
- ``interim/<source>/`` stores candidate audits and auxiliary survey sources.
- ``interim/<source>/models/<model>/`` stores classifications and reconciled audits.
- ``interim/<source>/baselines/<method>/`` stores baseline decisions and score audits.
- ``processed/<source>/models/<model>/`` stores repository or dataset/generator catalogues.
- ``plots/<source>/`` stores catalogue figures, model comparisons and pipeline funnels.
- ``evaluation/<source>/classifiers/`` stores rated metrics, instance tables and figures.
- ``evaluation/<source>/comparison/`` stores cross-method metrics and agreement views.
- ``runs/`` stores study identity, profile fingerprints, coverage and stage counts.

Ensemble outputs use a separate ``ensemble/`` variant alongside model outputs. GitHub catalogues
retain canonical project relationships and directly classified system types. OpenAlex catalogues
retain all accepted dataset and generator papers, including standalone generator frameworks; exports
preserve inclusion rather than applying another resource-type filter. Dataset names and publication
titles remain separate fields.

The review workflow stores candidate JSON, evidence-backed classifications and assessment
checkpoints. Its generic report includes evaluation and a reporting database beneath each model
report directory. See :ref:`study configuration <configuration-outputs-and-clean-state>` for
details.

``radar architecture`` generates ``pipeline_architecture.{svg,pdf}`` and
``system_architecture.{svg,pdf}`` independently of study collection.

Popularity fields never determine eligibility or relevance. The default GitHub star and OpenAlex
citation floors are both zero. Within an already verified OpenAlex artifact family, raw citation
counts provide only a configurable, log-scaled, capped secondary signal for canonical-paper
selection.

The GitHub plot bundle starts with one tall, scrollable feature matrix of every accepted repository
and a lossless ``data/00_repository_overview.csv``. It groups repositories by normalized system
family and shows role dots, system labels, methods, activity, stars, forks, canonical/variant
status, and latest commit time. Collections record the exact latest commit timestamp when available;
missing commit metadata stays unavailable in plots.

GitHub plot ``01_repository_timeline`` is a density heatmap: repository creation year on x and
primary IDS type on y. Each type has a stable colour across models, and darker cells represent more
canonical operational systems using the same count-to-intensity scale. Cells print exact counts;
blank cells are zero, and missing creation dates appear under ``Unknown``. Every repository
contributes once through its primary type. The aggregate CSV contains every type/year bin, while
``data/01_repository_timeline_records.csv`` preserves repository names, stars, URLs, and bin
membership. This replaces the crowded scatter plot with repository names under individual points.

The OpenAlex publication timeline uses a continuous horizontal year axis on a 36-inch-wide landscape
canvas and a linear citation axis from zero through 250. Above 250, equal citation differences use
2.5% of the lower range’s vertical spacing (40-fold compression), so the upper range remains
uniformly linear rather than logarithmic. The mapping is continuous at 250, the change is marked,
and tick labels show actual citation counts. Resource labels form year columns in citation order,
with the most-cited at the top; there are no scatter dots. Measured label boxes are placed as close
to their true citation heights as possible and shifted locally when necessary to avoid overlaps.
Short styled leaders end at exact citation-height ticks beside each year column. Domain colors
identify application areas; solid borders/lines indicate datasets, dashed generators, dotted
combined releases, and dash-dot unknown types. The companion CSV records true citation coordinates,
displayed label coordinates, displacement flags/offsets, line styles, and measured box boundaries.
SVG and PDF preserve the full wide landscape canvas. The classifier extracts
``artifact_display_name`` separately from the publication title: it prefers an explicit short name
or acronym, never invents one, and otherwise supplies a concise evidence-grounded descriptor. Paper
plotting requires the explicitly exported ``resource_name``.

The remaining figures deliberately avoid overloaded scatterplots. Citation context is an ordered dot
plot whose logarithmically spaced axis is labelled with actual citation counts. Discovery provenance
is an exact row matrix showing direct search, survey snowballing, and the number of distinct surveys
citing each accepted resource; auxiliary survey papers remain in their separate survey-source
figure. Domain/data-form and capture/label relationships are annotated count heatmaps; growth uses
cumulative counts with integer axes; and classifier outcomes show counts plus within-prefilter-band
percentages. Survey sources use three aligned panels for retrieval score, citations, and
referenced-work count. Labels occupy dedicated rows rather than being placed on top of data points.
Exact citation, provenance, score, and identifier tables are exported beside every figure. Large
SVG/PDF canvases are intended for scrolling and zooming.

OpenAlex discovery scans native results independently from each Boolean synonym-block query,
deduplicates the combined pool, and retains the best ``collection.collection_limit`` candidates when
that limit is numeric. Per-query caps are ``query_limit_factor`` (default 3) times the larger
numeric collection/classification limit. Optional ``survey_query_limit_factor`` and
``reference_limit_factor`` override this factor for survey searches and the total snowball reference
budget. If both pipeline limits are omitted or ``"all"``, query and reference retrieval have no
application cap; ``max_surveys``, snowball seeds and depth still define the traversal protocol.
Pagination follows the derived caps without a separate page limit. Survey detection and
release-exclusion phrases live in ``[survey_identification]`` in
``config/studies/ids_datasets/search/openalex.toml``. Title evidence has a configurable additional
weight (``collection.retrieval_title_weight``) so an abstract that merely mentions many relevant
concepts cannot completely hide an obviously weak title. Blank or structurally uninformative titles
and results below the configured generic retrieval floor are not retained. Retrieval, title,
combined rank, and survey-source scores are exported in the audit CSVs. Citations are logged and
exported but never enter retrieval ranking. The selected work IDs are recorded as the active
collection: later prefiltering reads this pool rather than mixing in stale works from older
collection strategies, while those historical raw records remain in SQLite for audit and reuse.

Survey discovery runs before direct discovery so it cannot be starved by the collection limit.
Surveys themselves are ranked using generic positive evidence, with each selected survey’s score,
citations, and reference-list size logged. They are auxiliary records and are excluded from both the
collection limit and LLM shortlist. For a 250-paper collection with the default
``survey_reference_fraction = 0.5``, the target allocation is 125 direct candidates plus 125 new
snowballed candidates, in addition to the separately configured survey sources. Unused snowball
capacity is backfilled with the next strongest direct papers. Reference IDs mentioned by multiple
selected surveys are resolved first. References shared by at least ``survey_consensus_threshold``
selected surveys (two by default) may bypass the generic text-score floor, and survey-source count
takes priority when filling the snowball budget. This preserves opaque dataset papers whose OpenAlex
metadata does not expose their release contribution. Each retained paper records both
``survey_reference`` and every originating ``survey_reference:<OpenAlex survey ID>`` relation. Run
manifests and coverage records expose scanned/selected result counts and bounded query/reference
lists.

.. _pipeline-csv-conventions:

CSV conventions
---------------

- Schema version is ``1.0``.
- Lists are semicolon-separated.
- Booleans are lowercase ``true`` or ``false``.
- Timestamps use UTC ISO 8601.
- ``unknown`` means the evidence did not support a taxonomy value; an empty cell means the source
  did not provide metadata.

- Processed OpenAlex metadata always comes from the accepted work itself, not from an earlier
  keyword catalogue.

- When an OpenAlex work has no abstract, classification resolves it by DOI or exact normalized title
  through Semantic Scholar. Successful abstracts and optional open-access PDF text are cached,
  written back to the candidate audit, and included in the evidence hash; missing full text never
  blocks classification.

- OpenAlex release roles and artifact types must agree: dataset releases use ``dataset`` or
  ``dataset_and_generator``; generator releases use ``generator_framework`` or
  ``dataset_and_generator``.

- Accepted OpenAlex rows always expose ``resource_name``; the name comes from the current explicit
  ``artifact_display_name``, with no backfill from aliases.
