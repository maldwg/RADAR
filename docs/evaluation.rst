.. _evaluation-external-reference-set-evaluation:

External reference-set evaluation
=================================

The evaluation stage compares saved pipeline classifications with versioned external reference sets.
It is offline: it does not call GitHub, OpenAlex, Ollama, or any classifier, and it does not change
classification decisions.

.. _evaluation-configured-reference-sets:

Configured reference sets
-------------------------

``config/studies/ids_repositories/study.toml`` declares reference sets as an array of tables:

.. code:: toml

   [evaluation]
   enabled = true
   confidence_level = 0.95

   [[evaluation.reference_sets]]
   name = "ghml23"
   source = "github"
   profile = "evaluation/references.toml"

The dataset study declares its own ``shield`` reference in its manifest. Paths resolve relative to
the declaring study manifest. A custom study can replace or add entries without changing Python. A
single configured set retains the normal classifier evaluation directory. If several sets target one
source, each result is isolated below a reference-set-named subdirectory.

The checked-in inputs are:

- ``config/studies/ids_repositories/evaluation/references.toml`` and its verbatim
  ``github_ghml23.csv`` snapshot.

- ``config/studies/ids_datasets/evaluation/references.toml``, the survey-derived SHIELD artifact
  profile.

Source-record profiles use exactly one matching selector per ``[[entities]]`` table:

- GitHub: ``repository = "owner/name"``. Matching uses the ``full_name`` saved in classifications
  and candidates.

- OpenAlex: ``openalex_url = "https://openalex.org/W..."`` is recommended.
  ``title = "Exact paper title"`` is also supported when no OpenAlex link is available.

- Explicit native ``id`` selectors can be used when the reference supplies a stable provider ID.

The checked-in OpenAlex profile instead uses ``evaluation_unit = "artifact"``: every entity names a
dataset with ``artifact``, optional ``aliases``, and survey-reference provenance. Evaluation
searches the saved OpenAlex results for an accepted release representing that dataset.

Repository names and titles are matched exactly after trimming repeated whitespace and ignoring
letter case. A trailing slash on an OpenAlex URL is ignored. If a selector matches multiple
rows–most plausibly a duplicated paper title–evaluation stops with an ambiguity error instead of
guessing. The resolved stable source ID remains visible in ``instances.csv``.

.. code:: toml

   schema_version = "1.0"
   source = "github"
   classification_fields = ["role"]

   [entity_source]
   format = "csv"
   path = "github_ghml23.csv"
   delimiter = ";"
   selector = "repository"
   selector_column = "Repository"

   [entity_source.mappings.decision]
   column = "Label"
   values = { secumon = "accept", pentest = "exclude", imageocr = "exclude" }

An ``entity_source`` imports a local CSV snapshot and maps its columns and category values
declaratively. Inline ``[[entities]]`` profiles remain supported for reference sets whose published
source has already been transcribed into TOML, such as SHIELD. Mixing both forms in one profile is
rejected.

.. code:: toml

   schema_version = "1.0"
   source = "openalex"
   evaluation_unit = "artifact"
   classification_fields = ["artifact_type", "domain", "data_form", "capture_environment", "labeling"]

   [[entities]]
   artifact = "UNSW-NB15"
   aliases = ["UNSW NB15"]
   name = "UNSW-NB15"
   decision = "accept"
   artifact_type = "dataset"
   domain = "general_network"
   survey_reference_openalex_url = "https://openalex.org/W2296509296"

The supported decisions are ``accept``, OpenAlex ``accept_adjacent``, ``exclude``, and
``uncertain``. Binary metrics use only human-rated include and exclude entities; a human-rated
``uncertain`` entry remains in exact decision and field reports but is outside the binary
confusion-matrix denominator.

Only field values present on a rated entity are compared. This permits decision-only ratings and
incrementally richer taxonomy ratings in the same file. Scalar comparisons trim whitespace and
ignore case. Array and semicolon-separated multi-label values are compared as unordered sets. Empty
and ``unknown`` remain distinct.

Classification files declare their output contract through ``classifier_supported_fields``. ``*``
means the classifier emits the complete source taxonomy. Fields outside a custom classifier’s
contract remain visible in ``field_metrics.csv`` with ``supported=false``, zero comparisons, and N/A
scores; they are not silently counted as incorrect taxonomy predictions. The checked-in Keyword,
NLI, LLM, and Ensemble approaches all emit the rated taxonomy fields.

Profiles are validated before evaluation. A missing profile, wrong source/schema, missing or
multiple selectors, duplicate selectors, ambiguous result matches, malformed field list, or
unsupported decision fails the command instead of emitting misleading scores.

.. _evaluation-benchmark-composition:

Benchmark composition
~~~~~~~~~~~~~~~~~~~~~

GitHub starts from GHML23’s published ``ground_truth_with_source.csv``: 100 security-monitoring
(``secumon``) repositories, 10 penetration-testing controls, and 10 image/OCR controls. The adjacent
CSV remains an immutable provenance snapshot. The executable TOML profile explicitly reviews and
crosswalks every repository into this project’s inclusion decision and controlled role,
observation-scope, observation-object, capability, detection-method, and IDS-type vocabularies. The
original GHML23 category and selection source remain attached to every reference entity for audit,
but do not mechanically determine the richer labels. Executable decision counts come from the
selected TOML reference profile; the :doc:`repository evidence index <ghml23_repository_audit>`
records the checked GitHub source for all 120 rows. The 2026-10-02 follow-up adds the
``traffic_analyzer`` role and ``NSM`` type and rechecks
all 13 prior ``flow_analysis`` labels against documented aggregate-record input paths (seven
retained, six removed), with Arkime gaining the label for its separate session-record Viewer. A
further 2026-10-02 review adds ``SM`` for non-network and mixed-source security telemetry
monitoring, assigning it to HELK, xnumon, osctrl, and ThreatHunting while preserving their
support-tool roles. The packet/endpoint/enforcement follow-up revises Fail2ban, sshwatch, Flare,
OpenSnitch, and CrowdSec; it records direct endpoint enforcement evidence for retained HIPS labels
and corrects the OpenSnitch/BZAR analysis-input labels. The antivirus follow-up adds ``antivirus``,
``AV``, and ``file_analysis``, relabels BinaryAlert, Hawk, and Loki as scanners, and retains
signature-base/CyberDoc as content/demo with AV targets. CyberDoc carries forward its prior reviewed
evidence because the source was unavailable in this follow-up. File quarantine alone does not
establish HIPS. The crosswalk is version 10; screening counts, centralization judgments, and flow
labels are unchanged.

OpenAlex uses a 105-row artifact fixture extracted from Table 2 of Ashraf and Masoodi (2026),
cross-checked against Figure 1 and SHIELD dataset descriptions. The published survey fields are
preserved verbatim; project labels may deliberately correct them (for example, LBNL is an enterprise
network trace, not an electrical-grid dataset). The selected rows contain 96 core accepts, five
adjacent accepts, and four excluded audit controls. FPR, specificity, accuracy, precision, and F1
are calculable for this selected fixture, but four non-random negative controls do **not** support a
corpus-wide false-positive or prevalence claim. ``artifact_type = "dataset"`` for every row because
the source inventory is of released datasets; a testbed used to produce data is not automatically a
released generator framework. Taxonomy-field accuracy and dataset-identified coverage use only
positive artifacts; excluded audit controls contribute to decision metrics, not missing-metadata
penalties.

Regression tests lock the exact GHML23 row/category/decision counts and representative crosswalk
entries, plus the source counts, exclusions, aliases, requested domain taxonomy, and provenance
fields of the SHIELD dataset profile.

.. _evaluation-running-evaluation:

Running evaluation
------------------

Study CLI evaluations compare ``relevance_decision`` independently of content quality.
Quality outcomes and quality-qualified exports live in separate bundles; see
:doc:`quality_assessment`. Outcome rates are not quality accuracy without an independently
annotated quality reference.

Evaluate existing default classification files:

.. code:: bash

   radar --study ids_repositories evaluate
   radar --study ids_datasets evaluate

The default catalogue baseline command creates keyword, DeBERTa NLI, plain LLM and configured
Ollama NLI classification files and evaluates them against the same reference profiles.
``--methods keyword,nli`` selects the two methods that do not call Ollama.
The :doc:`plain LLM benchmark <plain_llm>` also supports systematic reviews:

.. code:: bash

   radar --study ids_repositories baseline
   radar --study ids_datasets baseline
   radar --study ids_repositories baseline --methods nli
   radar --study ids_datasets baseline --methods keyword,nli

Use other classification and candidate files or another artifact directory:

.. code:: bash

   radar --study ids_repositories evaluate \
     --input path/to/classifications.csv \
     --candidates path/to/candidates.csv \
     --output artifacts/evaluation/experiment-a

The candidate CSV is optional. When supplied, it lets the evaluator distinguish a rated entity that
was retrieved but not classified (``not_classified``) from one that was never retrieved
(``not_retrieved``). Without it, missing classifications are ``unavailable`` and retrieval coverage
is ``null``, because retrieval cannot honestly be inferred.

To make evaluation an automatic stage of each complete source run, set:

.. code:: toml

   [evaluation]
   enabled = true

Automatic LLM evaluation runs immediately after classification and consensus, before reconciliation.
When ``[baselines].enabled = true``, a complete run subsequently executes the configured baseline
methods, evaluates them, and rebuilds the all-classifier plots. Explicit ``evaluate`` and
``baseline`` commands run regardless of their automatic-stage switches.

Every rated result lives under ``artifacts/studies/<study>/evaluation/<source>/``. The source-level
README is the directory index. Classifier details use ``classifiers/keyword/``,
``classifiers/nli/<model-slug>/``, ``classifiers/llm/<model-slug>/``, and ``classifiers/ensemble/``.
Ratings and candidates are shared, which makes comparisons use the same reference set.

Cross-approach plots are in the sibling ``comparison/`` directory. ``00_metric_overview`` is the
single starting matrix: its rows contain DR, FNR, FPR, FDR, precision, accuracy, decision-F1,
coverage, abstention, and aggregate taxonomy scores; its columns contain every available Keyword,
NLI, LLM, and Ensemble result. Arrows state whether higher or lower is preferable, grey cells mean
N/A, and exact values are in ``data/00_metric_overview.csv``. Enlarged operational-metric plots are
grouped under ``metrics/``, field/category comparisons under ``taxonomy/``, and decision agreement
under ``agreement/``. Agreement is descriptive and must not be read as correctness.
Single-classifier operational/strict/selective views and audit diagnostics remain under that
classifier’s ``figures/`` directory rather than being mixed with the comparison figures. The
OpenAlex comparison uses the 105-row artifact benchmark, including selected negative controls.
Performance figures use color and axes for values; companion CSVs retain exact values, per-label
TP/FP/FN, classifier IDs, rating paths, and Wilson intervals. Metrics that a custom positive-only
artifact fixture cannot estimate are shown as N/A, not as zero.

The descriptive ``agreement/01_model_decision_counts`` chart uses horizontal stacked bars over the
full union of classified papers or repositories, rather than the rated benchmark alone. Each bar
uses that common denominator; numbers inside sufficiently wide segments are record counts. Relevant
combines ``accept`` and ``accept_adjacent``, Other means ``exclude``, and Uncertain stays separate.
Errors, missing results, and unexpected outcomes receive separate segments when present. Exact raw
decisions remain in ``agreement/data/01_model_decision_counts.csv``; grouped counts and percentages
are in ``01_model_decision_shares.csv``. This display grouping does not change the exact-decision
agreement calculation.

NLI evaluation also writes ``classifiers/nli/<model-slug>/sensitivity/``. This is not a second NLI
result. It replays the aggregate probabilities already stored in ``nli_scores_json`` over the finite
grid configured in ``[baselines.nli.sensitivity]``, so it performs no inference and needs no network
access. An entailment score says how strongly the text supports a label hypothesis; it is not a
probability that the final answer is correct. The six-panel heatmap has minimum label evidence on
its columns and the required lead over the runner-up label on its rows. Moving right or upward
demands stronger, less ambiguous evidence and usually increases abstention. The black outline marks
the cutoffs actually used for the reported NLI result. Blue panels prefer darker cells; red
error-rate panels prefer lighter cells. The sweep describes robustness and trade-offs; it must not
silently tune the operating point on the same fixture used for final reporting.

Use ``radar <source> baseline --evaluate-only`` to rebuild baseline evaluations, NLI sensitivity
analysis, and cross-approach plots from existing baseline classification CSVs. This mode does not
open the source cache or initialize the keyword or NLI classifier.

.. _evaluation-ollama-nli:

Ollama NLI
~~~~~~~~~~

``radar <source> baseline --methods llm_nli`` runs each model configured in
``[baselines.llm_nli].models`` as a separate generative NLI baseline. A versioned prompt in
``config/studies/ids_repositories/baselines/llm_nli.toml`` asks for A (entailment), B (neutral), or
C (contradiction) for one hypothesis and one source-labelled evidence chunk. The model sees the
three possible outcomes in that prompt; taxonomy hypotheses are evaluated independently. Thinking is
disabled, generation is deterministic, and no JSON grammar restricts the answer probabilities.

The backend uses Ollama’s native ``/api/chat`` ``logprobs`` and ``top_logprobs`` fields. It requires
all three class alternatives at the same generated-token position, normalizes their probability mass
into an ``NLIScore``, and records the original mass. Missing alternatives, invalid output and
unavailable scoring support are errors, never neutral scores or silently imputed zeroes.
``--check-only`` checks server version, installed model digest and three synthetic pairs without
reading the corpus or loading a tokenizer. Passing it establishes score availability, not classifier
quality or calibration. Check completeness and label bias again on development examples before
scaling.

DeBERTa NLI and Ollama NLI share the tokenizer/chunk policy configured in ``[baselines.nli]``, the
source hypothesis profiles, representative chunk selection, strongest-chunk aggregation and source
normalization rules. Ollama NLI has its own decision threshold and margin in
``[baselines.llm_nli]``. It does not retrieve extra evidence or use structured-LLM quote gates or
repeated-run consensus. The existing structured LLM pipeline can use more evidence and adaptive
enrichment: the all-approach plots compare operational methods, and do not by themselves establish a
controlled, equal-evidence experiment.

For development pilots use ``--fields role --limit 5`` (GitHub) or ``--fields paper_role --limit 5``
(OpenAlex). Any selected field list must include the source’s role field. ``--limit`` selects the
first N rows of the candidate audit. Pilots have isolated paths and never enter automatic full
comparisons or funnels. Pilot evaluations still use the configured reference denominator; their
recovery rate is not a full-corpus performance estimate. Unselected taxonomy fields are marked
unsupported/N/A.

Full classifications are stored in
``artifacts/studies/<study>/interim/<source>/baselines/llm_nli/<model-slug>/classifications.csv``,
and evaluation files in
``artifacts/studies/<study>/evaluation/<source>/classifiers/llm_nli/<model-slug>/``. Pilots add
``pilots/<fields>-<limit>/`` below those paths. Full comparisons display ``NLI - <model-slug>``
alongside DeBERTa NLI and the structured model. Both methods receive the usual recovery, taxonomy,
decisions, coverage, unresolved-rate and offline threshold-sensitivity figures.

``llm_nli_metadata_json`` records model digest, server version, class codes, prompt fingerprint and
tokenizer identity. ``llm_nli_pair_scores_json`` records each scored hypothesis, premise hash, raw
token alternatives, three normalized scores, original class mass, token counts and inference timing.
``nli_evidence_chunks_json`` preserves the chunk texts; the existing supporting-chunk and
aggregate-score columns identify how they affected decisions. Pair cache fingerprints include the
model/server/scoring provenance. Errors are retried on resume, while valid prefix scores can be
reused.

``--methods llm_nli --evaluate-only`` initializes no Ollama backend and requires no network access.
If only decision thresholds changed it replays the stored aggregate scores and updates decisions
before evaluation. It refuses mismatched models, prompt settings, hypotheses or chunking. To replot
a pilot, repeat its ``--fields`` and ``--limit`` options with ``--evaluate-only``.

The presentation-style overviews are also generated by normal evaluation plotting for both sources:

- ``comparison/01_reference_recovery.svg``/``.pdf``: horizontal recovery percentages across
  classifiers, calculated as TP / (TP + FN). The denominator is the positive reference inventory,
  including missing and uncertain recoveries as misses; selected negative controls are excluded from
  this denominator. Companion CSVs preserve recovered/missed counts and Wilson intervals;
  ``metrics/01_detection_rate`` retains the detailed interval plot.

- ``comparison/taxonomy/04_taxonomy_comparison.svg``/``.pdf``: compact grouped horizontal bars
  comparing every available approach within each taxonomy field. Each classifier has a distinct
  colour and percentage labels. Field denominators are printed; missing/unsupported comparisons are
  N/A, and the companion CSV retains each classifier’s exact counts and denominator.

- ``classifiers/<classifier>/figures/06_taxonomy_accuracy.svg``/``.pdf``: horizontal exact
  field-accuracy percentages for each classifier. Each bar uses its field’s full exported comparison
  count; multi-label accuracy requires an exact set match. Unsupported fields or zero comparisons
  show N/A. This complements ``03_taxonomy_fields``, which also reports multi-label F1.

.. _evaluation-metric-views:

Metric views
------------

A human ``accept`` or ``accept_adjacent`` is a positive; ``exclude`` is a negative. An actual
``accept`` or ``accept_adjacent`` predicts positive.

The report provides three views so uncertainty and missing results cannot disappear behind one
denominator:

- **Operational:** every actual result other than inclusion predicts negative. This is the
  end-to-end screening view: an uncertain, failed, missing, or unretrieved expected inclusion is an
  FN; those results on an expected exclusion are TN.

- **Strict:** every non-definitive result is an error. On expected inclusions it is FN; on expected
  exclusions it is FP. Only explicit correct inclusion/exclusion decisions count as correct.

- **Selective:** uses only actual definitive include/exclude decisions. It measures correctness
  where the system committed to a decision; coverage and abstention must be reported beside it.

In plain language, the views answer different questions:

.. list-table::
   :header-rows: 1
   :class: radar-table

   * - View
     - Question answered
     - Uncertain/missing when inclusion was expected
     - Uncertain/missing when exclusion was expected
   * - Operational
     - What does the complete pipeline actually deliver in the final included set?
     - False negative
     - True negative
   * - Strict
     - What happens if every unresolved boundary case is considered wrong?
     - False negative
     - False positive
   * - Selective
     - How accurate is the system only where it made a definite decision?
     - Omitted
     - Omitted

For example, assume a repository should be excluded but the pipeline returns ``uncertain``.
Operational evaluation counts it as a true negative because it was not included. Strict evaluation
counts it as a false positive because the required explicit exclusion was not produced. Selective
evaluation removes it from its denominator. If an entity should instead be included, an uncertain or
missing result is a false negative in both operational and strict evaluation and is omitted
selectively.

Selective results must always be read together with ``definitive_coverage`` and ``abstention_rate``.
Otherwise, a system could appear highly accurate simply by leaving difficult cases unresolved.
Operational is the primary end-to-end screening view; strict is the conservative boundary-quality
view; selective describes committed decisions only.

Each view reports TP, FP, TN, FN and:

- DR / recall / sensitivity = TP / (TP + FN)
- FPR = FP / (FP + TN)
- FDR = FP / (TP + FP), the share of predicted inclusions that are false
- FNR = FN / (TP + FN)
- PREC / precision = TP / (TP + FP)
- ACC / accuracy = (TP + TN) / the view’s denominator
- specificity, negative predictive value, F1, and balanced accuracy

The report also records exact decision accuracy, classification-field exact accuracy, per-field
multi-label F1, retrieval coverage, classification coverage, definitive coverage, abstention rate,
status counts, and the number of actual entities that were not in the rated set. Unrated actual rows
are reported but do not enter rated-set metrics.

.. _evaluation-artifacts:

Artifacts
---------

Each evaluation directory contains:

- ``metrics.json``: machine-readable summary, all three metric views, and ``taxonomy_field_metrics``
  with per-field and per-category results.

- ``instances.csv``: one rated entity per row, configured selector and resolved stable ID,
  progression status, expected/actual decision, exact field results, and full JSON snapshots.

- ``decision_confusion.csv``: exact expected-to-actual disposition counts, including uncertain and
  unavailable.

- ``field_metrics.csv``: support status plus exact and multi-label metrics for every configured
  field; unsupported baseline fields are N/A.

- ``README.md``: compact human-readable headline values, metadata accuracy, metric semantics, and
  figure links.

- ``figures/01_decision_views.svg``/``.pdf``: operational, strict, and selective DR, precision,
  accuracy, F1, FPR, FDR, and FNR.

- ``figures/02_decision_confusion.svg``/``.pdf``: exact expected-to-actual dispositions, retaining
  uncertain and unavailable outcomes.

- ``figures/03_taxonomy_fields.svg``/``.pdf``: exact accuracy and macro/multi-label F1 for every
  configured taxonomy dimension, including observation objects and directly classified IDS types.

- ``figures/04_taxonomy_categories.svg``/``.pdf``: precision, recall, and F1 for every
  controlled-vocabulary value, grouped by taxonomy dimension.

- ``figures/05_pipeline_coverage.svg``/``.pdf``: retrieval, classification, definitive-decision,
  abstention, and final-status summaries.

- ``figures/06_taxonomy_accuracy.svg``/``.pdf``: presentation-style exact field accuracy with
  horizontal bars, percentage labels, and rated denominators.

- ``figures/08_reference_uncertainty.svg``/``.pdf``: uncertain decisions divided by attempted
  classifications of rated reference items, with uncertain/attempted counts. Missing retrieval and
  classification are excluded; errors remain in the denominator but are counted separately in the
  CSV. A matched artifact classification takes priority over the OpenAlex cited-paper fallback, so
  each reference entity contributes once. N/A means no attempts. This is distinct from the overall
  abstention metric and end-to-end unresolved artifact coverage.
  ``comparison/02_reference_uncertainty`` compares this rate across all approaches.

- ``figures/data/``: the exact aggregate CSVs behind the single-classifier figures.

Decision-distribution legends use **Accept**, **Exclude**, and **Uncertain**. Accept combines
OpenAlex ``accept`` and ``accept_adjacent``; these charts describe output decisions, not their
correctness.

Figure formats and resolution use the shared ``[plots]`` ``formats`` and ``dpi`` settings. The run
manifest additionally records the operational TP, FP, TN, FN, rated entity count, and artifact
count. This makes recurring runs comparable without treating the rating profiles as inputs to the
classifier itself.

.. _evaluation-regression-tests:

Regression tests
----------------

GitHub includes the single-label ``centralization`` dimension: ``centralized``, ``hierarchical``,
``fully_distributed``, or ``unknown``. It describes detection/analysis topology, not observation
scope, repository role, or IDS type. The LLM judges it directly; keyword and NLI baselines emit the
same vocabulary. Central/single-node phrases indicate centralized systems; collaborative IDS/CIDS
indicate hierarchical systems in the keyword baseline; distributed IDS/DIDS and peer-to-peer phrases
indicate fully distributed systems. These are lexical proxies: the LLM and reference annotations
prioritize explicit topology over acronyms. Bare “distributed” is not a keyword because it also
occurs in DDoS descriptions.

Centralization is specific to GitHub security systems. OpenAlex dataset reference entities and
classification fields do not include detection topology.

The GHML23 profile records per-repository topology plus sources/rationales for supported
annotations. ``unscored_field_values = { centralization = ["unknown"] }`` excludes only unknown
**reference** topology from field scoring; those repositories still participate in screening and
other taxonomy metrics. A prediction of unknown against known reference topology remains an error.
The current audited subset has centralized and hierarchical examples, but no verified fully
distributed examples; metrics therefore cannot establish performance on that class yet.

Centralization appears in single-classifier taxonomy/category figures and in the shared
keyword/NLI/LLM/ensemble taxonomy comparison heatmaps. General GitHub figures
``13_ids_centralization`` and ``14_centralization_role_matrix`` summarize canonical operational
systems, retaining unknowns. Each has CSV records and the configured figure formats (SVG and PDF by
default). Adding the field changes the GitHub classification fingerprint; existing classifications
cannot acquire model-judged topology from replotting or reevaluation alone. No OpenAlex taxonomy
changes are required.

``tests/test_evaluation.py`` validates the imported GHML23 snapshot, the survey-derived SHIELD
fixture, generic TOML/CSV loading, configurable job selection, and temporary saved classification
rows. It checks exact confusion matrices, requested metric aliases, abstention and missing-result
behavior, field comparisons, validation failures, artifact creation, both CLI stages, and manifest
counts. These tests are deterministic and network-free.

The lower-level ``radar.common.regression.evaluate_regression(...)`` dictionary helper remains
useful for small in-code unit fixtures. The evaluation stage is the persisted, centrally configured
run-level counterpart.
