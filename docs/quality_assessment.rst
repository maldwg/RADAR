Independent content quality
============================

Every study defines content quality in its own ``screening.toml``. Criteria
state what a paper or artifact documents; taxonomy and classification prompt
files continue to describe scientific classes. Each ``study.toml`` explicitly
selects ``quality = "../../shared/quality.toml"``. This shared TOML owns the
quality instructions, evidence limits and NLI defaults. Framework code loads
those instructions and constructs response schemas without editing class
templates.

The outputs keep three dimensions explicit:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``relevance_decision``
     - Relevance/eligibility, independently of content quality
   * - ``quality_status``
     - ``pass``, ``fail``, ``unknown``, ``error``, ``not_assessed`` or ``not_configured``
   * - ``quality_qualified_decision``
     - Inclusion after combining that approach's relevance and quality outcomes

``decision`` retains the original pipeline/model output. In catalogue workflows
it is the class decision; the conformal review retains its protocol-level
eligibility-and-quality screening decision. The main study CLI evaluations and
cross-classifier agreement plots explicitly use ``relevance_decision`` for all
approaches. Their metrics record ``decision_column = "relevance_decision"``.
This avoids counting a quality rejection as a relevance-classification error.
Standalone ``report`` and ``evaluate_results`` interfaces can still evaluate an
explicitly supplied prediction view; ``evaluate_results`` accepts
``decision_column`` to choose it. A nondefault view is saved as
``evaluated_predictions.csv`` in the evaluation directory, with the original
``pipeline_decision`` retained. Metrics pin both the source predictions and the
evaluated view by hash, and the reporting database freezes both files. This
keeps metrics, agreement plots and stored decisions tied to the exact values
that were evaluated.

Configuration
-------------

The shared :download:`quality configuration <../config/shared/quality.toml>` contains:

* ``[prompts]``: common instructions and separate quotation/source-block templates.
* ``[evidence]``: ``max_quotation_passages`` and ``max_block_passages`` per criterion.
* ``[nli]``: quality cutoffs and premise/hypothesis templates for the selected NLI backend.

Both RADAR and plain LLM receive the shared quality instructions in their
existing request. Their grounding, acquisition and voting remain separate.
A zero block limit permits any number of supporting blocks; the default
quotation limit is four. Limits are enforced in both response schemas and
local validation. Eligibility and taxonomy evidence limits are unaffected.
Python retains schema construction, grounding, threshold arithmetic, voting
and reporting. Missing or malformed shared profiles fail configuration
validation; there are no embedded quality-prompt defaults.

In each study's ``screening.toml``, define its scientific criteria and pass
thresholds, for example:

.. code:: toml

   [quality]
   Q1 = "The artifact documentation clearly describes its purpose."
   Q2 = "The artifact documentation describes inputs and outputs."
   Q3 = "The artifact documentation gives a concrete execution example."

   [decision]
   strategy = "repository_role_validation"
   quality_minimum = 2
   quality_required = ["Q1"]

Quality assessment is optional. Omit ``[quality]`` or leave it empty to disable
it. Omit its decision settings too, or explicitly set:

.. code:: toml

   [quality]

   [decision]
   strategy = "repository_role_validation"  # retain your study's strategy
   quality_minimum = 0
   quality_required = []

All approaches then report ``quality_status = "not_configured"`` and the joint
decision equals relevance. No quality judgments or NLI quality scoring are
requested. Classification errors remain classification errors and are not
reported as quality failures. Nonzero minimums and nonempty mandatory lists
are rejected when no quality criteria exist.

A quality pass requires at least ``quality_minimum`` yes judgments and yes for
every mandatory criterion. A supported no on a mandatory requirement, or an
unreachable minimum, fails quality. Insufficient evidence remains unknown.
Missing or truncated text does not demonstrate that a requirement fails.
Malformed quality responses and scoring failures have their own error state;
they do not replace a valid relevance prediction.

The repository study starts with five documentation criteria: purpose,
inputs/outputs, dependencies/environment, a concrete usage workflow, and
evaluation/limitations. It requires three yes judgments, including purpose and
usage. The dataset study starts with five reporting criteria: collection or
generation procedure, contents/formats, labeling provenance, access or
reproduction instructions, and limitations. It requires three yes judgments,
including procedure and contents. These are configurable starting protocols,
not quality labels inferred from the reference benchmarks. Review and
predeclare them before making experimental claims.

The conformal study retains the published nine criteria and five-of-nine
minimum. Its existing provenance remains in ``screening.toml``.

Assessment by approach
----------------------

.. list-table::
   :header-rows: 1

   * - Approach
     - How it assesses quality
     - Evidence and processing
   * - RADAR LLM
     - Criterion yes/no/unknown judgments in the same classification request
     - Uses its current evidence; definite catalogue judgments require source-grounded quotations. Repeated runs and ensembles vote quality independently.
   * - Plain LLM
     - Criterion judgments in its single model request
     - Uses original collection evidence. Model claims remain unverified; there are no quality correction calls, retries or enrichment.
   * - Local NLI
     - The same NLI model scores each quality description as a hypothesis
     - Uses original collection evidence and its ordinary chunking interface. Strong entailment supports yes; strong contradiction supports no; neutral or conflicting passages remain unknown.
   * - Ollama NLI
     - Its own class-token NLI scorer assesses the quality hypotheses
     - Uses original collection evidence; no separate structured-LLM quality judge is inserted.
   * - Keyword
     - Semantic content quality is unsupported
     - Reports ``not_assessed`` with no yes/no judgments and no hidden model calls.

NLI quality thresholds are independent of relevance thresholds. Their defaults
live in ``config/shared/quality.toml``:

.. code:: toml

   [nli]
   entailment_threshold = 0.5
   label_margin = 0.1
   hypothesis_template = "${criterion}"
   premise_template = "Evidence source: ${source}.\n${text}"

A study may explicitly override the cutoffs with
``baselines.nli.quality_entailment_threshold`` and
``baselines.nli.quality_label_margin`` in its baseline settings. Ordinary NLI
and Ollama NLI use the same resolved quality defaults or overrides.
Offline relevance-cutoff sensitivity keeps quality judgments
fixed. Changing quality cutoffs requires a new assessed run. Metadata-only
inputs yield unknown quality without scoring synthetic absence text. NLI keeps
source chunks, hypotheses and three-way scores in the quality audit. Contradiction
scores are model estimates and still need independent calibration.

Quality format/grounding failures never trigger an additional correction
request. Catalogue quality uncertainty does not trigger evidence expansion;
review uncertainty-driven expansion now depends on relevance uncertainty.
Literal source grounding establishes that quoted text is present, not that its
meaning actually proves a quality requirement. Neither source IDs nor NLI
scores remove the need for independent scientific validation.

Voting quality
--------------

RADAR votes criterion values separately from relevance, keeping failed and
missing voters in the denominator. A passing representative cannot supply the
ensemble's quality result by itself. Catalogue quality votes also require the
same scientific policy and visible evidence; differing evidence produces an
explicit quality error. Review ensembles retain their existing same-evidence
requirement and separate relevance and quality outcomes.

Each approach retains its own evidence policy. Baselines do not inherit RADAR's
expanded full text, repository documentation, quality judgments or reconciled
results. Thus a primary RADAR/plain-model contrast still includes the combined
effects of additional evidence and computation. It does not isolate one
quality-processing component or establish independent retrieval performance.

Reports and scientific comparison
-----------------------------------

Run the existing commands; no extra inference command is needed:

.. code:: bash

   radar --study ids_repositories run
   radar --study ids_datasets baseline --methods keyword,nli,plain_llm
   radar --study conformal_treatment_effects baseline --methods plain_llm

Catalogue evaluations write a ``quality/`` bundle alongside their relevance
metrics. Review reporting writes it alongside the review figures. Unified
classifier comparisons also have a separate ``quality/`` directory containing:

* ``summary.csv``: outcomes, assessment coverage, and relevance/joint inclusion counts.
* ``criteria.csv``: criterion judgments, supported criteria, backend and evidence/score audit.
* ``<classifier>/quality_outcomes.svg`` and matching configured formats: outcome counts.
* ``<classifier>/quality_qualified.csv``: a separate joint-inclusion prediction view.
* ``<classifier>/quality_qualified.audit.json``: input/output hashes and explicit view semantics.

The joint CSV retains every attempted record. It records the original pipeline
decision separately and does not overwrite classifier outputs, reference labels
or existing catalogues. Use an appropriate independently annotated reference
profile before interpreting it as a quality-qualified accuracy benchmark.

``quality_accuracy`` is N/A: no independent quality ground truth was supplied.
Quality pass rates are assessor outputs, not accuracy, and unsupported keyword
quality must not be treated as a failed semantic assessment. Assessment coverage
counts valid pass/fail/unknown outcomes; definitive coverage counts pass/fail
only. Errors, unknowns and unsupported criteria stay visible separately.
Quality comparisons reject mismatched scientific criteria or thresholds.

Classifiers retain separate scientific ``quality_profile_hash``, shared
``quality_config_hash`` and evidence fingerprints. The study manifest records
the shared file's SHA-256; review checkpoints also preserve its resolved
contents and hash. Catalogue and plain-model profile hashes bind it, and
review ensembles reject checkpoints assessed under another shared profile.
Editing this TOML therefore invalidates cached assessments without mixing
quality instructions into classification or taxonomy files.

Standalone review APIs resolve the explicit quality reference from the
``study.toml`` alongside the screening profile. When a protocol is stored
elsewhere, pass ``quality_path`` to the Python API or ``--quality-config`` to
``study-classify`` / ``study-screen``.
NLI records relevance and quality inference times separately, with total time
covering both. LLM relevance and quality share one request, so token/time usage
is joint and is not artificially allocated between the two tasks.

A changed quality protocol changes model requests or scoring hypotheses and
invalidates cached assessments. Use a fresh matched run for all approaches;
older predictions are not retroactively assigned quality judgments.
