Reproducing and auditing an experiment
======================================

RADAR makes the acquisition protocol, scientific assessment, and reporting
inputs inspectable. Live API results and model outputs can vary across time
and deployments. Reproduction therefore requires retaining the exact study
configuration, collected evidence, model identity, and generated audit bundle.

Record the environment
----------------------

Before a new experiment, record the repository revision, Python and dependency
versions, provider access settings, Ollama version, and installed model versions.
Keep the study directory and every shared profile it references with the run.
Credentials must remain outside the archived configuration.

.. code-block:: bash

   git rev-parse HEAD
   python --version
   python -m pip freeze
   ollama --version
   ollama list
   radar --study ids_datasets validate

Study manifests choose explicit output roots. Use a fresh root for a new
experiment, and retain the files under ``runs/`` with the candidate, evidence,
classification, and assessment audits. Resume is an explicit continuation of
compatible work; RADAR does not migrate old schemas or repair stale evaluations.

Evaluate the scientific outputs
-------------------------------

Catalogue studies evaluate saved classifications against their configured
reference sets. Review reports evaluate the combined cohort and each provider's
retained membership against the same references.

.. code-block:: bash

   radar --study ids_repositories evaluate
   radar --study ids_datasets evaluate
   radar --study conformal_treatment_effects report

Interpret recovery and taxonomy scores using their documented denominators.
The conformal reference set contains eleven positives and cannot estimate
false-positive performance. The dataset reference set's selected negative
controls do not represent all excluded literature. Repeated model votes do not
constitute independent statistical confidence estimates. See :doc:`evaluation`,
:doc:`classification_consensus`, and :doc:`cross_field_verification`.

Verify the reporting bundle
---------------------------

.. code-block:: bash

   radar verify --input artifacts/studies/ids_datasets/evaluation/openalex

Keep each figure's manifest, exported tables, and ``.provenance/objects/``
directory together. Verification checks frozen bytes and declared numerical
replay recipes. The report distinguishes ``recomputed`` from
``integrity_only`` verification. File hashes detect inconsistencies; they are
not cryptographic signatures authenticating the author. See :doc:`reporting`.

Scope of claims
---------------

Catalogues are bounded by the source index, query strategy, collection budget,
available evidence, and recorded coverage. They should not be described as
exhaustive beyond those observations. RADAR supplies an auditable artifact;
publication metadata, artifact badges, benchmark results, and licensing claims
must reflect the actual associated release.

The living :doc:`research_notes` records encountered caveats and design decisions
for paper writing; it does not report controlled ablations.
