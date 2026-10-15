Installation and first run
==========================

RADAR requires Python 3.11 or newer. Run study commands from the repository root
so the default study manifest is available. A local Ollama service is required
for model-based assessment. Collection requires network access to the selected
providers; offline verification and the test suite do not require a model service.

Install
-------

.. code-block:: bash

   git clone https://github.com/maldwg/RADAR.git
   cd RADAR
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e '.[nli]'
   radar --help

The ``nli`` extra supplies the transformer dependencies used by the checked-in
catalogue baselines. Use ``python -m pip install -e .`` for the core framework
when those baselines are disabled or when running the conformal review.

Configure services
------------------

Set the Ollama URL to your service and make the models selected by
``config/shared/models.toml`` available there. The checked-in model list is
``gemma4:31b`` and ``qwen3.8:27b``. Model availability, context capacity, and
hardware requirements depend on your Ollama deployment.

.. code-block:: bash

   export OLLAMA_URL=http://127.0.0.1:11434
   export GITHUB_TOKEN=your-token
   export OPENALEX_MAILTO=researcher@example.org

``OPENALEX_API_KEY`` and ``SEMANTIC_SCHOLAR_API_KEY`` are read when provided.
Use environment variables for credentials. Configure transport pacing in
``config/shared/sources/`` and models in ``config/shared/models.toml``.
``OLLAMA_MODEL`` replaces the selected model list; adjust explicit ensemble
membership in the study's pipeline when making that override.

Validate studies
----------------

Validation checks the configuration without contacting providers or Ollama.

.. code-block:: bash

   radar --study ids_repositories validate
   radar --study ids_datasets validate
   radar --study conformal_treatment_effects validate

Run the three studies
---------------------

.. code-block:: bash

   radar --study ids_repositories run
   radar --study ids_datasets run
   radar --study conformal_treatment_effects run

Each study has its own output root at ``artifacts/studies/<study>/``. Start with
a clean output root for a new experiment. ``--resume`` explicitly continues
compatible work within the same current-format study run.

The checked-in collection limit is 100 candidates, with a query budget factor
of 3.0. ``classification_limit = "all"`` assesses all retained eligible rows;
it does not disable the prefilter. To select another collection budget:

.. code-block:: bash

   radar --study ids_datasets run --collect-limit 1000 --classify-all

Catalogue studies support individual ``collect``, ``prefilter``, ``classify``,
``reconcile``, ``export``, ``plot``, ``baseline``, and ``evaluate`` commands.
Review studies support ``collect``, ``prefilter``, ``classify``, ``screen``, and
``report``; ``run`` executes their workflow. See :doc:`pipeline` and
:doc:`cross_field_verification` for complete commands.

Select providers
----------------

The conformal manifest selects OpenAlex by default. Add the supplied Semantic
Scholar source configuration to combine both providers, as described in
:doc:`adapters`. An unqualified review run uses all configured providers;
``--source semantic_scholar`` selects that provider's separate cohort.
Multi-provider union collection currently belongs to the review workflow.

Build documentation
-------------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -n -W --keep-going -b html docs docs/_build/html
   python -m http.server --directory docs/_build/html 8000

Open ``http://localhost:8000``. The same reStructuredText sources and pinned
documentation dependencies are used by ``.readthedocs.yaml``.
