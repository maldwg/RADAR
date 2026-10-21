RADAR
=====

**Reproducible Acquisition, Discovery, Assessment, and Reporting**

RADAR is an auditable research framework for discovering repositories and
scholarly papers, screening their evidence against explicit study protocols,
and evaluating results against external references. It keeps provider access,
scientific decisions, and reporting separate so studies can reuse infrastructure
while maintaining their own search vocabulary, criteria, taxonomy, and prompts.

The supplied studies cover IDS repositories, IDS datasets and generators,
and conformal inference for treatment-effect estimation. Review studies can
combine multiple paper providers into a single deduplicated candidate cohort.
Catalogue workflows currently use their specialized GitHub and OpenAlex paths.

Start with :doc:`quickstart`, inspect the :doc:`architecture`, and use
:doc:`reproducibility` to reproduce or audit a run.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   quickstart
   configuration
   pipeline

.. toctree::
   :maxdepth: 2
   :caption: Architecture and methods

   architecture
   class_diagrams
   database_model
   adapters
   classification_consensus
   evaluation
   reporting
   cross_field_verification

.. toctree::
   :maxdepth: 2
   :caption: Research artifact

   reproducibility
   research_notes
   development
   ghml23_repository_audit
   llm_nli_plan

.. toctree::
   :maxdepth: 2
   :caption: Developer reference

   api
