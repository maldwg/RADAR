.. _reporting-portable-and-verifiable-reporting:

Portable and verifiable reporting
=================================

Reporting works with existing GitHub/OpenAlex outputs and with locally supplied repository, paper or
artifact records. Collection and classifier execution still use the existing source pipelines.
Adding another API crawler requires implementing its adapter; reporting on its CSV/JSON exports does
not.

See :doc:`database_model` for the SQLite tables, keys and relationships.

.. _reporting-use-another-data-provider:

Use another data provider
-------------------------

A versioned TOML profile defines the provider name, entity kind, native identity, display label,
reference selectors, column mappings, taxonomy and available capabilities. Native decisions can be
mapped to ``accept``, ``accept_adjacent``, ``exclude``, ``uncertain`` or ``error``. See the complete
synthetic profiles and inputs in the ``examples/reporting/`` directory. Start with
the :download:`repository profile <../examples/reporting/repositories.toml>` or
the :download:`paper profile <../examples/reporting/papers.toml>`.

.. code:: bash

   radar report \
     --profile examples/reporting/repositories.toml \
     --input examples/reporting/repositories.csv \
     --candidates examples/reporting/repositories.csv \
     --ratings examples/reporting/repository_references.toml \
     --output artifacts/plots/reporting_examples/repositories

   radar report \
     --profile examples/reporting/papers.toml \
     --input examples/reporting/papers.json \
     --ratings examples/reporting/paper_references.toml \
     --output artifacts/plots/reporting_examples/papers

``--candidates``, ``--ratings`` and ``--exports`` are optional. Classification inputs must
explicitly declare ``classifier_supported_fields`` (``*`` for all profile fields).
``--database PATH`` imports into a shared reporting database instead of the default
``OUTPUT/reporting.sqlite3``. These commands perform local normalization, evaluation and plotting;
they do not call a model or collect new records. The existing global ``--no-subcaptions`` option
also applies.

For an external review’s explicit screening rules, :doc:`study-screen <cross_field_verification>`
converts local evidence-backed criterion assessments into these classification inputs. The
Memmesheimer profile includes eleven reviewed positives; metrics requiring negative references
remain N/A for positive-only paper and artifact benchmarks.

Inputs must be CSV records or a JSON array of objects. The importer retains native fields, adds
provider-scoped entity IDs and rejects missing/duplicate identities or unknown decisions. Array
taxonomy fields become semicolon-delimited sets. Titles do not merge records. Missing stages are
N/A. Classification stages require decisions; imported metadata alone is not a classification.
Candidate and classification sets can differ, so stage counts do not automatically imply nested
sets. Supplied exports must belong to accepted classification outputs. The publication/citation
views require both declared capabilities and populated columns;
``figures/data/plot_availability.csv`` records unavailable views.

Count charts use ``plots.top_n`` to keep large vocabularies readable; their CSVs and contribution
maps retain all categories.

.. _reporting-verify-a-reporting-bundle:

Verify a reporting bundle
-------------------------

.. code:: bash

   radar verify --input artifacts/plots/reporting_examples/repositories
   radar verify --database artifacts/plots/reporting_examples/repositories/reporting.sqlite3

Verification prints JSON and returns exit code 0 on success or 2 on disagreement/incomplete audit
data.

Each generated figure has a matching ``.manifest.json``. It records rendering settings, renderer
code hash and snapshot, frozen input hashes, exact exported table hashes, artifact hashes and the
available verification recipe. ``.provenance/objects/`` contains the frozen bytes. Keep this
directory with a copied figure bundle; changing the original live input does not invalidate the
historical snapshot. Changing a rendered artifact, exported table or frozen object is detected.
Hashes detect inconsistencies; they are not signatures against an attacker replacing the entire
bundle.

Two verification levels are explicit:

- **Recomputed:** independent checks reproduce the table values from frozen records or evaluation
  instance audits. The local provider counts and stage views, and all eight standard evaluation
  diagnostics, have these recipes. Taxonomy tables retain exact record contributors or link to field
  checks in ``instances.csv``.

- **Integrity only:** the exact inputs, tables and figure bytes are linked and checked, but a
  numerical replay recipe has not been declared. Existing specialized catalogue/timeline,
  architecture and comparison renderers preserve their styles and receive manifests at this level.
  Existing source funnels also capture a consistent SQLite snapshot through a read-only connection,
  including WAL data. Their historical stage attribution continues to use the existing
  observation/capacity and file-freshness rules; the manifest does not upgrade these rules to
  verified historical run membership.

Independent evaluation checks reconstruct operational/strict/selective confusion counts, coverage
denominators, exact field matches, category scores and Wilson intervals. Database verification
additionally checks SQLite integrity, foreign keys, snapshot bytes, record/decision consistency,
reference matches and figure recipes using stored bytes. It can run without the original corpus or
figure files.

.. _reporting-reporting-database:

Reporting database
------------------

Schema version 1 is additive and separate from retrieval/inference caches. It stores:

- ``entities``, ``identifiers``: stable IDs scoped by provider, entity kind and native ID.
- ``snapshots``, ``source_records``: compressed input bytes and their normalized records.
- ``runs``, ``decisions``, ``evidence``, ``stage_memberships``: interpretation contract, results,
  stored classifier evidence and observed memberships.

- ``relationships``: explicit typed links with a documented basis; no automatic cross-provider
  identity merging.

- ``reference_sets``, ``reference_entities``, ``evaluations``, ``reference_matches``: expected
  labels, metric definitions and individual matches.

- ``figure_manifests``: figures and all their frozen inputs, tables, code and artifacts.

An import is transactional per evaluation and repeat imports are idempotent. Supplying a retrieval
cache as the reporting database is rejected. Unsupported schema versions are rejected. Evidence
provenance means stored classifier evidence.

New evaluations pin classification/candidate hashes and freeze reference profile inputs, including
external reference CSVs. Existing GitHub/OpenAlex evaluation commands create a local reporting
database and index generated evaluation figure manifests.

.. _reporting-backfill-saved-evaluations:

Backfill saved evaluations
--------------------------

.. code:: bash

   radar audit-import \
     --input artifacts/evaluation \
     --database data/reporting/reporting.sqlite3
   radar verify --database data/reporting/reporting.sqlite3

This imports current saved audits without classification or retrieval. Source contracts and
reference, taxonomy, input and applicable cohort fingerprints are required. Missing or changed
inputs and inconsistent instance records are rejected. Imports do not reconstruct historical
provenance.

.. _reporting-compare-approaches:

Compare approaches
------------------

.. code:: bash

   radar report-compare \
     --evaluation 'Approach A=path/to/first/evaluation' \
     --evaluation 'Approach B=path/to/second/evaluation' \
     --output artifacts/plots/custom_comparison

Comparisons require matching reference-content, taxonomy and candidate-cohort fingerprints plus the
same evaluation unit. Identical reference copies can have different paths; changing reference
contents in place is detected. Classified output hashes must still agree with saved evaluations.
Unsupported scores remain N/A. Entity agreement is only computed within a provider; comparing
identities across providers requires an explicit reference mapping.

Comparisons assume evaluations generated by a clean run and use saved reports as supplied. They do
not automatically refresh older reports. Incompatible reference, taxonomy or cohort fingerprints
remain an error.

Run the reporting regression tests without inference:

.. code:: bash

   MPLCONFIGDIR=/tmp/radar-mpl python -m unittest tests.test_reporting -v
