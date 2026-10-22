Database models
===============

RADAR uses two independent SQLite schemas: the **reporting database** stores
imported records, decisions and evaluation audits; the **pipeline cache** stores
retrieval and classification cache entries. There is no SQL relationship between
these files. Both schemas are accessed directly through ``sqlite3`` rather than
an ORM. Their tables therefore do not correspond one-to-one to Python classes.

The models below come from the literal ``SCHEMA`` definitions in
``radar.common.reporting`` and ``radar.common.cache``, inspected using SQLite's
own schema metadata in memory. They describe the current code, not row counts or
the contents of a particular run. ``PK1``, ``PK2``, etc. show composite key order;
``FK`` marks a declared foreign key; ``NN`` means an explicit ``NOT NULL``.
Crow's feet denote zero or more child rows. Edge labels give the required or
optional parent multiplicity for each child.

Reporting database
------------------

``ReportingDatabase`` owns this 15-table schema. ``radar report`` normally creates
``OUTPUT/reporting.sqlite3``; ``--database`` can select a shared reporting file.
See :doc:`reporting` for import and verification commands.

.. figure:: _static/models/er-reporting.svg
   :target: _static/models/er-reporting.svg
   :alt: Complete reporting database ER model with 15 tables and 18 foreign keys.
   :width: 100%

   Complete reporting schema. Open the SVG to zoom; the focused views below
   separate the main record and evaluation relationships.

:download:`Full SVG <_static/models/er-reporting.svg>` ·
:download:`Full PDF <_static/models/er-reporting.pdf>` ·
:download:`Editable Mermaid ER model <_static/models/er-reporting.mmd>`

Records, decisions and evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``entities`` provides internal identity; ``identifiers`` maps provider-native
identifiers to it. ``snapshots`` stores frozen inputs, including compressed
bytes; ``source_records`` associates record payloads with a snapshot and entity.
``runs`` groups decisions, stage membership and entity relationships.
``evidence`` belongs to a decision. A decision identifies its run, entity and
input snapshot.

.. figure:: _static/models/er-reporting-decisions.svg
   :target: _static/models/er-reporting-decisions.svg
   :alt: Reporting entities, identifiers, source snapshots, runs, decisions, evidence, stages and entity relationships.
   :width: 100%

   Records and assessment subset of the reporting schema.

:download:`SVG <_static/models/er-reporting-decisions.svg>` ·
:download:`PDF <_static/models/er-reporting-decisions.pdf>`

References and evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

``reference_sets`` and ``reference_entities`` contain the evaluation targets.
``evaluations`` compares a run with a reference set. ``reference_matches`` stores
per-reference outcomes; its ``entity_id`` and ``decision_id`` are nullable,
allowing unmatched or missing artifacts to remain in the audit.

.. figure:: _static/models/er-reporting-evaluation.svg
   :target: _static/models/er-reporting-evaluation.svg
   :alt: Evaluation runs and reference sets connected to per-reference matches with optional entity and decision links.
   :width: 100%

   Evaluation subset. The full schema additionally shows snapshots and the
   other acquisition and reporting tables.

:download:`SVG <_static/models/er-reporting-evaluation.svg>` ·
:download:`PDF <_static/models/er-reporting-evaluation.pdf>`

``reference_matches.reference_key`` has **no declared foreign key** to
``reference_entities``. The application joins it using the evaluation's
``reference_id`` plus ``reference_key``; this logical relationship is deliberately
not drawn as an enforced FK. ``schema_meta`` and ``figure_manifests`` are
standalone tables. Detailed evidence passages, contracts, metrics and figure
manifests mostly reside in JSON payloads, not additional normalized tables.

Pipeline cache
--------------

``PipelineCache`` owns three tables. ``records`` caches source records by
namespace and record key. ``decisions`` adds profile and evidence hashes to its
composite key, distinguishing different classification inputs. ``coverage``
records source/query/shard retrieval status, including an optional hit count.
Collection writes both a mutable source record and an original snapshot in a
``.collected`` namespace within ``records``. The snapshot payload includes its
policy version and content hash. Baselines require the verified original;
subsequent RADAR enrichment updates only the working namespace. A genuine new
collection replaces both versions. No extra SQL table is needed for these
separate namespaces.

There are **no declared foreign keys** in this schema. Shared key names in
``records`` and ``decisions`` do not establish an enforced relationship.

.. figure:: _static/models/er-cache.svg
   :target: _static/models/er-cache.svg
   :alt: Three independent cache tables showing retrieval, classification and coverage composite primary keys.
   :width: 100%

   Pipeline cache schema.

:download:`SVG <_static/models/er-cache.svg>` ·
:download:`PDF <_static/models/er-cache.pdf>` ·
:download:`Editable Mermaid ER model <_static/models/er-cache.mmd>`

``RunManifest`` writes a file-based run manifest; it is not an ORM representation
of the reporting database's ``runs`` table. Source evidence blocks are likewise
in-memory objects whose audit information can be serialized into payloads.

Regenerate all database and class models with
``python scripts/generate_design_models.py``. See :doc:`class_diagrams` for
regeneration details and :doc:`class_inventory` for the Python classes.
