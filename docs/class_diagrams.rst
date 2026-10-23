Class diagrams
==============

These UML views describe the current Python implementation. Each class shows its
module, selected attributes and public operations. Open an SVG to zoom, or use
its PDF for a document. The :doc:`class_inventory` and the
:download:`complete PlantUML model <_static/models/uml-all-classes.puml>` contain
all production classes, including classes with the same name in different
modules. Instance attributes assigned inside methods are not exhaustively listed.

A dashed line with a hollow triangle means **structural protocol realization**:
the analyzer found the required members and compatible method call shapes
without Python inheritance. Such arrows are labelled ``structural candidate``;
static diagram analysis does not prove full type compatibility. A solid line with a hollow triangle means actual inheritance.
Solid arrows describe held references; dashed arrows describe calls, accepted
values or returned values. These are selected implementation relationships, not
a complete call graph. Associations do not claim exclusive object ownership.
All edges come from code analysis; the machine-readable inventory records their
source locations. Dynamic dispatch and factories without type annotations may
leave relationships unresolved.

RADAR's CLI and pipeline orchestration are largely module functions. There is no
central ``Pipeline`` class. Use :doc:`architecture` for the execution flow and
:doc:`database_model` for persisted relationships.

Complete SVG overview
---------------------

.. figure:: _static/models/uml-all-classes.svg
   :target: _static/models/uml-all-classes.svg
   :alt: All RADAR production classes and relationships inferred automatically from Python source.
   :width: 100%

   Complete class diagram generated from code. Open the SVG to zoom; clicking
   a class in the standalone SVG opens a focused view with selected members.

:download:`Complete class diagram — SVG <_static/models/uml-all-classes.svg>` ·
:download:`PDF <_static/models/uml-all-classes.pdf>`

The generator resolves imported aliases, base classes, field and method type
annotations, stored constructor parameters, constructor calls and annotated
factory results. Protocol candidates require both their declared attributes and
compatible method call shapes. Newly added classes appear automatically. The
focused layouts below group familiar subsystems for readability; their class
contents and edges are generated from the same analysis.

Provider discovery
------------------

The source analysis identifies every discovery provider as a structural
candidate for ``DiscoveryAdapter``. OpenAlex and Semantic
Scholar share ``JsonTransport``; GitHub retains its own rate-limit-aware client.

.. figure:: _static/models/uml-discovery.svg
   :target: _static/models/uml-discovery.svg
   :alt: DiscoveryAdapter realized by GitHub, OpenAlex and Semantic Scholar providers, with HTTP client associations.
   :width: 100%

   Provider discovery and HTTP clients.

:download:`SVG <_static/models/uml-discovery.svg>` ·
:download:`PDF <_static/models/uml-discovery.pdf>` ·
:download:`PlantUML <_static/models/uml-discovery.puml>`

Optional provider capabilities
------------------------------

``LookupAdapter`` resolves identifiers, ``EvidenceAdapter`` enriches evidence,
``ReferenceAdapter`` traverses scholarly references and ``CatalogueAdapter``
exposes a specialized collection strategy. Capabilities are independent.
OpenAlex implements all four; Semantic Scholar implements lookup and evidence;
GitHub implements catalogue collection. ``PaperEvidenceService`` can retain
multiple evidence adapters. Catalogue support remains specific to the declared
provider workflow.

.. figure:: _static/models/uml-capabilities.svg
   :target: _static/models/uml-capabilities.svg
   :alt: Optional provider protocols, their concrete implementations and the shared paper evidence service.
   :width: 100%

   Optional provider capabilities.

:download:`SVG <_static/models/uml-capabilities.svg>` ·
:download:`PDF <_static/models/uml-capabilities.pdf>` ·
:download:`PlantUML <_static/models/uml-capabilities.puml>`

Discovery records and configuration
-----------------------------------

``ProviderRegistration`` pairs a descriptor with a factory. ``DiscoverySource``
represents configured discovery, while ``PaperSearchClient`` adapts a provider
for common scholarly collection. ``SourceRecord`` separates native identity,
metadata and evidence from study judgments. Search pages contain provider-native
dictionaries; normalization produces ``SourceRecord`` instances.

.. figure:: _static/models/uml-records.svg
   :target: _static/models/uml-records.svg
   :alt: Provider registrations and discovery sources associated with discovery protocols and normalized value types.
   :width: 100%

   Discovery configuration and normalized records.

:download:`SVG <_static/models/uml-records.svg>` ·
:download:`PDF <_static/models/uml-records.pdf>` ·
:download:`PlantUML <_static/models/uml-records.puml>`

Screening and evidence
----------------------

``StudyScreeningAdapter`` inherits from ``LocalSourceAdapter`` and applies a
``StudyProtocol``. These local normalization adapters serve a different purpose
from the file discovery provider. ``SourceContract`` defines record identity,
mappings and taxonomy. ``EvidenceBlockIndex`` retains exact source blocks for
criterion assessments; ``SourceQuoteIndex`` resolves quotation spans for the
catalogue path. Their outputs remain separate value types.

.. figure:: _static/models/uml-screening.svg
   :target: _static/models/uml-screening.svg
   :alt: Study screening inheritance, source contracts, study protocols, and the distinct source block and quotation indexes.
   :width: 100%

   Scientific screening and evidence grounding.

:download:`SVG <_static/models/uml-screening.svg>` ·
:download:`PDF <_static/models/uml-screening.pdf>` ·
:download:`PlantUML <_static/models/uml-screening.puml>`

Model interfaces
----------------

The four ``Generator`` protocols belong to different modules. ``OllamaClient``
meets each structurally. ``GitHubEvidenceClient`` meets the repository pipeline's
``EvidenceEnricher`` protocol; this is distinct from the provider-level
``EvidenceAdapter`` contract.

.. figure:: _static/models/uml-generators.svg
   :target: _static/models/uml-generators.svg
   :alt: OllamaClient realizes four module-specific Generator protocols, and GitHubEvidenceClient realizes EvidenceEnricher.
   :width: 100%

   Model generation and repository evidence contracts.

:download:`SVG <_static/models/uml-generators.svg>` ·
:download:`PDF <_static/models/uml-generators.pdf>` ·
:download:`PlantUML <_static/models/uml-generators.puml>`

``NLIBackend`` is implemented by the local Transformers backend and the Ollama
scoring backend. Runtime configuration, study hypothesis profiles, evidence
chunks and three-way scores are distinct types.

.. figure:: _static/models/uml-nli.svg
   :target: _static/models/uml-nli.svg
   :alt: NLI backend implementations, runtime settings, hypothesis profiles, evidence chunks and score values.
   :width: 100%

   NLI backends and supporting types.

:download:`SVG <_static/models/uml-nli.svg>` ·
:download:`PDF <_static/models/uml-nli.pdf>` ·
:download:`PlantUML <_static/models/uml-nli.puml>`

Supporting classes
------------------

These views include the remaining configuration, persistence, provenance,
scoring, evaluation and utility types. External exception and formatter bases
are printed inside the class boxes and included as inheritance in the PlantUML
sources. Classes without connecting arrows are used by module functions or have
relationships outside the focused view.

.. figure:: _static/models/uml-support-1.svg
   :target: _static/models/uml-support-1.svg
   :alt: Supporting keyword profiles, settings, evaluation results, exceptions and diagnostics.
   :width: 100%

   Supporting classes, first view.

:download:`SVG <_static/models/uml-support-1.svg>` ·
:download:`PDF <_static/models/uml-support-1.pdf>` ·
:download:`PlantUML <_static/models/uml-support-1.puml>`

.. figure:: _static/models/uml-support-2.svg
   :target: _static/models/uml-support-2.svg
   :alt: Supporting provenance, reporting, run manifests, scoring, screening and funnel types.
   :width: 100%

   Supporting classes, second view.

:download:`SVG <_static/models/uml-support-2.svg>` ·
:download:`PDF <_static/models/uml-support-2.pdf>` ·
:download:`PlantUML <_static/models/uml-support-2.puml>`

Regenerate the models
---------------------

From an installed development checkout:

.. code-block:: bash

   python scripts/generate_design_models.py
   python scripts/generate_design_models.py --check
   PYTHONPATH=src python -m unittest discover -s tests -p test_design_models.py -v

Generation reads Python sources and executes SQL schema constants only in
in-memory databases. It makes no provider or model calls. CairoSVG, already a
runtime dependency, exports the PDFs. ``--no-pdf`` generates SVG, PlantUML,
Mermaid and inventories with the Python standard library alone; ``--output``
selects another output directory. The documentation build uses committed assets
and needs no diagram renderer.

Class members, UML edges and database schemas are derived automatically. Only
the presentation of focused views is configured in the generator. The complete
SVG and support views include new classes automatically. Tests cover newly
introduced classes, imported aliases, changed bases and associations, protocol
call shapes and SQL key semantics. ``--check`` detects stale generated SVG and
source assets without modifying files; the documentation workflow runs it before
building Sphinx. SVG is the primary figure format; PlantUML and Mermaid files are
optional editable exports, not renderer requirements.

.. toctree::
   :maxdepth: 1

   class_inventory
