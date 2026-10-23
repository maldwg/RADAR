Research notes
==============

Living development record for writing the paper. These are observed caveats and
engineering decisions, not controlled ablations or estimates of improvement.
Append a dated observation, response and remaining limitation when a new issue
affects research interpretation. Keep earlier observations as historical context.

2026-10-16: Generated quotations
--------------------------------

Models corrected spelling, reconstructed mathematical notation and joined
passages, even when asked to copy verbatim. PDF ligatures and line-wrap hyphens
also caused genuine matches to fail. We retained raw responses, grounded source
offsets and per-judgment warnings. Unsupported judgments became unknown without
an LLM correction request; technical/schema failures remained errors.

2026-10-16: Output limits
-------------------------

Responses reaching 4,000 output tokens ended inside JSON escapes. Repeating an
unchanged deterministic request reproduced the truncation. Transport retries
and the configured output limit must be recorded; a parsing failure is not a
scientific exclusion. Shorter evidence references reduce response size but do
not guarantee valid or complete generation.

2026-10-19: Missing full text
-----------------------------

An abstract or a PDF URL did not imply usable full text: publisher links returned
403 responses or HTML. The shared paper evidence chain now tries OpenAlex cached
PDF/TEI and alternative locations, then Semantic Scholar links. Text availability,
extraction quality and truncation remain separate from scientific eligibility.
``full_text_complete`` means no character-limit truncation, not proof of perfect
PDF extraction. Source URLs, attempts, formats and hashes are retained.

2026-10-20: Evidence selected by ID
-----------------------------------

Offline replay of the 100-paper conformal run found full text for all six
retrieved benchmark references, but three uncertain references per model and
four in the unanimous ensemble. Remaining blockers mixed non-verbatim quotations
with conservative PDF formatting checks. Review assessments now select multiple
source block IDs; Python copies their separate original passages and offsets.
TEI structure and PDF pages are preserved where available; unstructured text
uses bounded blocks, not guaranteed semantic paragraphs. References prove source
presence, not that a passage supports a claim. Catalogue quotation workflows
retain their separate evidence representation.

2026-10-20: Repetition, consensus and evaluation
------------------------------------------------

Most failing responses repeated identically with temperature 0 and seed 42.
Five repeated calls are not five independent observations. A unanimous ensemble
can abstain when one model accepts and another cannot ground evidence. The
eleven-paper benchmark contains positives only; five were outside this capped
cohort. Retrieval, evidence availability and screening outcomes must be reported
separately. Neither this audit nor the new implementation establishes a measured
performance improvement; that requires a fresh, frozen evaluation.

2026-10-20: Retrieval observability
-----------------------------------

Lookup-start messages obscured the difference between missing document links,
provider errors and non-PDF responses. Per-paper summaries and contextual
lookup/download outcomes now retain HTTP, media, redirect and fallback details.
Provider request failures remain visible after successful fallback; this improves
traceability without changing search scope or scientific screening criteria.

2026-10-21: Structured text before PDF extraction
-------------------------------------------------

A non-empty pypdf extraction previously prevented trying structured TEI, even
when its reading order or word layout was poor. The shared paper service now
prefers supplied or identity-matched provider TEI before PDF extraction. Semantic
Scholar discoveries can use OpenAlex's cached GROBID parse; pypdf remains the
fallback and PDF-only lookups remain deferred. Policy versions enter acquisition
and classification provenance. Existing frozen text is preserved. TEI can still
omit or misparse content, and this change has no measured screening improvement
until a fresh evaluation is completed.

2026-10-22: Baseline evidence isolation
---------------------------------------

Keyword and NLI branches read mutable cached records, so running them after
RADAR could expose them to later abstract/full-text or documentation enrichment.
Collection now preserves a separate, hashed original snapshot; keyword, local
NLI and Ollama NLI read only that version. Missing originals fail explicitly;
old overwritten text and earlier results are not reconstructed or relabelled.
The cohort and retrieval/preselection remain shared with RADAR, so this isolates
classification evidence rather than benchmarking independent discovery. No
performance improvement is claimed without a fresh, matched evaluation.

2026-10-22: Plain LLM comparison arm
------------------------------------

Added a configurable one-request-per-candidate model baseline using original
collection evidence and the same study definitions. Structural errors and
truncation remain errors; valid decisions are not scientifically repaired.
Raw replies, prompts, usage and model digests are retained. Cohort selection
remains shared with RADAR, which can subsequently acquire more evidence and
spend additional calls. The contrast measures these combined effects, not an
isolated processing or reconciliation improvement. No performance claim is
made before a fresh matched evaluation.

2026-10-22: Independent quality dimensions
------------------------------------------

Content-quality criteria now live in every screening profile. Relevance,
quality and joint inclusion are retained separately; primary classifier
metrics use relevance only. LLMs assess both in the same request, NLI uses
its own entailment/contradiction scorer, and keywords report semantic quality
as unassessed. Quality errors do not invalidate class predictions or request
LLM repairs. New IDS criteria/thresholds are starting protocols that require
predeclaration, not annotations derived from benchmark references. There is
no independent quality ground truth, so quality accuracy is N/A. Literal
grounding proves text presence, and NLI scores remain model estimates.

2026-10-23: Shared quality assessment configuration
---------------------------------------------------

Quality instructions, evidence limits and NLI defaults now come from an
explicit shared TOML profile; study criteria and pass thresholds remain
local. Its hash enters assessment caches and ensemble compatibility checks.
RADAR and plain LLM share assessment wording while retaining their different
evidence and processing policies; this does not isolate an individual
pipeline effect or supply independent quality ground truth.
