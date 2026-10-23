Plain LLM benchmark
====================

``plain_llm`` measures a configured model making one direct classification from
original collection evidence. It is available for repository catalogues, dataset
catalogues and systematic reviews. It is separate from ``llm_nli``, which scores
individual premise/hypothesis pairs.

Run it after collection and prefiltering, or after a completed RADAR run:

.. code:: bash

   radar --study ids_repositories baseline --methods plain_llm
   radar --study ids_datasets baseline --methods plain_llm
   radar --study conformal_treatment_effects baseline --methods plain_llm

   # Isolated pilot; does not enter the main comparison:
   radar --study ids_datasets baseline --methods plain_llm --limit 5

   # Re-evaluate intact saved predictions without contacting Ollama:
   radar --study ids_datasets baseline --methods plain_llm --evaluate-only

All checked-in studies include this method in ``baselines/settings.toml``.
Complete ``run`` commands execute it when ``[baselines].enabled = true``.
Explicit ``--methods plain_llm`` runs only this method.

Configuration
-------------

By default, plain inference inherits ``ollama.models`` and the shared generation
settings in ``config/shared/models.toml``. To choose a different model list or
sampling budget, edit that file's dedicated table:

.. code:: toml

   [baselines.plain_llm]
   models = ["gemma4:31b", "qwen3.8:27b"]
   temperature = 0.01
   seed = 42
   max_output_tokens = 3000
   context_length = 131072
   timeout_seconds = 300

Omit overrides to inherit RADAR's settings. For reviews, the default output
budget follows RADAR's effective minimum of 4,000 tokens; an explicit baseline
override takes precedence. Model names must have distinct filesystem slugs.
Retry, enrichment, reconciliation and consensus options are not accepted in this
table. Plain inference always has zero retries and thinking disabled, matching
the existing structured Ollama client.

One ``GET /api/tags`` metadata request pins the installed model digests before a
non-empty inference run. It does not generate a completion. Each uncached
candidate then has exactly one ``POST /api/generate`` request. Neither request
has an automatic retry. Do not replace model tags while an experiment is running.

Inputs and scientific task
--------------------------

When current RADAR model outputs exist, the plain method reuses their candidate
IDs; all available configured RADAR models must agree on that cohort. Their
predictions never enter the prompt. This also preserves a prior CLI classification
cap. Otherwise it uses the same deterministic eligible shortlist and configured
classification limit as RADAR. Catalogue selection retains the same band/score
ranking; review selection retains the same eligible cohort order. The optional
``--limit`` takes a prefix of that shortlist and writes an isolated pilot.
Discovery, provider union and preselection remain shared with RADAR.

Catalogue evidence comes exclusively from the original ``.collected`` SQLite
records. The candidate CSV supplies identities and common audit metadata.
Review evidence comes from the original ``candidates.json``, verified against
its collection audit, while ``prefiltered.json`` supplies the shortlist.
For the existing ``run --input`` entry point, RADAR first preserves that supplied
initial cohort and an input audit before acquisition; this is not a discovery
provider. Later abstracts, full text and expanded repository documentation are excluded.
Missing or changed originals stop execution before inference. Historical runs
without saved originals need a fresh collection and matched experiment.

The study's classification prompts, shared quality instructions from the manifest-selected
``quality.toml``, eligibility/quality criteria and taxonomy are
reused. The response additionally asks for the model's final ``decision`` and
``primary_reason``. A separate quality answer is requested in the same call;
framework-owned instructions keep it independent of relevance. Reviews reuse
their existing Q judgments. Shared quality thresholds are applied only to the
separate quality/joint output; see :doc:`quality_assessment`.
Abstract-only inputs are still sent to the model; insufficient evidence can
therefore produce its own uncertain decision.

Only structural response checks run: required keys, JSON types, permitted labels
and schema format constraints. Invalid JSON, truncation, transport failures and
invalid relevance response shapes produce ``decision = "error"``. Malformed
quality responses instead produce ``quality_status = "error"`` while a valid
relevance prediction remains intact. There is no correction
request, quotation grounding gate, criterion-rule replay, role correction,
evidence expansion, repeated vote, ensemble or reconciliation. A structurally
valid decision is preserved even if it conflicts with the model's other fields.
Evidence quotations and block references remain unverified model claims.

Outputs and evaluation
----------------------

Each model writes:

.. code:: text

   interim/<source>/baselines/plain_llm/<model-slug>/classifications.csv
   interim/<source>/baselines/plain_llm/<model-slug>/classifications.audit.json

For reviews, ``<source>`` is ``cohort`` or ``cohort_<provider>``. Pilots add
``pilots/<limit>/`` below the model directory. A custom ``--output`` requires a
single configured model.

The audit stores the exact rendered prompt, response schema, raw model output,
model digest, scientific profile hashes, generation settings, original-evidence
fingerprint, token counts, completion reason, durations and errors. API keys and
authorization headers are excluded. Every attempted candidate remains in the
classification output, including errors. The existing evaluator handles those
errors as unavailable decisions and retains them in operational denominators.

``--resume`` explicitly reuses matching saved predictions, including recorded
failures, without retrying them. Changed weights, evidence, prompts or generation
settings invalidate reuse. Use a fresh run when intentionally repeating a failed
experiment. ``--evaluate-only`` verifies a completed CSV and its audit against
the active scientific/inference configuration, then reports it offline.

Catalogue evaluations appear under
``evaluation/<source>/classifiers/plain_llm/<model-slug>/``. Review evaluations
use ``plots/<cohort>/baselines/plain_llm/<model-slug>/evaluation/``. Comparisons
include separately labelled Plain LLM and RADAR model outputs; ensembles remain
separate. Review comparisons are under ``evaluation/<cohort>/comparison/``.
Reference labels are read only by evaluation and do not enter inference prompts.

Interpretation
--------------

Compare Plain LLM and RADAR using the same model and candidate cohort. The
primary contrast includes RADAR's later evidence acquisition, repeated calls
and scientific processing, so any measured difference describes their combined
effect and additional computation. It does not isolate reconciliation: these
comparisons use classification outputs before catalogue reconciliation.
A separate evidence-matched experiment would be needed to isolate processing
from evidence access. No such extra arm or measured improvement is implied by
adding this benchmark.
