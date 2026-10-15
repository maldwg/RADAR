.. _llm_nli_plan-plan-llm-based-nli-through-ollama:

Plan: LLM-based NLI through Ollama
==================================

Status: the Ollama NLI backend, opt-in configuration, both source commands, isolated pilots,
auditable pair caching, offline rescoring and evaluation/plot integration are implemented.
Network-free tests cover both sources. On 2026-10-08, the live synthetic check passed against Ollama
0.34.0 and ``qwen3.8:27b``: all three class alternatives were available, and the three examples
produced the expected entailment, neutral and contradiction answers. This check used synthetic
evidence; it does not establish performance on the candidate corpus.
The additional equal-evidence structured-Qwen experiment below is an experimental protocol, not an
automatically generated comparison arm. See
:ref:`usage and score semantics <evaluation-ollama-nli>`.

.. _llm_nli_plan-objective:

Objective
---------

Add a separate ``llm_nli`` baseline for GitHub and OpenAlex. Start with the same ``qwen3.8:27b``
model used by the structured LLM approach, with other Ollama models configurable. Compare keyword
classification, DeBERTa NLI, Qwen NLI, structured Qwen/Gemma classification, and the existing
ensemble in the evaluation figures.

The new method asks whether each existing taxonomy hypothesis follows from a premise, instead of
requesting a joint classification and evidence JSON. This is a generative LLM performing NLI; it is
not a model fine-tuned with an NLI head.

.. _llm_nli_plan-establish-the-scoring-capability-first:

1. Establish the scoring capability first
-----------------------------------------

Current official Ollama documentation exposes ``logprobs`` and ``top_logprobs`` on both
``/api/generate`` and ``/api/chat``. These provide output-token log probabilities and a limited set
of alternatives, not a complete raw vocabulary-logit vector:

- https://docs.ollama.com/api/generate
- https://docs.ollama.com/api/chat
- https://github.com/ollama/ollama/blob/main/api/types.go

The configured server could not be reached during planning. The implementation subsequently verified
score support on the live server, with version, model digest and raw synthetic scores in
``artifacts/runs/2026-10-08-154ee043ce.json``. Every inference run repeats the check; do not assume
another installed server implements the same documented API.

The implementation should first read ``/api/version``, ``/api/tags``, and model metadata, then test
a few synthetic entailment, neutral, and contradiction pairs. Verify that thinking can be disabled
and that all three class-token alternatives are returned at the same decision position, with finite
usable log probabilities. Inspect actual tokenization, leading whitespace, output length, and
probability behavior under the chosen sampling settings. Fix the answer template accordingly.

Use the native API initially, with ``stream=false``, ``think=false``, deterministic generation, and
``logprobs=true``. Request enough alternatives within the server’s supported limit. Avoid
JSON/schema constraints and other generation restrictions that might distort the class scores.
Confirm score semantics in the supported server version rather than assuming that sampled or
grammar-masked probabilities are suitable for classification.

Missing scores are a capability/scoring error, not a probability of zero and not a semantic neutral
decision. If a class token falls outside the returned alternatives, exact three-class scoring is
unavailable for that pair. Do not silently approximate it. The capability pilot is a gate before a
full run; if this limitation is frequent, reliable scoring will require an API extension or another
serving backend rather than an unsupported Ollama implementation.

.. _llm_nli_plan-define-the-nli-task-and-score:

2. Define the NLI task and score
--------------------------------

Reuse the hypotheses in ``config/studies/ids_repositories/baselines/nli.toml`` and
``config/studies/ids_datasets/baselines/nli.toml``. A separate versioned prompt should state:

   Given only the premise, decide whether the hypothesis is entailed, contradicted, or neither
   established nor contradicted. Missing information is neutral. Return exactly one class code: A,
   B, or C.

Map A/B/C to entailment/neutral/contradiction only after the capability test confirms suitable
single-token codes for the selected model and template. Check label-order/verbalizer sensitivity on
pilot examples, then freeze the mapping. Do not request model-written confidence percentages.

Given log probabilities ``l_E``, ``l_N``, and ``l_C`` at that same position, calculate:

.. code:: text

   p(class | one of the three class codes) = exp(l_class) / sum(exp(l_E), exp(l_N), exp(l_C))

Use stable log-sum-exp. This conditional score is equivalent to normalizing the corresponding
logits; the full vocabulary logits are unnecessary if all required class scores are available. Also
preserve the unnormalized class probability mass and raw alternatives, since normalization can hide
poor adherence to the answer format. These scores are not calibrated probabilities of correctness.
Any calibration or additional low-mass rejection rule must use separate development data and be
frozen before reference-set evaluation.

.. _llm_nli_plan-make-the-comparison-interpretable:

3. Make the comparison interpretable
------------------------------------

Freeze candidate inputs and source-labelled evidence chunks. Qwen NLI and DeBERTa NLI must see the
same selected premise texts and hypotheses. Reuse the reference chunking/tokenizer policy, without
loading the DeBERTa classifier merely to split text. Do not permit model-specific evidence retrieval
in this baseline.

Reuse the existing NLI aggregation and decision rules: at most six selected chunks, strongest-two
aggregation, and initially entailment threshold ``0.50`` with margin ``0.10``. Preserve their exact
implementation, including exclusive, multilabel, and boolean field behavior. These thresholds
describe the initial comparison, not an assumption of equal calibration across models.

Score one premise/hypothesis pair per task, returning an ``NLIScore``. Neutral support or failure to
resolve the role leads to uncertainty under the existing adapter; a confidently excluded role leads
to exclusion. Do not add the LLM pipeline’s quote gates, retries for more evidence, or repeated-run
consensus to the NLI baseline.

Report two comparisons:

1. The operational comparison of all existing approaches on the frozen candidate and reference sets,
   with their respective evidence budgets documented.

2. A controlled Qwen comparison where NLI and structured classification receive the same evidence
   budget. This separates the task formulation from the effects of longer context and adaptive
   enrichment. Report repeated-run consensus separately from a single structured classification.

Before interpreting correctness, audit reference matching and report how many reference items
actually have available classifications. The current GitHub evaluation audits contain no matched
attempts; changing the inference method cannot by itself resolve retrieval or mapping gaps. Verify
the identifiers and candidate/reference overlap before presenting a GitHub classifier-quality
comparison, and keep missing-reference coverage distinct from uncertainty.

.. _llm_nli_plan-integrate-without-changing-existing-results:

4. Integrate without changing existing results
----------------------------------------------

- Add an Ollama NLI backend implementing the existing ``NLIBackend`` scoring contract; factor shared
  inference/aggregation code out of method-specific loading where needed.

- Add ``llm_nli`` as a distinct baseline method and a disabled-by-default ``[baselines.llm_nli]``
  configuration with a list of models, scoring options, threshold settings. Requests currently
  execute sequentially (concurrency one). Keep the existing ``nli`` configuration and cached outputs
  compatible.

- Support proposed commands such as ``radar --study ids_repositories baseline --methods llm_nli``
  and the OpenAlex equivalent, plus offline ``--evaluate-only`` reuse.

- Write classifications under ``data/interim/<source>/baselines/llm_nli/<model-slug>/`` and
  evaluations under ``artifacts/evaluation/<source>/classifiers/llm_nli/<model-slug>/``.

- Extend classifier discovery, labels, and plotting to show ``NLI - Qwen`` alongside
  ``NLI - deberta-v3`` and structured LLM approaches, without combining their rows. Generate
  recovery, taxonomy, decision, uncertainty, and threshold-sensitivity plots through the existing
  evaluation workflows.

- Cache pair scores and completed rows independently. Include the model digest, server version,
  prompt/profile hashes, chunk policy, score extraction and aggregation versions, and relevant
  inference settings in provenance/cache keys. Threshold-only changes should be rescored offline
  from stored scores.

- Keep raw alternatives, class mapping, conditional scores, class probability mass, supporting chunk
  references, errors, and inference timing auditable. An API failure remains an error, not an
  uncertain or excluded classification.

.. _llm_nli_plan-bound-cost-and-validate-before-scaling:

5. Bound cost and validate before scaling
-----------------------------------------

The existing profiles contain 52 GitHub hypotheses and 37 OpenAlex hypotheses. At six chunks, the
upper bounds are 312 and 222 pair requests per entity. For 2,000 GitHub and 1,000 OpenAlex entities,
that is up to 624,000 and 222,000 pairs. Ollama requests are not automatically a DeBERTa-style
tensor batch.

Start with synthetic capability checks, then a small role-only development pilot to measure score
completeness, label bias, latency, and prefix-cache behavior. Estimate the full taxonomy run from
that pilot before scaling. Use bounded concurrency, resumable pair caching, and model keep-alive. Do
not tune on the reported reference benchmark.

Tests should cover logprob parsing, class mapping and normalization, missing or duplicate class
tokens, thinking/whitespace handling, error separation, cache invalidation, chunk equality, and
identical shared NLI decisions for identical scores. Fake backends should keep routine tests
network-free. A small real-server integration check validates support; full runs should not be
required by tests.

.. _llm_nli_plan-completion-criteria:

Completion criteria
-------------------

Both sources produce independently cached and evaluated Qwen NLI outputs, and all comparison plots
distinguish them from DeBERTa NLI and structured Qwen. Each scored pair has three auditable class
scores or an explicit scoring error. Existing classifier decisions and production catalogues are
unaffected. Results report evidence budgets, unresolved/error coverage, and observed inference cost.
