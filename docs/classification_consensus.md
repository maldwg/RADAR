# Classification consensus

Consecutive classification runs combine both the overall decision and normalized taxonomy fields. This applies to repeated runs of one configured model. The separate cross-model ensemble remains a decision-disposition ensemble.

## Independence of runs

Every run is a separate generation request with the same evidence and prompt. Earlier responses are never inserted into later prompts, so there is no conversational history between runs. The model process is not explicitly restarted and the configured seed is not changed. With temperature zero and a fixed seed, repeated responses may therefore be identical. Use multiple models or a deliberately stochastic inference configuration when independent diversity is required.

## Voting

All configured runs remain in every denominator, including errors and unavailable results.

The decision first maps `accept` and `accept_adjacent` to inclusion, while exclusion and uncertainty remain distinct. One disposition must uniquely reach `consensus_threshold`; ties and values below the threshold become uncertain.

Normalized fields are then voted:

- OpenAlex single-value fields: paper role, relevance tier, artifact type, domain, data form, capture environment, labeling, direct-release evidence, and verified-evidence status.
- GitHub single-value fields: role, fork independence, direct self-description, runtime observation, runnable-detector status, offline-dataset-only status, upstream repository, and verified-evidence status.
- GitHub multi-label fields: every observation scope, observation object, capability, detection method, and system type is voted independently. A label is retained only when it reaches the threshold.
- GitHub primary system type is voted as a single value and must also occur in the selected `system_types`.
- Names, reasons, and evidence quotations are not merged. They remain attached to the representative coherent run.

A missing or below-threshold OpenAlex taxonomy value becomes `unknown` or `uncertain`. A below-threshold GitHub label is omitted; scopes and methods fall back to `unknown` when no concrete label survives. Eligibility-sensitive booleans conservatively fall back to false.

## Valid combinations and deterministic checks

A valid combination is internally consistent with the taxonomy. Validation is deterministic Python code and never asks another model.

OpenAlex checks include:

- Original and versioned releases must describe a `dataset` or `dataset_and_generator`.
- Generator releases must describe a `generator_framework` or `dataset_and_generator`.
- Accepted releases require an accepted tier, direct release evidence, a resource name, and verified evidence when configured.
- Generator releases cannot use the adjacent tier.
- Survey, downstream-application, dataset-analysis, and unrelated roles resolve to exclusion when their evidence is verified.
- The fields must imply the same include/exclude disposition as the decision vote.

GitHub checks include:

- A standalone detector requires runtime observation and a runnable detector and cannot be offline-dataset-only.
- An included role requires direct self-description and verified evidence when configured.
- A fork requires independent-significance evidence.
- An out-of-scope role resolves to exclusion when its evidence is verified.
- The fields must imply the same disposition as the decision vote.
- System types are direct model outputs. Validation checks that every code belongs to the configured vocabulary and that the primary type also occurs in `system_types`; it does not infer or overwrite a type from other dimensions.

## Contradictions and fallback

Independent field majorities can form a tuple that no run produced. If that tuple fails validation but the overall decision has a valid include or exclude consensus, the pipeline falls back to the strongest representative run supporting that disposition. This preserves a complete coherent classification instead of manufacturing a contradictory mixture. The proposed field votes remain available in the audit JSON.

If no coherent supporting run exists, or the overall disposition itself is uncertain, the final decision remains `uncertain`.

## Audit columns

- `ensemble_model_decisions`: every consecutive-run decision.
- `ensemble_*_score`, `ensemble_agreement`, and `ensemble_vote_status`: disposition vote details.
- `consensus_field_audit`: selected field values, per-value or per-label counts, agreement, and vote status.
- `consensus_validation_flags`: failed consistency rules and whether a coherent-run fallback was used.
- `consensus_fallback_run`: the run selected when field voting produced an invalid combination.

Changing `consecutive_runs`, `consensus_threshold`, prompt configuration, model, or taxonomy definitions changes the decision fingerprint and prevents incompatible cache reuse.
