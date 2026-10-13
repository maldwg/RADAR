# Memmesheimer verification pilots

`pilot_scoped/` contains an independent, bounded OpenAlex collection: at most
25 results from each of two adapted queries, then five retained candidates.
The collection audit preserves the exact profile used, raw API pages, query
ranks, caps and content hashes. Two candidates are submitted to the full-text
classifier as a small integration check, not a population performance study.

The model is `gemma4:31b`, already resident on the configured shared Ollama server.
The inference timeout is 90 seconds with no network retries; text is capped at
60,000 characters and truncation is recorded. Other inference settings come from
`config/default.toml`; their public values are saved in `inference_settings.json`.
The earlier request for `qwen3.6:27b` was stopped while waiting for the server and
produced no predictions.

`initial_classifications.*` preserves the first completed run, which rejected
both assessments under whitespace-only quote comparison. The first paper's
quoted wording matches after expanding PDF Latin ligatures such as `ﬀ` to `ff`.
`correction_only_classifications.*` preserves the intermediate run with a
correction request but without ligature normalization. The final
`classifications.*` run normalizes whitespace and Latin ligatures, uses one
bounded correction request and retains every response and validation error.
These results helped develop
the adapter, so they are development observations rather than an independent
test set.

`pilot/` is an earlier diagnostic search using general, stemmed search and
stable-ID selection. It returned substantial off-topic material. Its embedded
profile differs from the current search profile and it has no classifier
performance results. Use `pilot_scoped/` for the documented pilot.

The reference inventory is pinned to arXiv v1 of the review. It has eleven
positive references and no reviewed negatives, so false-positive performance
cannot be estimated. The collection date and metadata come from a later
OpenAlex snapshot; this is an adapted replication, not a reconstruction of the
review's original database state.
