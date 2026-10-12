# OpenAlex external-reference evaluation

## Start here

- `shield`: open `comparison/00_metric_overview.svg` 
  (6 classifiers; exact values in `comparison/data/00_metric_overview.csv`).
  Horizontal recovery overview: `comparison/01_reference_recovery.svg`.
  Compact taxonomy comparison: `comparison/taxonomy/04_taxonomy_comparison.svg`.
  Taxonomy accuracy among recovered items: `comparison/taxonomy/05_recovered_taxonomy_comparison.svg`.

## Directory guide

- `comparison/`: cross-classifier reporting, split by reference-set name when several are configured.
- `classifiers/`: detailed audits for one classifier, likewise split by reference set when needed.
- `classifiers/.../figures/06_taxonomy_accuracy`: horizontal exact field-accuracy overview with percentage labels.
- `classifiers/.../figures/07_recovered_taxonomy_accuracy`: field accuracy among recovered reference positives, with field-rated sample counts and recovery totals. No recovered comparisons means N/A; compare alongside recovery because approaches can recover different subsets.
- `classifiers/nli/<model>/sensitivity/`: offline NLI cutoff robustness; not another model result.
- Matching `.svg` and `.pdf` files are the same figure in screen and publication formats.

## Available classifier details

- `Keyword` → `classifiers/keyword/` (reference sets: shield)
- `NLI - qwen3-8-27b` → `classifiers/llm_nli/qwen3-8-27b/` (reference sets: shield)
- `NLI · moritzlaurer-deberta-v3-base-mnli-fever-anli` → `classifiers/nli/moritzlaurer-deberta-v3-base-mnli-fever-anli/` (reference sets: shield)
- `LLM · gemma4-31b` → `classifiers/llm/gemma4-31b/` (reference sets: shield)
- `LLM · qwen3-8-27b` → `classifiers/llm/qwen3-8-27b/` (reference sets: shield)
- `Ensemble` → `classifiers/ensemble/` (reference sets: shield)
