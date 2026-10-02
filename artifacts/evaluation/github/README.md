# GitHub external-reference evaluation

## Start here

- `ghml23`: open `comparison/00_metric_overview.svg`
  (2 classifiers; exact values in `comparison/data/00_metric_overview.csv`).

## Directory guide

- `comparison/`: cross-classifier reporting, split by reference-set name when several are configured.
- `classifiers/`: detailed audits for one classifier, likewise split by reference set when needed.
- `classifiers/nli/<model>/sensitivity/`: offline NLI cutoff robustness; not another model result.
- Matching `.svg` and `.pdf` files are the same figure in screen and publication formats.

## Available classifier details

- `LLM · gemma4-31b` → `classifiers/llm/gemma4-31b/` (reference sets: ghml23)
- `LLM · qwen3-8-27b` → `classifiers/llm/qwen3-8-27b/` (reference sets: ghml23)
