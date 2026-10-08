# GitHub classifier comparison

## Start here

Open `00_metric_overview.svg`. Rows are metrics and columns are classifiers. The color is the raw rate, so use the arrow on each row to see whether higher or lower is preferable. Grey N/A cells mean the benchmark cannot estimate that metric.

Open `01_reference_recovery.svg` for a horizontal overview of recovered positive reference entities by classifier, with percentage labels. Exact recovered/missed counts, denominators and confidence intervals are in `data/01_reference_recovery.csv`. The sibling classifier `figures/06_taxonomy_accuracy.svg` charts show exact field accuracy as horizontal percentage bars.

Open `02_reference_uncertainty.svg` for uncertain decisions divided by attempted classifications of rated reference items. Each bar shows its uncertain/attempted counts; retrieval gaps and unclassified references are excluded. Errors remain in the denominator and are reported separately in `data/02_reference_uncertainty.csv`. OpenAlex uses a classified cited paper if no artifact match exists. Each reference entity contributes once, and N/A means no attempts. This is distinct from the legacy abstention metric.

## Drill-down directories

- `metrics/`: one enlarged operational-metric comparison per figure.
- `taxonomy/`: field and individual-category comparisons. Start with `04_taxonomy_comparison.svg` for compact grouped horizontal bars comparing every approach within each field. Colours identify individual classifiers; percentages and rated denominators are printed, with exact values in its companion CSV.
  `05_recovered_taxonomy_comparison.svg` compares field accuracy only among each approach's recovered reference positives. Bar annotations show field-rated sample counts; the legend shows recovered/expected totals. Negative controls and missed references are excluded, while missing field predictions are incorrect. Multi-label fields require an exact set match. N/A means unsupported fields, no recovered comparisons, or a missing entity audit. Different approaches can recover different subsets, so read with recovery. The per-classifier equivalent is `figures/07_recovered_taxonomy_accuracy.svg`.
- `agreement/`: where classifiers make the same decision; agreement is not correctness.
  `01_model_decision_counts` uses horizontal stacked shares of the full source union, with record counts inside sufficiently wide segments. Accept combines accept and accept_adjacent; Exclude means exclude. Errors and missing results stay visible. Exact decisions remain in `data/01_model_decision_counts.csv`; grouped counts and percentage denominators are in `data/01_model_decision_shares.csv`.
- `data/00_metric_overview.csv`: exact values behind the overview.

Matching SVG and PDF files are the same figure in screen and publication formats.

Single-classifier operational/strict/selective views, confusion matrices, and coverage diagnostics are under the sibling `classifiers/` directory. Those diagnose one output; the figures here compare approaches on the common rated set.
