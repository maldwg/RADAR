# GitHub classifier comparison

## Start here

Open `00_metric_overview.svg`. Rows are metrics and columns are classifiers. The color is the raw rate, so use the arrow on each row to see whether higher or lower is preferable. Grey N/A cells mean the benchmark cannot estimate that metric.

## Drill-down directories

- `metrics/`: one enlarged operational-metric comparison per figure.
- `taxonomy/`: field and individual-category comparisons.
- `agreement/`: where classifiers make the same decision; agreement is not correctness.
- `data/00_metric_overview.csv`: exact values behind the overview.

Matching SVG and PDF files are the same figure in screen and publication formats.

Single-classifier operational/strict/selective views, confusion matrices, and coverage diagnostics are under the sibling `classifiers/` directory. Those diagnose one output; the figures here compare approaches on the common rated set.
