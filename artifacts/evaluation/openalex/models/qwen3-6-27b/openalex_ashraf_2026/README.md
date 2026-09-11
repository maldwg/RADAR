# Classification evaluation

- Source: `openalex`
- Evaluation unit: `artifact`
- Rated entities: 123
- Dataset recall: 0.2764
- Dataset miss rate: 0.7236
- Metadata field accuracy: 0.2179 (615 comparisons)
- Abstention rate: 0.6504

This artifact fixture is a positive reference set. Metadata accuracy is end-to-end across the rated inventory, so an unidentified dataset also misses its expected metadata. False-positive rate, specificity, and balanced accuracy are not estimable because the profile contains no negative artifacts. Candidate-level retrieval coverage is omitted because raw paper candidates do not yet have classified artifact names.

## Interpreting the three views

- **Operational:** dataset recall across the complete positive reference set.
- **Strict:** identical here because every unresolved positive is a miss.
- **Selective:** recall only among datasets with definitive artifact decisions; read it with coverage and abstention.

## Figures

- `01_decision_performance.svg`
- `01_decision_performance.pdf`
- `02_decision_confusion.svg`
- `02_decision_confusion.pdf`
- `03_field_accuracy.svg`
- `03_field_accuracy.pdf`
- `04_coverage_and_status.svg`
- `04_coverage_and_status.pdf`

Companion aggregate tables are stored under `plot_data/`. Metadata accuracy is shown separately from binary inclusion/exclusion metrics.
