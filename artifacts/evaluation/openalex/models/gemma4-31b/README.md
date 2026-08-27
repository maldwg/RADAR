# Classification evaluation

- Source: `openalex`
- Rated entities: 24
- Operational DR: 0.8333
- Operational FPR: 0.0000
- Operational accuracy: 0.9167
- Operational precision: 1.0000
- Metadata field accuracy: 0.8512 (168 comparisons)
- Abstention rate: 0.3750

## Interpreting the three views

- **Operational:** asks what the complete pipeline delivers. Every non-inclusion predicts negative.
- **Strict:** asks how performance looks when every unresolved boundary result is wrong.
- **Selective:** measures only definite include/exclude decisions. Always read it with definitive coverage and abstention rate.

Example: if exclusion was expected but the result is uncertain, operational counts a true negative, strict counts a false positive, and selective omits the entity. If inclusion was expected, the same unresolved result is a false negative operationally and strictly, and is omitted selectively.

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
