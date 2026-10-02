# Classification evaluation

- Reference set: GHML23 repository reference set with IDS-Meta taxonomy crosswalk
- Source: `github`
- Evaluation unit: `source_record`
- Rated entities: 120
- Operational DR: 0.0000
- Operational FPR: 0.0000
- Operational FDR: 0.0000
- Operational accuracy: 0.3983
- Operational precision: 0.0000
- Metadata field accuracy: 0.0655 (855 comparisons)
- Mean taxonomy-field macro-F1: 0.0000
- Abstention rate: 1.0000

## Interpreting the three views

- **Operational:** asks what the complete pipeline delivers. Every non-inclusion predicts negative.
- **Strict:** asks how performance looks when every unresolved boundary result is wrong.
- **Selective:** measures only definite include/exclude decisions. Always read it with definitive coverage and abstention rate.

Example: if exclusion was expected but the result is uncertain, operational counts a true negative, strict counts a false positive, and selective omits the entity. If inclusion was expected, the same unresolved result is a false negative operationally and strictly, and is omitted selectively.

## Figures

- `figures/01_decision_views.svg`
- `figures/02_decision_confusion.svg`
- `figures/03_taxonomy_fields.svg`
- `figures/04_taxonomy_categories.svg`
- `figures/05_pipeline_coverage.svg`

Each listed SVG has a matching PDF for publication; they are two formats of the same figure. These are single-classifier diagnostics. Use the source-level `comparison/` directory for cross-classifier reporting. Companion aggregate tables are stored under `figures/data/`.
