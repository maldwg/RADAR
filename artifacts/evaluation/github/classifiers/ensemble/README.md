# Classification evaluation

- Reference set: GHML23 repository reference set with IDS-Meta taxonomy crosswalk
- Source: `github`
- Evaluation unit: `source_record`
- Rated entities: 120
- Operational DR: 0.0000
- Operational FPR: 0.0000
- Operational FDR: 0.0000
- Operational accuracy: 0.4034
- Operational precision: 0.0000
- Metadata field accuracy: 0.0666 (856 comparisons)
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
- `figures/06_taxonomy_accuracy.svg`
- `figures/07_recovered_taxonomy_accuracy.svg`
- `figures/08_reference_uncertainty.svg`

Each listed SVG has a matching PDF for publication; they are two formats of the same figure. These are single-classifier diagnostics. Use the source-level `comparison/` directory for cross-classifier reporting. Companion aggregate tables are stored under `figures/data/`.

`06_taxonomy_accuracy` is a horizontal overview of exact field accuracy with percentage labels. Each field uses its exported rated comparison count, retaining unresolved results; unsupported or empty comparisons are N/A. `03_taxonomy_fields` additionally reports multi-label F1.

`07_recovered_taxonomy_accuracy` evaluates taxonomy fields only among expected relevant reference items matched to accepted outputs (`accept` or `accept_adjacent`). Its denominator excludes missed references and negative controls. Each field displays its own rated recovered count, and the axis reports recovery against all reference positives. Missing field predictions count as incorrect; multi-label fields require an exact set match. N/A means an unsupported field or no recovered field comparisons. Read this conditional quality alongside reference recovery: different classifiers may recover different subsets. Matching and field checks reuse `instances.csv`, including the OpenAlex cited-paper fallback. Older bundles without that audit cannot render this view.

`08_reference_uncertainty` shows uncertain decisions divided by attempted classifications of rated reference items. Accept and exclude are definitive outcomes. Errors and unavailable outputs remain in the attempted denominator but are counted separately in the CSV; references without a classification attempt are excluded. A matched artifact classification takes priority over the OpenAlex cited-paper fallback, so each reference entity contributes once. N/A means zero attempts. This differs from the legacy abstention metric and from end-to-end unresolved artifact coverage.
