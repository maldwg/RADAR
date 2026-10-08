# Classification evaluation

- Reference set: Ashraf and Masoodi (2026) Table 2 dataset reference set
- Source: `openalex`
- Evaluation unit: `artifact`
- Rated entities: 105
- Dataset recall: 0.3168
- Dataset miss rate: 0.6832
- Metadata field accuracy: 0.3386 (505 comparisons)
- Mean taxonomy-field macro-F1: 0.3016
- Classifier abstention rate: 0.0286
- Unresolved artifact rate: 0.5644
- Reference positives/negatives: 101/4

Exact artifact names and aliases are matched first. An accepted cited paper is a fallback only for positive artifacts; cited-paper outcomes for negative artifacts remain diagnostics, since a citation may be an analysis or documentation paper. Metadata accuracy is end-to-end across the rated inventory.

This fixture includes a small, selected set of negative controls. FPR and related rates are calculable for these controls, but do not estimate corpus-wide specificity or prevalence. An unmatched negative is operationally correct but counts as an error under the strict view.

## Interpreting the three views

- **Operational:** complete-inventory decisions; unresolved positives are misses and unresolved negatives are non-inclusions.
- **Strict:** unresolved negatives are counted as errors.
- **Selective:** only definitive artifact decisions; read with coverage and abstention.

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
