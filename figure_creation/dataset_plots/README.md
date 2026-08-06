# Dataset Plot Summary

Input: `/home/max/Desktop/Promotion/Doktorarbeit/data-gathering/ids_datasets_openalex.csv`
Load mode: `accepted_csv`
Rows: 30
Rows with canonical score: 22

## Heuristic-vs-citation diagnostics

- Pearson r(log10(citations+1), canonical_score): `0.390`
- Spearman rho(log10(citations+1), canonical_score): `0.496`

- Score/citation plots omit `8` row(s) without `canonical_score`: KDD, CICIDS2017, CSE-CIC-IDS2018, CAIDA, Edge-IIoTset, ADFA, CICFlowmeter, ICISSP

Interpretation hint: a strong positive correlation means the heuristic largely follows citation count; a weak correlation means the heuristic is using other signals. The residual plot highlights entries that score unusually high or low for their citation count.

## Generated Figures

- `00_dataset_overview.svg`
- `01_dataset_domain_counts.svg`
- `02_dataset_artifact_counts.svg`
- `03_dataset_timeline_by_domain.svg`
- `04_dataset_publications_by_year.svg`
- `05_dataset_domain_artifact_heatmap.svg`
- `06_dataset_domain_dataform_heatmap.svg`
- `07_dataset_temporal_environment_heatmap.svg`
- `08_dataset_confidence_artifact_heatmap.svg`
- `09_heuristic_score_vs_citations.svg`
- `10_heuristic_score_citation_bins.svg`
- `11_heuristic_score_citation_residuals.svg`
- `12_discovery_method_score_citations.svg`
- `13_discovery_method_score_distribution.svg`
- `14_dataset_metadata_completeness.svg`
