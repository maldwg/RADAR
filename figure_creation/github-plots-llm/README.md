# GitHub IDS Repository Plots

Input CSV: `../data-gathering/ids_repos.llm.csv`
Rows: `267`

## Summary

- Total stars: `272,079`
- Median stars: `267`
- Total forks: `42,430`
- Active within one year: `45%`
- Deployable IDS share: `16%`
- Spearman correlation between log-stars and log-forks: `0.79`

## Figures

- `00_github_repo_overview.svg`
- `01_primary_family_counts.svg`
- `02_repository_role_counts.svg`
- `03_detection_type_counts.svg`
- `04_language_counts.svg`
- `05_family_role_heatmap.svg`
- `06_family_activity_heatmap.svg`
- `07_top_repositories_by_stars.svg`
- `08_stars_vs_forks.svg`
- `09_popularity_vs_activity.svg`
- `10_open_issues_vs_stars.svg`
- `11_activity_counts.svg`
- `12_role_confidence_counts.svg`
- `13_top_github_topics.svg`
- `14_role_popularity_distribution.svg`

## Notes

- Plots use only the compact accepted repository CSV, so they visualize what the crawler kept after filtering and deduplication.
- Activity buckets use `pushed_at` first and fall back to `updated_at`; archived repositories are separated regardless of timestamp.
- Detection type counts are multi-label, so totals can exceed the number of repositories.
- The output directory intentionally uses `github_plots` to avoid overwriting the older `github-plots` figures.
