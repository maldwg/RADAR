# Legacy artifacts

These files are the unchanged outputs of the pre-refactor scripts. They are preserved only for provenance and must not be interpreted as schema `1.0` pipeline outputs.

## GitHub

| Previous root path | Legacy path |
|---|---|
| `ids_repos.csv` | `github/ids_repos.csv` |
| `ids_repos.evaluated.csv` | `github/ids_repos.evaluated.csv` |
| `ids_repos.partial.csv` | `github/ids_repos.partial.csv` |
| `ids_repos.llm.csv` | `github/ids_repos.llm.csv` |
| `ids_repos.llm_accepted.csv` | `github/ids_repos.llm_accepted.csv` |
| `ids_repos.llm_classified.csv` | `github/ids_repos.llm_classified.csv` |

## OpenAlex

Every former root file beginning with `ids_datasets_openalex` is preserved under `openalex/` with the same filename, including the checkpoint, entity/candidate audits, validation files, and prior LLM exports.

The old LLM OpenAlex catalogue contains accepted entity decisions without resolved paper metadata for many rows. It is retained for audit only and is not migrated into `data/processed/openalex/base_papers.csv`.

