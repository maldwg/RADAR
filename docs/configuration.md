# Study configuration

Select a study before running a stage:

```bash
ids-meta --study ids_repositories validate
ids-meta --study ids_repositories run --collect-limit 100 --classify-all
ids-meta --study ids_datasets baseline --methods keyword
ids-meta --study conformal_treatment_effects collect --collect-limit 200
ids-meta --study conformal_treatment_effects classify --classify-limit 5
```

`validate` reads the configuration, prompts, taxonomy, baselines and references
without making API or inference calls. `--config /path/to/study.toml` selects an
explicit manifest, including a study outside the checked-in configuration tree.
The default study is `ids_repositories`. Standalone reporting and verification
commands use their explicit inputs and do not require a study manifest.

## Ownership

```text
config/
  shared/
    runtime.toml                    # logging and backend connection
    models.toml                     # model names, revisions and inference settings
    sources/{github,openalex}.toml   # source identity and transport settings
  studies/
    ids_repositories/
      study.toml                    # entry point and explicit file references
      screening.toml                # collection, eligibility and reconciliation settings
      taxonomy.toml                 # system types and label descriptions
      search/github.toml            # discovery queries and scoring vocabulary
      prompts/classification.toml   # classification and reconciliation templates
      baselines/
        settings.toml               # method selection, thresholds and ensemble policy
        keyword.toml                # weighted taxonomy phrases
        nli.toml                    # taxonomy hypotheses
        llm_nli.toml                # three-way NLI prompt
      evaluation/
        references.toml             # evaluation-only reference labels
        github_ghml23.csv            # immutable upstream snapshot
      reporting.toml                # figure settings
    ids_datasets/                   # same ownership, with OpenAlex search
    conformal_treatment_effects/     # paper eligibility/quality protocol and extraction
```

Shared profiles contain no study search terms or reference labels. Scientific
settings belong to the study that uses them. To select different models or
transport settings for one study, reference another explicit profile from its
manifest. There is no implicit profile inheritance or override chain: declaring
the same execution setting in two referenced profiles is an error.

Classification label descriptions belong in `taxonomy.toml`, under
`[prompt_tables.<table>]` for catalogue workflows. Prompt templates reference
those tables rather than maintaining another copy of their descriptions.
The review taxonomy defines extraction descriptions and allowed values under
`[assessment_fields.<field>]`. Its screening file owns the eligibility and
quality criteria and explicitly references its taxonomy and prompt.

## Manifest

The manifest schema is `2.0`; the individual profile schemas remain `1.0`.
Every file or directory reference resolves relative to the file declaring it,
including the reference sets and `output_root`. No project-root or working-directory
heuristics are used.

The [repository manifest](../config/studies/ids_repositories/study.toml) selects
the shared runtime and models, local baseline/reporting settings, source profile,
scientific profiles and reference set. The
[dataset manifest](../config/studies/ids_datasets/study.toml) owns its separate
OpenAlex configuration. The
[review manifest](../config/studies/conformal_treatment_effects/study.toml) selects
a generic eligibility/quality screening engine.

| Workflow | Discovery source | Research outputs |
| --- | --- | --- |
| `repository_catalogue` | GitHub | Repository classifications and project catalogue |
| `dataset_catalogue` | OpenAlex | Dataset/generator release classifications and artifact catalogue |
| `systematic_review` | OpenAlex | Evidence-backed criterion judgments and configured extraction fields |

These are execution engines, not study-name branches. A new study can reuse an
engine under an arbitrary identifier. Source/workflow combinations are checked
before execution. Semantic Scholar currently supplies optional scholarly evidence
enrichment; adding it as a discovery source requires an adapter implementation.
Unknown APIs cannot be enabled merely by adding their name to a manifest.

## Adding a study

1. Copy the directory of the closest existing study.
2. Set a unique `id` and `output_root` in `study.toml`.
3. Edit the queries, taxonomy, screening rules, templates and baseline settings.
4. Replace the evaluation references independently of the scientific prompts.
5. Run `ids-meta --study <directory-name> validate`.

A review can define different eligibility rules, quality thresholds and extraction
fields without adding a named-study branch to the CLI. The repository and dataset
engines retain their specialized catalogue semantics; a fundamentally different
catalogue schema requires changes to that engine.

## Outputs and clean state

The checked-in manifests isolate outputs under `artifacts/studies/<study>/`:

```text
raw/<source>/cache.sqlite3
interim/<source>/                         # selected candidates
interim/<source>/models/<model>/          # classifications and reconciliation
interim/<source>/baselines/<method>/      # baseline results
processed/<source>/models/<model>/        # exported catalogues
plots/<source>/models/<model>/            # figures and their audit bundles
evaluation/<source>/classifiers/          # catalogue classifier evaluations
runs/                                    # stage manifests
```

The review workflow writes its candidate JSON under `interim/<source>/` and uses
the same model-specific classification paths. Its generic report contains its
evaluation and reporting database beneath the model report directory.

Use a clean output root for each independent experiment. `--resume` explicitly
continues compatible current-format work; it is disabled by default. Stage
manifests record the study identity, resolved configuration paths and profile
hashes. Reference labels remain outside classification prompts and decision
fingerprints.

The former flat configuration layout is removed. There are no old-path aliases,
automatic migrations or stale-report repairs.
