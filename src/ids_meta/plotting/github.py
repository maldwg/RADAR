from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd

from ids_meta.plotting.common import COLORS, add_heading, configure_style, count_legend, count_plot, explode_values, load_frame, metadata_caption, save_figure, stamp, write_index, write_table
from ids_meta.common.logging import logger
from ids_meta.github.schema import DEFAULT_SYSTEM_TYPE_PROFILE

LOG = logger("plotting.github")

ROLE_LABELS = {
    "standalone_detector": "Standalone detector", "platform_framework": "Platform/framework",
    "classifier_model": "Classifier/model", "rules_content": "Rules/content",
    "support_tooling": "Support tooling", "research_demo": "Research demo",
}
ROLE_MATRIX = [
    ("standalone_detector", "Det.", "Standalone detector", "#2A9D8F"),
    ("platform_framework", "Plat.", "Platform/framework", "#7B5EA7"),
    ("classifier_model", "Class.", "Classifier/model", "#D1842F"),
    ("rules_content", "Rules", "Rules/content", "#2F6F9F"),
    ("support_tooling", "Supp.", "Support tooling", "#6F7782"),
    ("research_demo", "Demo", "Research demo", "#9A7D0A"),
]
FAMILY_COLORS = ["#2F6F9F", "#2A9D8F", "#D1842F", "#7B5EA7", "#C94F4F", "#5B8F3A", "#6F7782"]


def _activity(value: str) -> str:
    """Map a repository timestamp to a human-readable activity-age band."""
    try:
        pushed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "unknown"
    days = (datetime.now(timezone.utc) - pushed).days
    return "≤90 days" if days <= 90 else "≤1 year" if days <= 365 else "1–3 years" if days <= 1095 else "3–5 years" if days <= 1825 else ">5 years"


def _primary_family(row: pd.Series) -> str:
    """Choose the most informative normalized family label for a repository."""
    primary = str(row.get("primary_system_type", "")).strip()
    if primary and primary != "unknown":
        return primary
    labels = [value for value in str(row.get("system_types", "") or row.get("derived_labels", "")).split(";") if value and value != "unknown"]
    if labels:
        return labels[0]
    scopes = [value for value in str(row.get("observation_scopes", "")).split(";") if value and value != "unknown"]
    return (scopes[0].replace("_", " ").title() + " scope") if scopes else ROLE_LABELS.get(str(row.get("role", "")), "Other")


def _short(value: object, length: int) -> str:
    """Normalize separators and truncate display text to a fixed length."""
    text = str(value or "").replace("_", "-").replace(";", "; ")
    return text if len(text) <= length else text[:length - 1] + "…"


def _compact_number(value: float) -> str:
    """Format a numeric count with compact thousands or millions suffixes."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(int(value))


def _overview_matrix(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """One scrollable feature matrix inspired by the prior thesis overview, using the normalized taxonomy."""
    view = frame.copy()
    exact_commit = view.get("last_commit_at", pd.Series("", index=view.index)).fillna("")
    view["overview_commit_at"] = exact_commit.where(exact_commit != "", view["pushed_at"])
    view["overview_family"] = view.apply(_primary_family, axis=1)
    view["stars_numeric"] = pd.to_numeric(view["stars"], errors="coerce").fillna(0)
    view["forks_numeric"] = pd.to_numeric(view["forks"], errors="coerce").fillna(0)
    family_order = list(dict.fromkeys(view["overview_family"]))
    view["family_order"] = view["overview_family"].map({value: index for index, value in enumerate(family_order)})
    view["variant_order"] = (view["repository_relationship"] != "canonical").astype(int)
    view = view.sort_values(["family_order", "variant_order", "role", "stars_numeric", "full_name"], ascending=[True, True, True, False, True], kind="stable")

    group_count = view["overview_family"].nunique() if not view.empty else 1
    total_units = 4.2 + len(view) * .78 + group_count * .92 + 2.1
    fig, ax = plt.subplots(figsize=(18, max(7, total_units * .34)))
    ax.set_xlim(0, 100); ax.set_ylim(total_units, 0); ax.axis("off")
    ax.text(50, .55, "Repository feature matrix", ha="center", va="center", fontsize=17, fontweight="bold", color="#263238")
    ax.text(50, 1.18, "Every accepted repository; grouped by normalized system family; role dots distinguish detectors, frameworks, models, content, tooling, and demos", ha="center", va="center", fontsize=8.5, color="#6F7782")

    headers = [(1.3, "Repository"), (20.5, "System"), (30.5, "Detection method(s)"), (68.5, "Activity"), (75, "Stars"), (84.2, "Forks"), (91.2, "Last commit")]
    for x, label in headers:
        ax.text(x, 2.25, label, fontsize=8, color="#59636F", va="center")
    role_x = [51.2, 54.2, 57.2, 60.2, 63.2, 66.2]
    for x, (_role, abbreviation, full, _color) in zip(role_x, ROLE_MATRIX):
        ax.text(x, 2.25, abbreviation, fontsize=7.2, color="#59636F", ha="center", va="center")
        ax.text(x, 2.58, full, fontsize=.01, alpha=0)  # Retains the semantic label in vector output.
    ax.plot([.8, 99.2], [2.75, 2.75], color="#D8DEE6", linewidth=.8)

    max_stars = max(1.0, float(view["stars_numeric"].max() if not view.empty else 1))
    max_forks = max(1.0, float(view["forks_numeric"].max() if not view.empty else 1))
    activity_colors = {"≤90 days": "#2A9D8F", "≤1 year": "#5B8F3A", "1–3 years": "#D1842F", "3–5 years": "#C94F4F", ">5 years": "#8E3B46", "unknown": "#6F7782"}
    activity_labels = {"≤90 days": "≤90d", "≤1 year": "≤1y", "1–3 years": "1–3y", "3–5 years": "3–5y", ">5 years": ">5y", "unknown": "?"}
    family_colors = {family: FAMILY_COLORS[index % len(FAMILY_COLORS)] for index, family in enumerate(family_order)}
    cursor = 3.05
    row_index = 0
    for family, group in view.groupby("overview_family", sort=False):
        color = family_colors.get(str(family), FAMILY_COLORS[0])
        ax.add_patch(FancyBboxPatch((.8, cursor), 98.4, .68, boxstyle="round,pad=.02,rounding_size=.16", facecolor=color, alpha=.16, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((.8, cursor), .65, .68, boxstyle="round,pad=.02,rounding_size=.16", facecolor=color, edgecolor="none"))
        ax.text(1.8, cursor + .35, str(family), fontsize=8.5, fontweight="bold", color="#263238", va="center")
        ax.text(15.5, cursor + .35, f"{len(group)} repositories", fontsize=7.2, color="#59636F", va="center")
        cursor += .85
        for _, row in group.iterrows():
            if row_index % 2 == 0:
                ax.add_patch(Rectangle((.8, cursor), 98.4, .72, facecolor="#F7F9FA", edgecolor="none"))
            canonical = row.get("repository_relationship") == "canonical"
            ax.scatter([1.15], [cursor + .36], s=13, marker="o", facecolor="#2F6F9F" if canonical else "white", edgecolor="#2F6F9F" if canonical else "#9AA5AE", linewidth=.8, zorder=3)
            ax.text(1.55, cursor + .36, _short(row.get("full_name", ""), 34), fontsize=7.2, color="#263238", va="center")
            system = str(row.get("system_types", "") or row.get("derived_labels", "")) or str(row.get("observation_scopes", ""))
            ax.text(20.5, cursor + .36, _short(system, 18), fontsize=7, color="#43505C", va="center")
            ax.text(30.5, cursor + .36, _short(row.get("detection_methods", ""), 34), fontsize=7, color="#43505C", va="center")
            for x, (role, _abbreviation, _full, role_color) in zip(role_x, ROLE_MATRIX):
                selected = row.get("role") == role
                ax.scatter([x], [cursor + .36], s=18, marker="o", facecolor=role_color if selected else "white", edgecolor=role_color if selected else "#CDD5DB", linewidth=.75, zorder=3)
            activity = "unknown" if row.get("archived") == "true" else _activity(str(row.get("overview_commit_at", "")))
            activity_text = "arch." if row.get("archived") == "true" else activity_labels[activity]
            badge_color = "#6F7782" if row.get("archived") == "true" else activity_colors[activity]
            ax.add_patch(FancyBboxPatch((68.2, cursor + .13), 4.4, .46, boxstyle="round,pad=.02,rounding_size=.10", facecolor=badge_color, edgecolor="none"))
            ax.text(70.4, cursor + .36, activity_text, fontsize=6.5, color="white", ha="center", va="center", fontweight="bold")
            stars, forks = float(row["stars_numeric"]), float(row["forks_numeric"])
            for start, width, value, maximum, bar_color, text_x in (
                (74.7, 5.1, stars, max_stars, "#2F6F9F", 80.15),
                (84.0, 4.1, forks, max_forks, "#7B5EA7", 88.4),
            ):
                ax.add_patch(FancyBboxPatch((start, cursor + .25), width, .22, boxstyle="round,pad=.01,rounding_size=.05", facecolor="#E9EEF1", edgecolor="none"))
                scaled = width * math.log1p(value) / math.log1p(maximum)
                ax.add_patch(FancyBboxPatch((start, cursor + .25), scaled, .22, boxstyle="round,pad=.01,rounding_size=.05", facecolor=bar_color, edgecolor="none"))
                ax.text(text_x, cursor + .36, _compact_number(value), fontsize=6.7, color="#43505C", va="center")
            commit_date = str(row.get("overview_commit_at", ""))[:10] or "unknown"
            ax.text(91.2, cursor + .36, commit_date, fontsize=6.8, color="#43505C", va="center")
            cursor += .72
            row_index += 1
        cursor += .10

    legend_y = total_units - 1.45
    ax.text(1.2, legend_y, "Role:", fontsize=7.2, fontweight="bold", color="#43505C", va="center")
    legend_x = 5.0
    for _role, _abbreviation, full, role_color in ROLE_MATRIX:
        ax.scatter([legend_x], [legend_y], s=18, marker="o", facecolor=role_color, edgecolor=role_color, linewidth=.7, zorder=3)
        ax.text(legend_x + .45, legend_y, full, fontsize=6.7, color="#59636F", va="center")
        legend_x += 10.7
    ax.scatter([72.0], [legend_y], s=14, marker="o", facecolor="#2F6F9F", edgecolor="#2F6F9F", linewidth=.7, zorder=3); ax.text(72.45, legend_y, "canonical", fontsize=6.7, color="#59636F", va="center")
    ax.scatter([79.2], [legend_y], s=14, marker="o", facecolor="white", edgecolor="#9AA5AE", linewidth=.7, zorder=3); ax.text(79.65, legend_y, "related variant", fontsize=6.7, color="#59636F", va="center")
    ax.text(1.2, total_units - .65, metadata_caption(frame, "bars use log scaling; popularity is descriptive only"), fontsize=6.5, color="#6F7782", va="center")
    return save_figure(fig, out_dir, "00_repository_overview", formats, dpi)


def _system_type_definitions(profile: dict[str, object] | None) -> pd.DataFrame:
    """Flatten configured system-type rules into an auditable plot-side table."""
    types = (profile or DEFAULT_SYSTEM_TYPE_PROFILE).get("types", {})
    rows: list[dict[str, object]] = []
    if not isinstance(types, dict):
        return pd.DataFrame()
    for code, value in types.items():
        rule = value if isinstance(value, dict) else {}
        rows.append({
            "code": code, "name": rule.get("name", code), "description": rule.get("description", ""),
            "priority": rule.get("priority", 0), "fallback_only": str(bool(rule.get("fallback_only"))).lower(),
            "any_scopes": ";".join(rule.get("any_scopes", [])), "all_scopes": ";".join(rule.get("all_scopes", [])),
            "any_capabilities": ";".join(rule.get("any_capabilities", [])), "all_capabilities": ";".join(rule.get("all_capabilities", [])),
            "any_methods": ";".join(rule.get("any_methods", [])), "all_methods": ";".join(rule.get("all_methods", [])),
            "any_roles": ";".join(rule.get("any_roles", [])),
        })
    return pd.DataFrame(rows).sort_values(["priority", "code"], ascending=[False, True], kind="stable")


def build(
    input_path: Path, out_dir: Path, config: dict[str, object], system_type_profile: dict[str, object] | None = None,
) -> list[Path]:
    """Generate the complete GitHub catalogue plot and aggregate-table bundle."""
    LOG.info("building GitHub plots input=%s output=%s", input_path, out_dir)
    configure_style()
    frame = load_frame(input_path, ("github_repo_id", "project_id", "repository_relationship", "role", "observation_scopes", "capabilities", "detection_methods", "language", "stars", "forks", "pushed_at"))
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi, top_n = int(config.get("dpi", 160)), int(config.get("top_n", 15))
    caption = metadata_caption(frame, "multi-label figures may count a repository more than once")
    outputs: list[Path] = []
    exact_commit = frame.get("last_commit_at", pd.Series("", index=frame.index)).fillna("")
    type_values = frame.get("system_types", frame.get("derived_labels", pd.Series("unknown", index=frame.index))).replace("", "unknown")
    primary_types = frame.get("primary_system_type", pd.Series("", index=frame.index)).replace("", pd.NA).fillna(type_values.str.split(";").str[0]).replace("", "unknown")
    overview = pd.DataFrame({
        "repository": frame["full_name"],
        "system_role": frame["role"].map(ROLE_LABELS).fillna(frame["role"]),
        "primary_system_type": primary_types,
        "system_types": type_values,
        "system_type_names": frame.get("system_type_names", pd.Series("", index=frame.index)),
        "observation_scope": frame["observation_scopes"],
        "capabilities": frame["capabilities"],
        "methods": frame["detection_methods"],
        "relationship": frame["repository_relationship"],
        "stars": frame["stars"], "forks": frame["forks"],
        "last_commit_at": exact_commit.where(exact_commit != "", frame["pushed_at"]),
        "last_commit_time_source": exact_commit.map(lambda value: "commit_metadata" if value else "repository_pushed_at_fallback"),
        "repository_url": frame.get("html_url", pd.Series("", index=frame.index)),
    })
    overview["_canonical"] = (overview["relationship"] != "canonical").astype(int)
    overview = overview.sort_values(["_canonical", "system_role", "repository"], kind="stable").drop(columns="_canonical")
    write_table(overview, out_dir / "data", "00_repository_overview")
    write_table(_system_type_definitions(system_type_profile), out_dir / "data", "00_system_type_definitions")
    for stale in out_dir.glob("00_repository_overview_p*.*"):
        stale.unlink()
    outputs += _overview_matrix(frame, out_dir, formats, dpi)
    outputs += count_plot(frame, "role", "Repository roles", caption, out_dir, "01_role_counts", formats, dpi)
    outputs += count_plot(frame, "repository_relationship", "Repository relationships", caption, out_dir, "02_repository_relationships", formats, dpi)
    canonical = frame[frame["repository_relationship"] == "canonical"].copy()
    canonical["system_types"] = canonical.get("system_types", canonical.get("derived_labels", pd.Series("unknown", index=canonical.index))).replace("", "unknown")
    system_type_rows = explode_values(canonical, "system_types", "github_repo_id")
    outputs += count_plot(system_type_rows, "system_types", "Configured system types", metadata_caption(canonical, "canonical projects; deterministically derived; multi-label"), out_dir, "02b_system_types", formats, dpi)
    detectors = canonical[canonical["role"] == "standalone_detector"]
    method_rows = canonical[canonical["role"].isin(["standalone_detector", "classifier_model", "platform_framework"])]
    for number, column, title, source, filter_note in (
        (3, "observation_scopes", "Observation scopes", detectors, "canonical standalone_detector projects only"),
        (4, "capabilities", "Detection and response capabilities", detectors, "canonical standalone_detector projects only"),
        (5, "detection_methods", "Detection methods", method_rows, "canonical detectors, classifier models, and platforms only"),
    ):
        exploded = explode_values(source, column, "github_repo_id")
        outputs += count_plot(exploded, column, title, metadata_caption(source, f"{filter_note}; multi-label"), out_dir, f"0{number}_{column}", formats, dpi)
    role_scope = explode_values(method_rows, "observation_scopes", "github_repo_id").merge(method_rows[["github_repo_id", "role"]], on="github_repo_id")
    matrix = pd.crosstab(role_scope["role"], role_scope["observation_scopes"])
    if matrix.empty:
        matrix = pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    write_table(matrix.reset_index(), out_dir / "data", "06_role_scope_heatmap")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    image = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(value).replace("_", " ") for value in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), [str(value).replace("_", " ") for value in matrix.index])
    add_heading(fig, "Repository role by observation scope", metadata_caption(method_rows, "canonical detectors, classifier models, and platforms; multi-label"))
    fig.colorbar(image, ax=ax, label="Repositories")
    stamp(fig, metadata_caption(method_rows, "detectors, classifier models, and platforms only; multi-label"))
    outputs += save_figure(fig, out_dir, "06_role_scope_heatmap", formats, dpi)
    frame["activity"] = frame["pushed_at"].map(_activity)
    outputs += count_plot(frame, "activity", "Repository activity", metadata_caption(frame, "all accepted repositories; based on pushed_at at plot time"), out_dir, "07_activity", formats, dpi)
    outputs += count_plot(frame, "language", "Primary languages", caption, out_dir, "08_languages", formats, dpi, top_n)
    numeric = canonical.copy()
    numeric["stars"] = pd.to_numeric(numeric["stars"], errors="coerce").fillna(0)
    numeric["forks"] = pd.to_numeric(numeric["forks"], errors="coerce").fillna(0)
    fig, ax = plt.subplots()
    for index, (role, group) in enumerate(numeric.groupby("role")):
        ax.scatter(group["stars"] + 1, group["forks"] + 1, label=str(role).replace("_", " "), alpha=.68, s=24, color=COLORS[index % len(COLORS)])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Stars + 1 (log)"); ax.set_ylabel("Forks + 1 (log)")
    add_heading(fig, "Repository popularity—not relevance", metadata_caption(canonical, "canonical projects only; popularity never affects selection"))
    ax.legend(title="System role", fontsize=7, ncol=2, frameon=False)
    stamp(fig, metadata_caption(canonical, "canonical projects only; popularity never affects inclusion or canonical selection"))
    write_table(numeric[["github_repo_id", "project_id", "full_name", "role", "stars", "forks"]], out_dir / "data", "09_popularity")
    outputs += save_figure(fig, out_dir, "09_popularity", formats, dpi)
    metadata_fields = ["html_url", "description", "topics", "license_spdx", "language", "pushed_at", "last_commit_sha", "last_commit_at"]
    completeness = pd.DataFrame({"field": metadata_fields, "present": [((frame[field] != "") & (frame[field].str.casefold() != "unknown")).sum() if field in frame else 0 for field in metadata_fields]})
    completeness["percent"] = completeness["present"] / max(1, len(frame)) * 100
    write_table(completeness, out_dir / "data", "10_metadata_completeness")
    fig, ax = plt.subplots(); ordered = completeness.iloc[::-1]
    ax.barh(ordered["field"], ordered["percent"], color=COLORS[1]); ax.set_xlim(0, 100); ax.set_xlabel("Rows populated (%)")
    add_heading(fig, "Repository metadata completeness", metadata_caption(frame, "all accepted roles"))
    count_legend(ax, "Rows populated (%)", COLORS[1])
    stamp(fig, metadata_caption(frame, "all accepted roles"))
    outputs += save_figure(fig, out_dir, "10_metadata_completeness", formats, dpi)
    outputs.append(write_index(out_dir, input_path, frame, outputs, [
        "Role, relationship, activity, language, and completeness cover all accepted repository rows.",
        "System types are deterministic rules from config/taxonomy/github_system_types.toml; data/00_system_type_definitions.csv records the exact definitions.",
        "Scope/capability figures use canonical standalone detector projects; method figures also include canonical classifier models and platforms.",
        "Popularity uses canonical projects only and never affects inclusion or canonical selection.",
        "Multi-label plots may count one repository in more than one category.",
        "The complete repository overview is data/00_repository_overview.csv; its rendered pages include every accepted entry.",
    ]))
    LOG.info("GitHub plot bundle complete files=%d output=%s", len(outputs), out_dir)
    return outputs
