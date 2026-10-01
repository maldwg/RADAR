from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import pandas as pd

from ids_meta.plotting.common import COLORS, add_heading, configure_style, count_legend, count_plot, evidence_escalation_plot, explode_values, load_frame, metadata_caption, save_figure, score_decision_plot, stamp, write_index, write_table
from ids_meta.common.logging import logger
from ids_meta.github.schema import DEFAULT_SYSTEM_TYPE_PROFILE

LOG = logger("plotting.github")

ROLE_LABELS = {
    "native_detector": "Native detector", "classifier_engine": "Classifier engine",
    "detection_framework": "Detection framework", "detection_platform": "Detection platform",
    "research_project": "Research project", "demo": "Demo",
    "rules_content": "Rules/content", "support_tooling": "Support tooling",
}
ROLE_MATRIX = [
    ("native_detector", "Native", "Native detector", "#2A9D8F"),
    ("classifier_engine", "Class.", "Classifier engine", "#D1842F"),
    ("detection_framework", "Frame", "Detection framework", "#7B5EA7"),
    ("detection_platform", "Plat.", "Detection platform", "#4C78A8"),
    ("research_project", "Res.", "Research project", "#9A7D0A"),
    ("demo", "Demo", "Demo", "#B59B6A"),
    ("rules_content", "Rules", "Rules/content", "#2F6F9F"),
    ("support_tooling", "Supp.", "Support tooling", "#6F7782"),
]
OPERATIONAL_ROLES = {"native_detector", "classifier_engine", "detection_framework"}
FAMILY_COLORS = ["#2F6F9F", "#2A9D8F", "#D1842F", "#7B5EA7", "#C94F4F", "#5B8F3A", "#6F7782"]
CANONICAL_RELATIONSHIPS = {"canonical"}


def _canonical_mask(frame: pd.DataFrame) -> pd.Series:
    """Select canonical project representatives."""
    return frame["repository_relationship"].isin(CANONICAL_RELATIONSHIPS)


def _operational_mask(frame: pd.DataFrame) -> pd.Series:
    """Select the configured operational reporting roles."""
    return _canonical_mask(frame) & frame["role"].isin(OPERATIONAL_ROLES)


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
    labels = [value for value in str(row.get("system_types", "")).split(";") if value and value != "unknown"]
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
    view["variant_order"] = (~_canonical_mask(view)).astype(int)
    view = view.sort_values(["family_order", "variant_order", "role", "stars_numeric", "full_name"], ascending=[True, True, True, False, True], kind="stable")

    group_count = view["overview_family"].nunique() if not view.empty else 1
    total_units = 4.2 + len(view) * .78 + group_count * .92 + 2.1
    fig, ax = plt.subplots(figsize=(18, max(7, total_units * .34)))
    ax.set_xlim(0, 100); ax.set_ylim(total_units, 0); ax.axis("off")
    ax.text(50, .55, "Repository feature matrix", ha="center", va="center", fontsize=17, fontweight="bold", color="#263238")

    headers = [(1.3, "Repository"), (20.5, "System"), (30.5, "Detection method(s)"), (68.5, "Activity"), (75, "Stars"), (84.2, "Forks"), (91.2, "Last commit")]
    for x, label in headers:
        ax.text(x, 2.25, label, fontsize=8, color="#59636F", va="center")
    role_x = [47.0 + index * 2.8 for index in range(len(ROLE_MATRIX))]
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
            canonical = row.get("repository_relationship") in CANONICAL_RELATIONSHIPS
            ax.scatter([1.15], [cursor + .36], s=13, marker="o", facecolor="#2F6F9F" if canonical else "white", edgecolor="#2F6F9F" if canonical else "#9AA5AE", linewidth=.8, zorder=3)
            ax.text(1.55, cursor + .36, _short(row.get("full_name", ""), 34), fontsize=7.2, color="#263238", va="center")
            system = str(row.get("system_types", "")) or str(row.get("observation_scopes", ""))
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
    """Flatten the configured LLM-facing system-type vocabulary for audit."""
    types = (profile or DEFAULT_SYSTEM_TYPE_PROFILE).get("types", {})
    rows: list[dict[str, object]] = []
    if not isinstance(types, dict):
        return pd.DataFrame()
    for code, value in types.items():
        definition = value if isinstance(value, dict) else {}
        rows.append({
            "code": code, "name": definition.get("name", code),
            "description": definition.get("description", ""), "prompt_order": len(rows) + 1,
        })
    return pd.DataFrame(rows)


def _clean_values(value: object) -> list[str]:
    """Split a semicolon/comma separated multi-value cell into stable non-empty labels."""
    raw = str(value or "").strip()
    if not raw or raw.casefold() in {"unknown", "none", "nan", "null", "n/a", "na"}:
        return []
    values: list[str] = []
    for semicolon_part in raw.split(";"):
        for item in semicolon_part.split(","):
            item = item.strip()
            if item and item.casefold() not in {"unknown", "none", "nan", "null", "n/a", "na"}:
                values.append(item)
    return list(dict.fromkeys(values))


def _display_system_type(value: object) -> str:
    """Format normalized IDS-type codes such as nids/hids/pids for plot labels."""
    text = str(value or "unknown").strip()
    if not text:
        return "Unknown"
    upper_codes = {"nids", "hids", "pids", "wids", "xdr", "ndr", "ids", "ips"}
    if text.casefold() in upper_codes:
        return text.upper()
    return text.replace("_", " ").title()


def _primary_system_type_series(frame: pd.DataFrame) -> pd.Series:
    """Return one primary IDS type per row with deterministic fallbacks."""
    system_types = frame.get("system_types", pd.Series("unknown", index=frame.index)).fillna("").astype(str)
    primary = frame.get("primary_system_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    fallback = system_types.map(lambda value: (_clean_values(value) or ["unknown"])[0])
    return primary.where(primary.str.strip() != "", fallback).replace("", "unknown")


def _system_type_colors(values: list[str]) -> dict[str, str]:
    """Assign stable colors to the IDS types present in a figure."""
    ordered = list(dict.fromkeys(values))
    return {value: FAMILY_COLORS[index % len(FAMILY_COLORS)] for index, value in enumerate(ordered)}


def _repository_timeline(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot repository creation time on x, GitHub stars on y, and IDS type as color."""
    canonical = frame[_operational_mask(frame)].copy()
    canonical["primary_system_type"] = _primary_system_type_series(canonical)
    canonical["stars_numeric"] = pd.to_numeric(canonical.get("stars", 0), errors="coerce").fillna(0).clip(lower=0)

    created_raw = canonical.get("created_at", pd.Series("", index=canonical.index)).fillna("")
    canonical["timeline_date"] = pd.to_datetime(created_raw, errors="coerce", utc=True)
    canonical["timeline_date_source"] = "created_at"
    missing_created = canonical["timeline_date"].isna()
    pushed_raw = canonical.get("pushed_at", pd.Series("", index=canonical.index)).fillna("")
    canonical.loc[missing_created, "timeline_date"] = pd.to_datetime(pushed_raw[missing_created], errors="coerce", utc=True)
    canonical.loc[missing_created, "timeline_date_source"] = "pushed_at_fallback"
    canonical = canonical.dropna(subset=["timeline_date"]).sort_values(["timeline_date", "stars_numeric", "full_name"], kind="stable")

    columns = [
        "github_repo_id", "project_id", "full_name", "primary_system_type", "system_types", "role",
        "timeline_date", "timeline_date_source", "stars_numeric", "repository_relationship", "html_url",
    ]
    for column in columns:
        if column not in canonical:
            canonical[column] = ""
    timeline_table = canonical[columns].rename(columns={"stars_numeric": "stars"})
    write_table(timeline_table, out_dir / "data", "01_repository_timeline")

    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    add_heading(
        fig,
        "IDS repository timeline",
        metadata_caption(canonical, "canonical operational systems; x = repository creation date; y = GitHub stars; color = primary IDS type"),
        top=.84,
    )
    fig.subplots_adjust(left=.09, right=.84, bottom=.22)
    if canonical.empty:
        ax.axis("off")
        ax.text(.5, .5, "No dated canonical repositories", transform=ax.transAxes, ha="center", va="center")
        return save_figure(fig, out_dir, "01_repository_timeline", formats, dpi)

    type_values = canonical["primary_system_type"].astype(str).tolist()
    colors = _system_type_colors(type_values)
    for system_type, group in canonical.groupby("primary_system_type", sort=False):
        ax.scatter(
            group["timeline_date"], group["stars_numeric"], s=58,
            color=colors[str(system_type)], edgecolor="white", linewidth=.9,
            label=_display_system_type(system_type), zorder=3,
        )

    # Repository names are mandatory. Place them underneath each point; duplicate dates/stars
    # get deterministic horizontal offsets without moving the actual data point.
    canonical["_duplicate_index"] = canonical.groupby(["timeline_date", "stars_numeric"], sort=False).cumcount()
    canonical["_duplicate_size"] = canonical.groupby(["timeline_date", "stars_numeric"], sort=False)["github_repo_id"].transform("size")
    for _, row in canonical.iterrows():
        duplicate_index = int(row["_duplicate_index"])
        duplicate_size = int(row["_duplicate_size"])
        dx = (duplicate_index - (duplicate_size - 1) / 2.0) * 32
        ax.annotate(
            _short(row.get("full_name", ""), 32),
            (row["timeline_date"], row["stars_numeric"]),
            xytext=(dx, -12), textcoords="offset points",
            ha="center", va="top", fontsize=6.6, color="#43505C",
            arrowprops={"arrowstyle": "-", "color": "#C7D0D6", "lw": .5} if abs(dx) > 1 else None,
            annotation_clip=False,
        )

    ax.set_xlabel("Repository creation date")
    ax.set_ylabel("GitHub stars")
    ax.set_ylim(bottom=min(-1, -float(canonical["stars_numeric"].max()) * .03))
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="x", alpha=.35)
    ax.grid(axis="y", alpha=.25)
    handles = [Patch(facecolor=colors[value], label=_display_system_type(value)) for value in dict.fromkeys(type_values)]
    ax.legend(handles=handles, title="IDS type", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    return save_figure(fig, out_dir, "01_repository_timeline", formats, dpi)


def _role_type_matrix(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Cross all accepted repository roles with normalized IDS types."""
    canonical = frame[_canonical_mask(frame)].copy()
    canonical["system_types"] = canonical.get("system_types", pd.Series("unknown", index=canonical.index)).fillna("").astype(str)
    records: list[dict[str, object]] = []
    for _, row in canonical.iterrows():
        types = _clean_values(row.get("system_types", "")) or ["unknown"]
        for system_type in types:
            records.append({
                "github_repo_id": row.get("github_repo_id", ""),
                "full_name": row.get("full_name", ""),
                "role": row.get("role", "unknown") or "unknown",
                "system_type": system_type,
            })
    expanded = pd.DataFrame(records, columns=["github_repo_id", "full_name", "role", "system_type"])
    write_table(expanded, out_dir / "data", "02_role_type_matrix_records")
    matrix = pd.crosstab(expanded["role"], expanded["system_type"]) if not expanded.empty else pd.DataFrame()
    preferred_roles = [role for role, *_ in ROLE_MATRIX]
    role_order = [role for role in preferred_roles if role in matrix.index] + [role for role in matrix.index if role not in preferred_roles]
    if role_order:
        matrix = matrix.reindex(role_order)
    write_table(matrix.reset_index(), out_dir / "data", "02_role_type_matrix")

    display = matrix if not matrix.empty else pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    fig, ax = plt.subplots(figsize=(max(10.5, 6.5 + .8 * len(display.columns)), max(5.7, 3.0 + .55 * len(display.index))))
    add_heading(fig, "Repository role by IDS type", metadata_caption(canonical, "canonical projects; multi-label IDS types"), top=.84)
    image = ax.imshow(display.values, cmap="Blues", aspect="auto", vmin=0)
    maximum = max(1, int(display.to_numpy().max()))
    for r in range(len(display.index)):
        for c in range(len(display.columns)):
            count = int(display.iloc[r, c])
            ax.text(c, r, str(count), ha="center", va="center", fontsize=8, fontweight="bold", color="white" if count > maximum * .52 else "#263238")
    ax.set_xticks(range(len(display.columns)), [_display_system_type(value) for value in display.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(display.index)), [ROLE_LABELS.get(str(value), str(value).replace("_", " ").title()) for value in display.index])
    ax.set_xlabel("IDS type")
    ax.set_ylabel("Repository role")
    ax.grid(False)
    fig.colorbar(image, ax=ax, pad=.02, label="Repositories")
    return save_figure(fig, out_dir, "02_role_type_matrix", formats, dpi)


def _detection_type_matrix(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Cross detection methods with normalized IDS types for operational systems."""
    canonical = frame[_operational_mask(frame)].copy()
    records: list[dict[str, object]] = []
    for _, row in canonical.iterrows():
        types = _clean_values(row.get("system_types", "")) or ["unknown"]
        methods = _clean_values(row.get("detection_methods", "")) or ["unknown"]
        for system_type in types:
            for method in methods:
                records.append({
                    "github_repo_id": row.get("github_repo_id", ""),
                    "full_name": row.get("full_name", ""),
                    "system_type": system_type,
                    "detection_method": method,
                })
    expanded = pd.DataFrame(records, columns=["github_repo_id", "full_name", "system_type", "detection_method"])
    write_table(expanded, out_dir / "data", "04_detection_method_type_matrix_records")
    matrix = pd.crosstab(expanded["detection_method"], expanded["system_type"]) if not expanded.empty else pd.DataFrame()
    if not matrix.empty:
        matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    write_table(matrix.reset_index(), out_dir / "data", "04_detection_method_type_matrix")

    display = matrix if not matrix.empty else pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    fig, ax = plt.subplots(figsize=(max(10.5, 6.5 + .8 * len(display.columns)), max(5.8, 3.0 + .55 * len(display.index))))
    add_heading(fig, "Detection method by IDS type", metadata_caption(canonical, "canonical operational systems; methods and IDS types are multi-label"), top=.84)
    image = ax.imshow(display.values, cmap="Blues", aspect="auto", vmin=0)
    maximum = max(1, int(display.to_numpy().max()))
    for r in range(len(display.index)):
        for c in range(len(display.columns)):
            count = int(display.iloc[r, c])
            ax.text(c, r, str(count), ha="center", va="center", fontsize=8, fontweight="bold", color="white" if count > maximum * .52 else "#263238")
    ax.set_xticks(range(len(display.columns)), [_display_system_type(value) for value in display.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(display.index)), [str(value).replace("_", " ") for value in display.index])
    ax.set_xlabel("IDS type")
    ax.set_ylabel("Detection method")
    ax.grid(False)
    fig.colorbar(image, ax=ax, pad=.02, label="Repositories")
    return save_figure(fig, out_dir, "04_detection_method_type_matrix", formats, dpi)


def _write_count_table(frame: pd.DataFrame, column: str, out_dir: Path, stem: str) -> None:
    """Write exact counts for a count plot plus row-level records used by it."""
    if column not in frame:
        write_table(pd.DataFrame(columns=[column, "count"]), out_dir / "data", stem)
        write_table(pd.DataFrame(), out_dir / "data", f"{stem}_records")
        return
    values = frame[column].fillna("").astype(str).replace("", "unknown")
    counts = values.value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    write_table(counts, out_dir / "data", stem)
    record_columns = [value for value in ["github_repo_id", "project_id", "full_name", column] if value in frame]
    write_table(frame[record_columns].copy(), out_dir / "data", f"{stem}_records")


def _metadata_completeness(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot percentage of repositories with each metadata field populated, including GitHub topics/tags."""
    fields = [
        ("html_url", "Repository URL"),
        ("description", "Description"),
        ("topics", "Topics/tags"),
        ("tags", "Tags"),
        ("license_spdx", "License"),
        ("language", "Primary language"),
        ("created_at", "Created at"),
        ("pushed_at", "Pushed at"),
        ("last_commit_sha", "Last commit SHA"),
        ("last_commit_at", "Last commit timestamp"),
        ("system_types", "IDS type"),
        ("detection_methods", "Detection methods"),
        ("observation_objects", "Observation objects"),
        ("observation_scopes", "Observation scopes"),
        ("capabilities", "Capabilities"),
        ("role", "Repository role"),
        ("repository_relationship", "Repository relationship"),
        ("discovery_relations", "Discovery provenance"),
    ]
    # Do not add a duplicate standalone Tags row if the schema has no tags field.
    fields = [(field, label) for field, label in fields if field in frame.columns or field != "tags"]
    missing = {"", "unknown", "none", "nan", "null", "n/a", "na", "[]", "{}"}
    rows: list[dict[str, object]] = []
    for field, label in fields:
        if field not in frame:
            present = 0
        else:
            present = int(frame[field].map(lambda value: str(value).strip().casefold() not in missing).sum())
        rows.append({"field": field, "label": label, "present": present, "total": len(frame), "percent": present / max(1, len(frame)) * 100})
    completeness = pd.DataFrame(rows).sort_values(["percent", "label"], ascending=[True, True], kind="stable")
    write_table(completeness, out_dir / "data", "06_metadata_completeness")

    record_columns = [
        value for value in [
            "github_repo_id", "project_id", "full_name", "html_url", "description", "topics", "tags", "license_spdx", "language",
            "created_at", "pushed_at", "last_commit_sha", "last_commit_at", "system_types", "detection_methods", "observation_objects", "observation_scopes",
            "capabilities", "role", "repository_relationship", "discovery_relations",
        ] if value in frame
    ]
    write_table(frame[record_columns].copy(), out_dir / "data", "06_metadata_completeness_records")

    fig, ax = plt.subplots(figsize=(11.3, max(6.5, 3.0 + .36 * len(completeness))))
    add_heading(fig, "Repository metadata completeness", metadata_caption(frame, "all accepted repositories; non-empty/non-placeholder values"), top=.84)
    positions = list(range(len(completeness)))
    ax.barh(positions, [100] * len(positions), height=.58, color="#EEF2F4", edgecolor="none")
    ax.barh(positions, completeness["percent"], height=.58, color=COLORS[1], edgecolor="none")
    ax.set_yticks(positions, completeness["label"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Repositories with metadata available (%)")
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    for position, (_, row) in zip(positions, completeness.iterrows()):
        ax.text(min(109, float(row["percent"]) + 1.5), position, f"{row['percent']:.0f}% ({int(row['present'])}/{int(row['total'])})", va="center", fontsize=7.4, color="#43505C")
    return save_figure(fig, out_dir, "06_metadata_completeness", formats, dpi)


def _tag_counts(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int, top_n: int) -> list[Path]:
    """Plot the most common GitHub topics/tags across accepted repositories."""
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        combined: list[str] = []
        for field in ("topics", "tags"):
            if field in frame:
                combined.extend(_clean_values(row.get(field, "")))
        for tag in dict.fromkeys(combined):
            records.append({"github_repo_id": row.get("github_repo_id", ""), "full_name": row.get("full_name", ""), "tag": tag})
    expanded = pd.DataFrame(records, columns=["github_repo_id", "full_name", "tag"])
    counts = expanded["tag"].value_counts().rename_axis("tag").reset_index(name="repositories") if not expanded.empty else pd.DataFrame(columns=["tag", "repositories"])
    counts["percent_of_repositories"] = counts["repositories"] / max(1, len(frame)) * 100
    write_table(counts, out_dir / "data", "07_common_tags")
    write_table(expanded, out_dir / "data", "07_common_tags_records")

    plotted = counts.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.2, max(5.8, 3.2 + .35 * max(1, len(plotted)))))
    add_heading(fig, "Most common repository tags", metadata_caption(frame, f"all accepted repositories; top {top_n}; a repository may have multiple tags"), top=.84)
    if plotted.empty:
        ax.axis("off")
        ax.text(.5, .5, "No repository topics/tags available", transform=ax.transAxes, ha="center", va="center")
        return save_figure(fig, out_dir, "07_common_tags", formats, dpi)
    positions = list(range(len(plotted)))
    ax.barh(positions, plotted["repositories"], color=COLORS[0])
    ax.set_yticks(positions, plotted["tag"])
    ax.set_xlabel("Repositories")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", visible=False)
    for position, (_, row) in zip(positions, plotted.iterrows()):
        ax.text(float(row["repositories"]) + .15, position, f"{int(row['repositories'])} ({row['percent_of_repositories']:.0f}%)", va="center", fontsize=7.3, color="#43505C")
    return save_figure(fig, out_dir, "07_common_tags", formats, dpi)


def _discovery_source_label(token: str) -> str:
    """Normalize detailed GitHub discovery-relation tokens into readable source groups."""
    normalized = token.strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "Other"
    if normalized == "direct_search" or ("direct" in normalized and "search" in normalized):
        return "Direct search"
    if "snowball" in normalized or "reference" in normalized or "referenced" in normalized or normalized.startswith("related_"):
        return "Reference / snowball"
    if "seed" in normalized:
        return "Seed source"
    if "search" in normalized:
        return "Search source"
    return token.strip().replace("_", " ").title()


def _discovery_source_boxplot(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Boxplot annual repository counts by discovery source, using repository creation year as the time grouping."""
    view = frame.copy()
    relations = view.get("discovery_relations", pd.Series("", index=view.index)).fillna("").astype(str)
    created = pd.to_datetime(view.get("created_at", pd.Series("", index=view.index)), errors="coerce", utc=True)
    fallback = pd.to_datetime(view.get("pushed_at", pd.Series("", index=view.index)), errors="coerce", utc=True)
    view["discovery_year"] = created.fillna(fallback).dt.year
    view["discovery_year_source"] = created.notna().map({True: "created_at", False: "pushed_at_fallback"})

    record_rows: list[dict[str, object]] = []
    for index, row in view.iterrows():
        source_tokens = _clean_values(relations.loc[index])
        sources = list(dict.fromkeys(_discovery_source_label(token) for token in source_tokens)) or ["Other"]
        for source in sources:
            record_rows.append({
                "github_repo_id": row.get("github_repo_id", ""),
                "full_name": row.get("full_name", ""),
                "discovery_year": row.get("discovery_year"),
                "discovery_year_source": row.get("discovery_year_source", ""),
                "source": source,
                "discovery_relations": row.get("discovery_relations", ""),
            })
    records = pd.DataFrame(record_rows, columns=["github_repo_id", "full_name", "discovery_year", "discovery_year_source", "source", "discovery_relations"])
    write_table(records, out_dir / "data", "08_discovery_sources_records")

    dated = records.dropna(subset=["discovery_year"]).copy()
    if not dated.empty:
        dated["discovery_year"] = dated["discovery_year"].astype(int)
        source_order = dated.groupby("source")["github_repo_id"].nunique().sort_values(ascending=False).index.tolist()
        years = range(int(dated["discovery_year"].min()), int(dated["discovery_year"].max()) + 1)
        grid = pd.MultiIndex.from_product([years, source_order], names=["discovery_year", "source"]).to_frame(index=False)
        annual = grid.merge(
            dated.groupby(["discovery_year", "source"])["github_repo_id"].nunique().rename("repositories").reset_index(),
            how="left", on=["discovery_year", "source"],
        ).fillna({"repositories": 0})
        annual["repositories"] = annual["repositories"].astype(int)
    else:
        source_order = []
        annual = pd.DataFrame(columns=["discovery_year", "source", "repositories"])
    write_table(annual, out_dir / "data", "08_discovery_sources")

    fig, ax = plt.subplots(figsize=(max(10.5, 2.0 + 2.0 * max(1, len(source_order))), 6.4))
    add_heading(fig, "Discovery-source contribution", metadata_caption(view, "boxplots show annual repository counts per discovery source; multi-source repositories may appear in multiple groups"), top=.84)
    if annual.empty:
        ax.axis("off")
        ax.text(.5, .5, "No dated discovery-provenance data", transform=ax.transAxes, ha="center", va="center")
        return save_figure(fig, out_dir, "08_discovery_sources", formats, dpi)
    values = [annual.loc[annual["source"] == source, "repositories"].tolist() for source in source_order]
    positions = list(range(1, len(source_order) + 1))
    boxes = ax.boxplot(values, positions=positions, widths=.55, patch_artist=True, showfliers=False)
    for index, patch in enumerate(boxes["boxes"]):
        color = FAMILY_COLORS[index % len(FAMILY_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(.24)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for median in boxes["medians"]:
        median.set_color("#263238")
        median.set_linewidth(1.5)
    for name in ("whiskers", "caps"):
        for item in boxes[name]:
            item.set_color("#7A8791")
    ax.set_xticks(positions, source_order, rotation=20, ha="right")
    ax.set_ylabel("Repositories per creation year")
    ax.set_xlabel("Discovery source")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    return save_figure(fig, out_dir, "08_discovery_sources", formats, dpi)


def _github_why_not_included(row: pd.Series) -> tuple[str, str, str]:
    """Explain one excluded or unresolved repository in reader-facing language."""
    decision = str(row.get("decision", "")).strip().casefold()
    flags = [value.strip() for value in str(row.get("classification_validation_flags", "")).split(";") if value.strip()]
    flag_meanings = {
        "runtime_observation_missing": ("Could not verify live or operational input", "The evidence does not show the software processing live or continuously produced observations."),
        "reproducible_execution_missing": ("Could not verify how to run or train it", "No concrete build, training, inference, service, command-line, or execution path was verified."),
        "offline_dataset_only": ("Only works on saved datasets or captures", "The available execution path is limited to static datasets, saved logs, or saved packet captures."),
        "detection_engine_implementation_missing": ("Could not verify its own detection logic", "The repository does not clearly implement the logic that decides whether activity is malicious or anomalous."),
        "external_detector_required": ("Another IDS performs the actual detection", "The repository depends on an external detector to make the intrusion decision."),
        "framework_architecture_conflict": ("Architecture is a framework, not one native detector", "The evidence describes a cohesive agent-manager or multi-subsystem framework rather than a single detector engine."),
        "framework_architecture_missing": ("Could not verify a framework architecture", "The evidence does not establish a cohesive agent-manager or inseparable multi-subsystem detection system."),
        "multiple_detector_integration_missing": ("Could not verify integration of multiple detectors", "The evidence does not show coordination of at least two independently usable detector engines."),
        "research_basis_missing": ("No research basis could be verified", "No referenced paper, named IDS dataset, or direct research and reproducibility framing was verified."),
        "direct_self_description_missing": ("Repository does not clearly describe its own contribution", "The available evidence does not explicitly say what this repository itself contributes."),
        "independent_fork_evidence_missing": ("Could not verify that the fork is independently significant", "The fork lacks evidence of materially distinct functionality."),
        "verified_evidence_missing": ("No supporting quotation could be verified", "The classification lacks a quotation that can be matched to the collected repository evidence."),
        "decision_fields_conflict": ("Classification fields conflict", "The combined role and evidence fields do not support the ensemble decision."),
    }
    if decision == "error":
        return "Processing failed", "Final decision is error", "The classification request failed, so no inclusion decision was available."
    if decision == "uncertain":
        if flags:
            label, explanation = flag_meanings.get(flags[0], (flags[0].replace("_", " ").capitalize(), "A deterministic classification requirement was not satisfied."))
            return label, f"Automatic validation check: {flags[0]}", explanation
        vote_status = str(row.get("ensemble_vote_status", "")).strip()
        if vote_status in {"tie", "below_threshold"}:
            return "Models did not agree", f"Ensemble result: {vote_status}", "The configured models did not reach the agreement threshold for a definitive decision."
        return "Not enough evidence for a decision", "Final decision is uncertain", "The collected repository evidence did not support either inclusion or exclusion."
    reason = str(row.get("primary_reason", "")).casefold()
    rules = (
        ("Dataset files only — no runnable IDS approach", "The repository supplies data but no material trainable or runnable IDS implementation.", ("dataset-only", "dataset only", "only provides a dataset", "dataset rather than")),
        ("Tutorial or lab only — no reusable implementation", "The repository documents learning, setup, or demonstration work without a reusable IDS contribution.", ("tutorial", "walkthrough", "learning project", "student project", "coursework", "lab setup", "hands-on lab", "how-to", "guide for")),
        ("Reference list only — no implementation", "The repository is a bibliography or resource list rather than IDS software, detection content, or research code.", ("awesome list", "curated list", "resource list", "collection of links", "bibliography")),
        ("Offensive or scanning tool — not an intrusion detector", "The repository attacks, scans, exploits, or evades systems but does not provide a defensive intrusion-detection contribution.", ("offensive-only", "offensive tool", "penetration testing", "scanner", "exploit", "evasion", "red team")),
        ("Configuration or integration only — another system detects", "The repository configures, deploys, wraps, or uses an existing detector without supplying its own qualifying implementation or content.", ("existing third-party", "using an existing", "configuration for", "deployment of", "wrapper around", "integration with", "rather than implementing")),
        ("Generic monitoring or security tool — not IDS-specific", "The contribution is generic monitoring, logging, or security functionality without a material intrusion-detection purpose.", ("generic monitor", "generic monitoring", "general-purpose monitoring", "logging tool", "security tool without", "not specific to intrusion")),
        ("Unrelated project or IDS name collision", "The repository is unrelated to intrusion detection even though its name or text matched discovery terms.", ("unrelated", "name collision", "does not refer to intrusion", "not an intrusion")),
    )
    for label, explanation, keywords in rules:
        match = next((keyword for keyword in keywords if keyword in reason), "")
        if match:
            return label, f"Model explanation contains: {match}", explanation
    return "No concrete IDS contribution was established", "Model explanation did not match a narrower group", "The model excluded the repository, but its explanation did not fit one of the more specific reporting groups."


def _why_repositories_not_included(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot plain-language repository exclusion reasons and preserve detailed evidence."""
    decisions = frame.get("decision", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    view = frame[decisions.isin({"exclude", "uncertain", "error"})].copy()
    details = [_github_why_not_included(row) for _, row in view.iterrows()]
    view["why_not_included"] = [value[0] for value in details]
    view["how_reason_was_assigned"] = [value[1] for value in details]
    view["criterion_explanation"] = [value[2] for value in details]
    columns = [
        value for value in (
            "github_repo_id", "full_name", "html_url", "decision", "role", "why_not_included",
            "criterion_explanation", "how_reason_was_assigned", "classification_validation_flags",
            "primary_reason", "evidence_json", "prefilter_band", "prefilter_score",
            "ensemble_vote_status", "ensemble_agreement", "error",
        ) if value in view
    ]
    write_table(view[columns].copy(), out_dir / "data", "12_why_not_included_records")
    definitions = view[["why_not_included", "criterion_explanation"]].drop_duplicates().sort_values("why_not_included")
    write_table(definitions, out_dir / "data", "12_why_not_included_definitions")
    return count_plot(
        view, "why_not_included", "Why repositories are not in the catalogue",
        metadata_caption(view, "excluded, unresolved, and failed classifications; definitions and exact explanations are exported below"),
        out_dir, "12_why_not_included", formats, dpi,
    )


def _classification_audit(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Cross deterministic candidate/borderline/reject bands with final model inclusion/exclusion decisions."""
    view = frame.copy()
    band_column = next((column for column in ("prefilter_band", "filter_band", "candidate_band") if column in view), None)
    decision_column = next((column for column in ("decision", "final_decision", "classification_decision", "model_decision") if column in view), None)
    score_column = next((column for column in ("prefilter_score", "filter_score", "score") if column in view), None)

    if band_column is None or decision_column is None:
        write_table(pd.DataFrame(columns=["prefilter_band", "decision", "count", "percent_within_band"]), out_dir / "data", "09_classification_audit")
        write_table(pd.DataFrame(), out_dir / "data", "09_classification_audit_records")
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        add_heading(fig, "Candidate filtering versus final model decision", metadata_caption(view, "requires prefilter/filter band and final decision columns"), top=.84)
        ax.axis("off")
        ax.text(.5, .5, "No candidate-audit classification columns available", transform=ax.transAxes, ha="center", va="center")
        return save_figure(fig, out_dir, "09_classification_audit", formats, dpi)

    view["audit_band"] = view[band_column].fillna("").astype(str).replace("", "unknown")
    view["audit_decision"] = view[decision_column].fillna("").astype(str).replace("", "unknown")
    view["audit_score"] = pd.to_numeric(view[score_column], errors="coerce") if score_column else pd.Series(float("nan"), index=view.index)

    preferred_bands = ["candidate", "borderline", "reject", "unknown"]
    observed_bands = list(dict.fromkeys(view["audit_band"].tolist()))
    bands = [value for value in preferred_bands if value in observed_bands] + [value for value in observed_bands if value not in preferred_bands]
    preferred_decisions = ["include", "accept", "accept_adjacent", "exclude", "reject", "uncertain", "error", "unknown"]
    observed_decisions = list(dict.fromkeys(view["audit_decision"].tolist()))
    decisions = [value for value in preferred_decisions if value in observed_decisions] + [value for value in observed_decisions if value not in preferred_decisions]

    matrix = pd.crosstab(view["audit_band"], view["audit_decision"]).reindex(index=bands, columns=decisions, fill_value=0)
    percentages = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100
    score_summary = view.groupby("audit_band")["audit_score"].agg(["min", "median", "max"]).reindex(bands)
    table = matrix.stack().rename("count").reset_index().rename(columns={"audit_band": "prefilter_band", "audit_decision": "decision"})
    pct = percentages.stack().rename("percent_within_band").reset_index().rename(columns={"audit_band": "prefilter_band", "audit_decision": "decision"})
    table = table.merge(pct, on=["prefilter_band", "decision"])
    if not score_summary.empty:
        scores = score_summary.reset_index().rename(columns={"audit_band": "prefilter_band", "min": "score_min", "median": "score_median", "max": "score_max"})
        table = table.merge(scores, on="prefilter_band", how="left")
    write_table(table, out_dir / "data", "09_classification_audit")

    record_columns = [value for value in ["github_repo_id", "project_id", "full_name", band_column, score_column, decision_column, "role", "system_types", "discovery_relations"] if value and value in view]
    write_table(view[record_columns].copy(), out_dir / "data", "09_classification_audit_records")

    display = matrix if not matrix.empty else pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    display_pct = percentages if not percentages.empty else display.astype(float)
    fig, ax = plt.subplots(figsize=(max(10.5, 6.5 + .85 * len(display.columns)), max(5.5, 3.4 + .8 * len(display.index))))
    add_heading(fig, "Candidate filtering versus final model decision", metadata_caption(view, "cells show exact count and percentage within prefilter band"), top=.84)
    image = ax.imshow(display_pct.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for r in range(len(display.index)):
        for c in range(len(display.columns)):
            count = int(display.iloc[r, c])
            percent = float(display_pct.iloc[r, c])
            ax.text(c, r, f"{count}\n{percent:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color="white" if percent >= 55 else "#263238")
    row_labels: list[str] = []
    for band in display.index:
        label = str(band).replace("_", " ")
        if band in score_summary.index:
            summary = score_summary.loc[band]
            if pd.notna(summary.get("min")) and pd.notna(summary.get("max")):
                label += f"\nscore {summary['min']:g}–{summary['max']:g}; med {summary['median']:g}"
        row_labels.append(label)
    ax.set_xticks(range(len(display.columns)), [str(value).replace("_", " ") for value in display.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(display.index)), row_labels)
    ax.set_xlabel("Final model / review decision")
    ax.set_ylabel("Deterministic filter classification")
    ax.grid(False)
    fig.colorbar(image, ax=ax, pad=.02, label="Share within filter band (%)")
    return save_figure(fig, out_dir, "09_classification_audit", formats, dpi)


def build(
    input_path: Path,
    out_dir: Path,
    config: dict[str, object],
    system_type_profile: dict[str, object] | None = None,
    classifications_path: Path | None = None,
) -> list[Path]:
    """Generate the GitHub IDS catalogue plot and audit-table bundle."""
    LOG.info("building GitHub plots input=%s output=%s", input_path, out_dir)
    configure_style()
    frame = load_frame(
        input_path,
        (
            "github_repo_id", "project_id", "repository_relationship", "role", "observation_scopes", "observation_objects",
            "capabilities", "detection_methods", "language", "stars", "forks", "pushed_at",
        ),
    )
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    top_n = int(config.get("top_n", 15))
    outputs: list[Path] = []

    # 00 — keep the existing detailed repository overview unchanged.
    exact_commit = frame.get("last_commit_at", pd.Series("", index=frame.index)).fillna("")
    type_values = frame.get("system_types", pd.Series("unknown", index=frame.index)).replace("", "unknown")
    primary_types = _primary_system_type_series(frame)
    overview = pd.DataFrame({
        "repository": frame["full_name"],
        "system_role": frame["role"].map(ROLE_LABELS).fillna(frame["role"]),
        "primary_system_type": primary_types,
        "system_types": type_values,
        "system_type_names": frame.get("system_type_names", pd.Series("", index=frame.index)),
        "observation_scope": frame["observation_scopes"],
        "observation_objects": frame["observation_objects"],
        "capabilities": frame["capabilities"],
        "methods": frame["detection_methods"],
        "relationship": frame["repository_relationship"],
        "reporting_segment": frame["role"].map(lambda role: "operational_system" if role in OPERATIONAL_ROLES else "other_relevant_type"),
        "stars": frame["stars"],
        "forks": frame["forks"],
        "last_commit_at": exact_commit.where(exact_commit != "", frame["pushed_at"]),
        "last_commit_time_source": exact_commit.map(lambda value: "commit_metadata" if value else "repository_pushed_at_fallback"),
        "repository_url": frame.get("html_url", pd.Series("", index=frame.index)),
    })
    overview["_canonical"] = (~overview["relationship"].isin(CANONICAL_RELATIONSHIPS)).astype(int)
    overview = overview.sort_values(["_canonical", "system_role", "repository"], kind="stable").drop(columns="_canonical")
    write_table(overview, out_dir / "data", "00_repository_overview")
    write_table(_system_type_definitions(system_type_profile), out_dir / "data", "00_system_type_definitions")
    for stale in out_dir.glob("00_repository_overview_p*.*"):
        stale.unlink()
    outputs += _overview_matrix(frame, out_dir, formats, dpi)

    # 01 — creation timeline with stars and IDS-type colors.
    outputs += _repository_timeline(frame, out_dir, formats, dpi)

    # 02 — role x IDS-type overview matrix.
    outputs += _role_type_matrix(frame, out_dir, formats, dpi)

    # 03 — repository relationships, visually unchanged but with explicit audit CSVs.
    _write_count_table(frame, "repository_relationship", out_dir, "03_repository_relationships")
    outputs += count_plot(
        frame, "repository_relationship", "Repository relationships",
        metadata_caption(frame, "all accepted repository rows"), out_dir, "03_repository_relationships", formats, dpi,
    )

    # 04 — detection method x IDS-type matrix.
    outputs += _detection_type_matrix(frame, out_dir, formats, dpi)

    # 05 — activity bar chart, kept as before.
    frame["activity"] = frame["pushed_at"].map(_activity)
    _write_count_table(frame, "activity", out_dir, "05_activity")
    outputs += count_plot(
        frame, "activity", "Repository activity",
        metadata_caption(frame, "all accepted repositories; based on pushed_at at plot time"),
        out_dir, "05_activity", formats, dpi,
    )

    # 06 — expanded metadata completeness including GitHub topics/tags and taxonomy fields.
    outputs += _metadata_completeness(frame, out_dir, formats, dpi)

    # 07 — most common topics/tags.
    outputs += _tag_counts(frame, out_dir, formats, dpi, top_n)

    # 08 — discovery-source boxplot. Each source has a distribution of annual repository counts.
    outputs += _discovery_source_boxplot(frame, out_dir, formats, dpi)

    # 09 — deterministic filter band versus final model decision.
    if classifications_path and classifications_path.is_file():
        classifications = pd.read_csv(classifications_path, dtype=str, keep_default_na=False)
        outputs += _classification_audit(classifications, out_dir, formats, dpi)
        outputs += evidence_escalation_plot(classifications, out_dir, formats, dpi, "10_evidence_escalation", "GitHub")
        outputs += score_decision_plot(classifications, out_dir, formats, dpi, "11_keyword_score_vs_decision", "GitHub")
        outputs += _why_repositories_not_included(classifications, out_dir, formats, dpi)
    else:
        outputs += _classification_audit(frame, out_dir, formats, dpi)
        outputs += evidence_escalation_plot(frame, out_dir, formats, dpi, "10_evidence_escalation", "GitHub")
        outputs += score_decision_plot(frame, out_dir, formats, dpi, "11_keyword_score_vs_decision", "GitHub")
        outputs += _why_repositories_not_included(frame, out_dir, formats, dpi)

    outputs.append(write_index(out_dir, input_path, frame, outputs, [
        "00_repository_overview keeps the existing detailed repository feature matrix and exports every accepted row.",
        "01_repository_timeline reports operational systems (native detectors, classifier engines, and detection frameworks) using repository creation time on x, GitHub stars on y, and primary IDS type as color; every repository is named below its point.",
        "02_role_type_matrix reports every accepted role and crosses native detectors, classifier engines, frameworks, platforms, research projects, demos, content, and tooling with normalized IDS types such as NIDS/HIDS/PIDS.",
        "03_repository_relationships retains the existing repository-relationship count view.",
        "04_detection_method_type_matrix covers operational systems only and crosses detection methods with IDS types; both dimensions may be multi-label.",
        "05_activity retains the existing pushed_at activity-age bar chart.",
        "06_metadata_completeness includes GitHub topics/tags plus taxonomy and discovery metadata where available.",
        "07_common_tags reports the most common GitHub topics/tags and the share of repositories carrying each tag.",
        "08_discovery_sources uses annual per-source repository counts to form real boxplot distributions; creation year is used with pushed_at only as a fallback.",
        "09_classification_audit requires candidate/borderline/reject filter bands and a final model/review decision. Pass classifications_path when excluded candidates are stored separately from the accepted catalogue.",
        "10_evidence_escalation reports how often extra README/docs evidence was required, found, and beneficial; every bar states its denominator.",
        "11_keyword_score_vs_decision shows every scored item as a dot by final decision; the box is the middle 50%, its line is the median, and whiskers reach the furthest non-outlier within 1.5 x IQR. Dots beyond the whiskers are valid outliers; the companion summary exports the fences, actual whisker endpoints, and outlier count.",
        "12_why_not_included uses plain-language reasons for repositories absent from the catalogue; its definitions and records tables preserve each criterion, assignment basis, validation flag, exact model explanation, and evidence.",
        "Every plotted aggregate has a matching CSV under data/, and matrix/tag/discovery/classification figures also export row-level *_records.csv files for traceability.",
    ]))
    LOG.info("GitHub plot bundle complete files=%d output=%s", len(outputs), out_dir)
    return outputs
