from __future__ import annotations

import tempfile
import unittest
from xml.etree import ElementTree as ET
from pathlib import Path
from tests.fixtures import SYSTEM_TYPE_PROFILE

import pandas as pd

from tests.fixtures import write_csv as atomic_write_csv
from ids_meta.cli import build_parser
from ids_meta.plotting.common import add_heading, configure_style, percentage_bars, save_figure, stamp, subcaption
from ids_meta.github.schema import CATALOGUE_COLUMNS as GITHUB_COLUMNS
from ids_meta.openalex.schema import CATALOGUE_COLUMNS as OPENALEX_COLUMNS
from ids_meta.plotting.github import OPERATIONAL_ROLES, _github_why_not_included, _repository_density_data, _system_type_colors, build as build_github
from ids_meta.plotting.openalex import _cumulative_dimension, _openalex_why_not_included, build as build_openalex, prepare_frame, publication_timeline, timeline_citation_coordinate, spread_label_boxes, spread_label_coordinates
from ids_meta.plotting.compare import build as build_compare
from ids_meta.plotting.compare import build_models
from ids_meta.architecture import _draw_pipeline_figure, _draw_system_figure, build as build_architecture

from ids_meta.common.provenance import audited_bundle
save_figure = audited_bundle(save_figure)
publication_timeline = audited_bundle(publication_timeline)



class PlottingTests(unittest.TestCase):
    def test_architecture_labels_fit_inside_padded_card_areas(self) -> None:
        """Check rendered text bounds, including long labels and narrow icon rows."""
        import matplotlib.pyplot as plt

        for draw in (_draw_pipeline_figure, _draw_system_figure):
            with self.subTest(figure=draw.__name__):
                fig = draw()
                try:
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    ax = fig.axes[0]
                    checked = 0
                    for label in ax.texts:
                        gid = label.get_gid() or ""
                        if not gid.startswith("architecture-card-label:"):
                            continue
                        x, y, width, height = map(float, gid.split(":")[1:])
                        left, bottom = ax.transData.transform((x, y))
                        right, top = ax.transData.transform((x + width, y + height))
                        extent = label.get_window_extent(renderer)
                        self.assertGreaterEqual(extent.x0, left - .1, label.get_text())
                        self.assertGreaterEqual(extent.y0, bottom - .1, label.get_text())
                        self.assertLessEqual(extent.x1, right + .1, label.get_text())
                        self.assertLessEqual(extent.y1, top + .1, label.get_text())
                        self.assertGreaterEqual(label.get_fontsize(), 5.5)
                        checked += 1
                    self.assertGreaterEqual(checked, 14)
                    self.assertTrue(any("\n" in label.get_text() for label in ax.texts))
                finally:
                    plt.close(fig)

    def test_subcaptions_can_be_hidden_and_restored_without_hiding_plot_labels(self) -> None:
        """Toggle all explanatory text while retaining titles, axes and numeric labels."""
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for show in (False, True):
                    configure_style({"show_subcaptions": show})
                    fig = percentage_bars(
                        ["Approach"], [.4], title="Protected title", xlabel="Protected axis",
                        caption="Optional footer",
                    )
                    add_heading(fig, "Protected heading", "Optional subtitle")
                    stamp(fig, "Optional provenance")
                    subcaption(fig.axes[0], 50, 0, "Optional axis caption")
                    path = save_figure(fig, root, f"captions-{show}", ["svg"], 72)[0]
                    svg = path.read_text()
                    for text in ("Protected title", "Protected axis", "Protected heading", "Approach", "40.0%"):
                        self.assertIn(text, svg)
                    for text in ("Optional footer", "Optional subtitle", "Optional provenance", "Optional axis caption", "ids-meta-subcaption"):
                        self.assertEqual(text in svg, show)
                configure_style()
                fig = percentage_bars(["Default"], [0], title="Title", xlabel="Axis", caption="Restored default")
                svg = save_figure(fig, root, "default", ["svg"], 72)[0].read_text()
                self.assertIn("Restored default", svg)
                self.assertFalse(build_parser().parse_args(["--no-subcaptions", "compare", "github"]).subcaptions)
                self.assertTrue(build_parser().parse_args(["--subcaptions", "funnel", "openalex"]).subcaptions)
                with self.assertRaisesRegex(ValueError, "show_subcaptions"):
                    configure_style({"show_subcaptions": "false"})
        finally:
            configure_style()

    def test_model_distribution_preserves_counts_and_missing_results_on_common_denominator(self) -> None:
        """Merge adjacent accepts for display without changing exact agreement or hiding gaps."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for model, rows in {
                "model-a": [("W1", "accept"), ("W2", "accept_adjacent"), ("W3", "exclude"), ("W4", "uncertain")],
                "model-b": [("W1", "accept"), ("W2", "error"), ("W5", "unexpected")],
            }.items():
                path = root / f"{model}.csv"
                atomic_write_csv(path, [
                    {"openalex_id": identity, "decision": decision, 'classifier_supported_fields': '*'} for identity, decision in rows
                ], ["openalex_id", "decision"])
                inputs[model] = path
            output = root / "comparison"
            build_models("openalex", inputs, output, {"formats": ["svg"], "dpi": 72})
            shares = pd.read_csv(output / "data/01_model_decision_shares.csv")
            for _model, rows in shares.groupby("model"):
                self.assertEqual(int(rows["count"].sum()), 5)
                self.assertAlmostEqual(float(rows["share_percent"].sum()), 100)
                self.assertEqual(set(rows["total_sources"]), {5})
            by_disposition = shares.set_index(["model", "disposition"])
            self.assertEqual(int(by_disposition.loc[("model-a", "Accept"), "count"]), 2)
            self.assertEqual(float(by_disposition.loc[("model-a", "Accept"), "share_percent"]), 40)
            self.assertEqual(int(by_disposition.loc[("model-b", "Missing"), "count"]), 2)
            self.assertEqual(int(by_disposition.loc[("model-b", "Error"), "count"]), 1)
            self.assertEqual(int(by_disposition.loc[("model-b", "Other outcomes"), "count"]), 1)
            raw = pd.read_csv(output / "data/01_model_decision_counts.csv")
            self.assertEqual(int(raw["count"].sum()), 7)
            self.assertIn("accept_adjacent", set(raw["decision"]))
            agreement = pd.read_csv(output / "data/02_pairwise_agreement.csv")
            pair = agreement[(agreement["left_model"] == "model-a") & (agreement["right_model"] == "model-b")].iloc[0]
            self.assertEqual(int(pair["shared_sources"]), 2)
            self.assertEqual(float(pair["exact_agreement"]), .5)
            svg = (output / "01_model_decision_counts.svg").read_text(encoding="utf-8")
            self.assertIn("Share of 5 papers (%)", svg)
            self.assertIn("Accept", svg)
            self.assertIn("Exclude", svg)
            self.assertNotIn("Relevant", svg)
            self.assertIn("Missing", svg)

    def test_timeline_label_positions_are_collision_spread_and_bounded(self) -> None:
        """Keep dense timeline callouts separated without disconnecting their ordering."""
        positions = spread_label_coordinates([1.0, 1.0, 1.02, 1.5], 0.0, 2.0, .25)
        self.assertTrue(all(0.0 <= value <= 2.0 for value in positions))
        ordered = sorted(positions)
        self.assertTrue(all(right - left >= .249 for left, right in zip(ordered, ordered[1:])))

    def test_measured_timeline_labels_share_displacement_and_stay_separated(self) -> None:
        """Pack mixed-height names around their points instead of pushing a label fan upward."""
        heights = [.3, .6, .2, .4]
        centers = spread_label_boxes([2.0] * 4, heights, 0.0, 4.0, .15)
        self.assertAlmostEqual(sum(centers) / len(centers), 2.0)
        for center, height in zip(centers, heights):
            self.assertGreaterEqual(center - height / 2, 0.0)
            self.assertLessEqual(center + height / 2, 4.0)
        for index in range(3):
            self.assertGreaterEqual(centers[index + 1] - centers[index],
                                    (heights[index] + heights[index + 1]) / 2 + .15 - 1e-9)
        with self.assertRaisesRegex(ValueError, "more vertical space"):
            spread_label_boxes([1.0, 1.0], [2.0, 2.0], 0.0, 2.0, .1)

    def test_dense_timeline_stacks_labels_above_their_years(self) -> None:
        """Keep every label above its real publication year without overlap or scatter dots."""
        years = [2020] * 18 + [2021] * 12 + [2022] * 6 + [2023] * 2 + [2024] * 2
        rows = [
            {"openalex_id": f"W{index}", "artifact_name": f"Long network traffic resource {index}",
             "publication_year": year, "cited_by_count": ("0", "250", "9", "100", "100", "")[index % 6],
             "artifact_type": ("dataset", "generator_framework", "dataset_and_generator")[index % 3],
             "domain": "general_network", 'artifact_display_name': f"Long network traffic resource {index}", 'resource_name': f"Long network traffic resource {index}"}
            for index, year in enumerate(years)
        ]
        rows.append({**rows[0], "openalex_id": "Wold", "artifact_name": "Early resource",
                     "publication_year": 1997, "cited_by_count": 5000, 'artifact_display_name': "Early resource", 'resource_name': "Early resource"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication_timeline(prepare_frame(pd.DataFrame(rows)), root, ["svg"], 72)
            table = pd.read_csv(root / "data/00_publication_timeline.csv")
            svg_source = (root / "00_publication_timeline.svg").read_text(encoding="utf-8")
            svg = ET.fromstring(svg_source)
            svg_width = float(svg.get("width").removesuffix("pt"))
            svg_height = float(svg.get("height").removesuffix("pt"))
        self.assertEqual(len(table), len(rows))
        self.assertEqual(set(table["citations"]), {0, 9, 100, 250, 5000})
        self.assertEqual(set(table["panel_number"]), {1})
        self.assertEqual(set(table["panel_start_year"]), {1997})
        self.assertEqual(set(table["panel_end_year"]), {2024})
        self.assertEqual(set(table["timeline_x_offset"]), {0.0})
        self.assertTrue((table["timeline_x"] == table["publication_year"]).all())
        self.assertTrue(((table["citation_coordinate"] - table["citations"].map(timeline_citation_coordinate)).abs() < 1e-12).all())
        self.assertFalse(any(element.get("id", "").startswith("PathCollection") for element in svg.iter()))
        self.assertEqual([timeline_citation_coordinate(value) for value in (0, 50, 100, 150, 200, 250)],
                         [0, 50, 100, 150, 200, 250])
        self.assertAlmostEqual(timeline_citation_coordinate(251) - timeline_citation_coordinate(250), .025)
        self.assertEqual(timeline_citation_coordinate(5_000), 368.75)
        upper_steps = [timeline_citation_coordinate(value + 1000) - timeline_citation_coordinate(value)
                       for value in (1000, 2000, 3000, 4000)]
        self.assertEqual(upper_steps, [25.0] * 4)
        occupied_band = timeline_citation_coordinate(250) - timeline_citation_coordinate(0)
        self.assertGreater(occupied_band / (1.08 * timeline_citation_coordinate(5_000)), .60)
        self.assertIn("Uniform scale compression above 250", svg_source)
        self.assertEqual(svg_width / 72, 36.0)
        self.assertGreater(svg_width / svg_height, 2.0)
        self.assertEqual(set(table["label_side"]), {"above"})
        isolated = table[table["openalex_id"] == "Wold"].iloc[0]
        self.assertAlmostEqual(isolated["label_coordinate"], isolated["citation_coordinate"])
        self.assertTrue(table["label_adjusted_for_overlap"].any())
        self.assertTrue(((table["label_citation_offset"] - (table["label_coordinate"] - table["citation_coordinate"])).abs() < 1e-12).all())
        self.assertEqual(set(table["line_style"]), {"solid", "dashed", "dotted"})
        self.assertTrue((table["label_x"] == table["publication_year"]).all())
        self.assertTrue((table["label_lower_coordinate"] >= -1e-8).all())
        self.assertTrue((table["label_left_coordinate"] >= 1997 - .65).all())
        self.assertTrue((table["label_right_coordinate"] <= 2024 + .65).all())
        self.assertTrue((table["label_right_coordinate"] - table["label_left_coordinate"] < .9).all())
        for _year, group in table.groupby("publication_year"):
            ordered = group.sort_values("same_year_index")
            self.assertTrue((ordered["label_coordinate"].diff().dropna() > 0).all())
            self.assertTrue(ordered["citations"].is_monotonic_increasing)
            for _count, tied in ordered.groupby("citations", sort=False):
                self.assertEqual(list(tied["resource_name"]), sorted(tied["resource_name"]))
        for _, group in table.groupby("panel_number"):
            records = group.to_dict("records")
            for index, first in enumerate(records):
                for second in records[index + 1:]:
                    overlap_x = min(first["label_right_coordinate"], second["label_right_coordinate"]) - max(first["label_left_coordinate"], second["label_left_coordinate"])
                    overlap_y = min(first["label_upper_coordinate"], second["label_upper_coordinate"]) - max(first["label_lower_coordinate"], second["label_lower_coordinate"])
                    self.assertFalse(overlap_x > 1e-8 and overlap_y > 1e-8)

    def test_operational_reporting_roles_are_explicit(self) -> None:
        """Keep reporting limited to genuine operational approaches without hiding other roles."""
        self.assertEqual(OPERATIONAL_ROLES, {"native_detector", "classifier_engine", "detection_framework"})

    def test_repository_density_counts_unique_reporting_rows_and_missing_years(self) -> None:
        """Aggregate overlapping systems without losing unknown dates or including variants."""
        common = {"role": "native_detector", "repository_relationship": "canonical", "system_types": "NIDS"}
        frame = pd.DataFrame([
            {**common, "github_repo_id": "1", "full_name": "a/one", "primary_system_type": "NIDS", "created_at": "2020-01-01", "stars": "0"},
            {**common, "github_repo_id": "2", "full_name": "a/two", "primary_system_type": "nids", "created_at": "2020-01-01", "stars": "100"},
            {**common, "github_repo_id": "3", "full_name": "a/three", "primary_system_type": "HIDS", "created_at": "2022-01-01"},
            {**common, "github_repo_id": "4", "full_name": "a/undated", "primary_system_type": "unknown", "created_at": "invalid", "pushed_at": "2025-01-01"},
            {**common, "github_repo_id": "5", "full_name": "a/fork", "primary_system_type": "NIDS", "created_at": "2020-01-01", "repository_relationship": "fork"},
            {**common, "github_repo_id": "6", "full_name": "a/analyzer", "primary_system_type": "NSM", "created_at": "2020-01-01", "role": "traffic_analyzer"},
        ])
        records, counts = _repository_density_data(frame, profile=SYSTEM_TYPE_PROFILE)
        bins = counts.set_index(["primary_system_type", "creation_year"])["count"]
        self.assertEqual(int(counts["count"].sum()), 4)
        self.assertEqual(int(bins.loc[("NIDS", "2020")]), 2)
        self.assertEqual(int(bins.loc[("NIDS", "2021")]), 0)
        self.assertEqual(int(bins.loc[("HIDS", "2022")]), 1)
        self.assertEqual(int(bins.loc[("unknown", "Unknown")]), 1)
        self.assertEqual(records.set_index("github_repo_id").loc["4", "date_status"], "missing_created_at")
        self.assertNotIn("2025", set(counts["creation_year"]))
        self.assertEqual(_system_type_colors(["NIDS"])["NIDS"], _system_type_colors(["HIDS", "NIDS"])["NIDS"])

    def test_why_not_included_labels_are_plain_and_auditable(self) -> None:
        """Use reader-facing reasons while retaining how each reason was assigned."""
        repository_category, repository_basis, repository_explanation = _github_why_not_included({
            "decision": "exclude", "primary_reason": "This is a tutorial for configuring an existing detector.",
        })
        self.assertEqual(repository_category, "Tutorial or lab only — no reusable implementation")
        self.assertEqual(repository_basis, "Model explanation contains: tutorial")
        self.assertIn("without a reusable IDS contribution", repository_explanation)
        paper_category, paper_basis, paper_explanation = _openalex_why_not_included({
            "decision": "exclude", "paper_role": "survey",
        })
        self.assertEqual(paper_category, "Survey only — no new dataset or generator")
        self.assertEqual(paper_basis, "Structured paper role: survey")
        self.assertIn("reviews existing resources", paper_explanation)
        uncertain_category, uncertain_basis, uncertain_explanation = _github_why_not_included({
            "decision": "uncertain", "classification_validation_flags": "framework_architecture_missing;verified_evidence_missing",
        })
        self.assertEqual(uncertain_category, "Could not verify a framework architecture")
        self.assertEqual(uncertain_basis, "Automatic validation check: framework_architecture_missing")
        self.assertIn("cohesive agent-manager", uncertain_explanation)

    def test_cumulative_timeline_adds_categories_over_time(self) -> None:
        """Cumulative type and domain series must sum to the catalogue total each year."""
        dated = pd.DataFrame([
            {"publication_year": 2020, "artifact_type": "dataset", "domain": "general_network"},
            {"publication_year": 2022, "artifact_type": "dataset_and_generator", "domain": "iot_iiot"},
        ])
        type_table, years, categories, cumulative = _cumulative_dimension(
            dated, "artifact_type", ["dataset", "dataset_and_generator"],
        )
        self.assertEqual(years, [2020, 2021, 2022])
        self.assertEqual(categories, ["dataset", "dataset_and_generator"])
        self.assertEqual(int(cumulative.loc[2021].sum()), 1)
        self.assertEqual(int(cumulative.loc[2022].sum()), 2)
        self.assertEqual(set(type_table["dimension"]), {"resource_type"})
        domain_table, _, _, domain_cumulative = _cumulative_dimension(
            dated, "domain", ["general_network", "iot_iiot"],
        )
        self.assertEqual(int(domain_cumulative.loc[2022].sum()), 2)
        self.assertEqual(set(domain_table["dimension"]), {"security_domain"})

    def test_offline_plot_smoke_and_plot_data(self) -> None:
        """Verify both plot bundles render offline with matching data tables."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_csv = root / "github.csv"
            github_row = {column: "" for column in GITHUB_COLUMNS}
            github_row.update({"schema_version": "1.0", "run_id": "run", "github_repo_id": "1", "project_id": "p1", "canonical_repo_id": "1", "repository_relationship": "canonical", "full_name": "example/repo", "role": "native_detector", "observation_scopes": "network", "observation_objects": "packet_analysis", "capabilities": "detection", "detection_methods": "signature_based", "primary_system_type": "NIDS", "system_types": "NIDS", "system_type_names": "Network intrusion detection system", "language": "Python", "stars": "5", "forks": "1", "pushed_at": "2025-01-01T00:00:00Z", "html_url": "https://example.invalid", "evidence_json": "[]", "prefilter_band": "candidate", "prefilter_score": "10", "decision": "accept", "evidence_expansion_required": "true", "evidence_expansion_attempted": "true", "evidence_expansion_changed": "true", "evidence_expansion_improved": "true", "evidence_expansion_resolved": "true", 'classifier_supported_fields': '*'})
            analyzer_row = {**github_row, "github_repo_id": "2", "project_id": "p2", "canonical_repo_id": "2",
                            "full_name": "example/analyzer", "role": "traffic_analyzer", "capabilities": "monitoring",
                            "detection_methods": "unknown", "primary_system_type": "NSM", "system_types": "NSM",
                            "system_type_names": "Network security monitoring / traffic analysis system"}
            antivirus_row = {**github_row, "github_repo_id": "3", "project_id": "p3", "canonical_repo_id": "3",
                             "full_name": "example/file-scanner", "role": "antivirus",
                             "observation_scopes": "host_endpoint", "observation_objects": "file_analysis",
                             "primary_system_type": "AV", "system_types": "AV",
                             "system_type_names": "Antivirus / malware file scanner"}
            atomic_write_csv(github_csv, [github_row, analyzer_row, antivirus_row], GITHUB_COLUMNS)
            outputs = build_github(github_csv, root / "github-plots", {"formats": ["svg", "pdf"], "dpi": 72, "top_n": 5}, classifications_path=github_csv, system_type_profile=SYSTEM_TYPE_PROFILE)
            self.assertTrue(outputs)
            self.assertTrue((root / "github-plots/data/01_repository_timeline.csv").is_file())
            overview = root / "github-plots/data/00_repository_overview.csv"
            self.assertTrue(overview.is_file())
            self.assertIn("last_commit_at", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("primary_system_type", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "github-plots/data/00_system_type_definitions.csv").is_file())
            self.assertTrue((root / "github-plots/data/02_role_type_matrix.csv").is_file())
            role_matrix = (root / "github-plots/data/02_role_type_matrix.csv").read_text(encoding="utf-8")
            self.assertIn("traffic_analyzer", role_matrix)
            self.assertIn("NSM", role_matrix)
            self.assertIn("antivirus", role_matrix)
            self.assertIn("AV", role_matrix)
            overview_rows = pd.read_csv(overview, keep_default_na=False).set_index("repository")
            self.assertEqual(overview_rows.loc["example/file-scanner", "reporting_segment"], "other_relevant_type")
            self.assertEqual(overview_rows.loc["example/file-scanner", "system_role"], "Antivirus / file scanner")
            self.assertTrue((root / "github-plots/data/01_repository_timeline_records.csv").is_file())
            for name in ("01_repository_timeline_records", "04_detection_method_type_matrix_records"):
                self.assertNotIn("example/file-scanner", (root / "github-plots/data" / f"{name}.csv").read_text())
            self.assertIn("Antivirus / file scanner", (root / "github-plots/00_repository_overview.svg").read_text())
            self.assertTrue((root / "github-plots/00_repository_overview.svg").is_file())
            self.assertTrue((root / "github-plots/00_repository_overview.pdf").is_file())
            self.assertTrue((root / "github-plots/10_evidence_escalation.svg").is_file())
            self.assertIn("denominator_scope", (root / "github-plots/data/10_evidence_escalation.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "github-plots/11_keyword_score_vs_decision.svg").is_file())
            self.assertTrue((root / "github-plots/12_why_not_included.svg").is_file())
            self.assertTrue((root / "github-plots/data/12_why_not_included_records.csv").is_file())
            github_reason_header = (root / "github-plots/data/12_why_not_included_records.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("primary_reason", github_reason_header)
            self.assertIn("how_reason_was_assigned", github_reason_header)
            github_score_header = (root / "github-plots/data/11_keyword_score_vs_decision.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("score_median", github_score_header)
            self.assertIn("lower_whisker", github_score_header)
            self.assertIn("upper_whisker", github_score_header)
            self.assertIn("outlier_count", github_score_header)
            self.assertIn("llm_decision", (root / "github-plots/data/11_keyword_score_vs_decision_records.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse((root / "github-plots/00_repository_overview_p01.svg").exists())
            comparison = build_compare("github", github_csv, root / "comparison", {"formats": ["svg"], "dpi": 72})
            self.assertTrue(comparison)
            second_csv = root / "github-second.csv"
            second_row = dict(github_row)
            second_row["decision"] = "exclude"
            atomic_write_csv(second_csv, [second_row], GITHUB_COLUMNS)
            model_outputs = build_models("github", {"model-a": github_csv, "model-b": second_csv}, root / "model-comparison", {"formats": ["svg"], "dpi": 72, "top_n": 5})
            self.assertTrue(model_outputs)
            self.assertTrue((root / "model-comparison/data/02_pairwise_agreement.csv").is_file())
            self.assertTrue((root / "model-comparison/data/03_disagreement_records.csv").is_file())

            openalex_csv = root / "openalex.csv"
            paper = {column: "" for column in OPENALEX_COLUMNS}
            paper.update({"schema_version": "1.0", "run_id": "run", "openalex_id": "W1", "artifact_id": "a1", "artifact_name": "Fixture trace corpus", "artifact_display_name": "FTC", "publication_family_id": "f1", "canonical_openalex_id": "W1", "paper_relationship": "canonical_base", "artifact_type": "dataset", "publication_year": "2024", "domain": "general_network", "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled", "cited_by_count": "42", "citation_as_of": "2026-08-07", "discovery_relations": "direct_search;survey_reference:W9", "prefilter_score": "11", "prefilter_band": "candidate", "decision": "accept", "paper_role": "original_release", "evidence_expansion_required": "true", "evidence_expansion_attempted": "true", "evidence_expansion_changed": "true", "evidence_expansion_improved": "true", "evidence_expansion_resolved": "true", 'classifier_supported_fields': '*'})
            atomic_write_csv(openalex_csv, [paper], OPENALEX_COLUMNS)
            stale = root / "openalex-plots/08_publication_timeline.svg"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale", encoding="utf-8")
            outputs = build_openalex(openalex_csv, None, root / "openalex-plots", {"formats": ["svg"], "dpi": 72}, openalex_csv, openalex_csv)
            self.assertTrue(outputs)
            timeline = root / "openalex-plots/data/00_publication_timeline.csv"
            provenance = root / "openalex-plots/data/02_discovery_provenance.csv"
            self.assertTrue(timeline.is_file())
            self.assertTrue(provenance.is_file())
            provenance_header = provenance.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("citing_surveys", provenance_header)
            self.assertNotIn("survey_source", provenance_header)
            contribution = (root / "openalex-plots/data/10_discovery_source_contribution.csv").read_text(encoding="utf-8")
            self.assertNotIn("Survey source", contribution)
            self.assertNotIn("Other", contribution)
            citation_table = root / "openalex-plots/data/01_citation_landscape.csv"
            self.assertIn("citations", citation_table.read_text(encoding="utf-8").splitlines()[0])
            timeline_header = timeline.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("resource_name", timeline_header)
            self.assertIn("citation_coordinate", timeline_header)
            self.assertIn("timeline_x_offset", timeline_header)
            self.assertIn("label_coordinate", timeline_header)
            self.assertIn("survey_snowball", provenance.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "openalex-plots/00_publication_timeline.svg").is_file())
            timeline_svg = (root / "openalex-plots/00_publication_timeline.svg").read_text(encoding="utf-8")
            self.assertIn("Citations (linear to 250; uniformly compressed above)", timeline_svg)
            self.assertIn("Publication year", timeline_svg)
            self.assertIn("Dataset domain (color)", timeline_svg)
            self.assertIn("Dataset / generator (line style)", timeline_svg)
            self.assertTrue((root / "openalex-plots/01_citation_landscape.svg").is_file())
            self.assertTrue((root / "openalex-plots/03_cumulative_growth_by_type_and_domain.svg").is_file())
            cumulative_data = (root / "openalex-plots/data/03_cumulative_growth_by_type_and_domain.csv").read_text(encoding="utf-8")
            self.assertIn("cumulative_all_resources", cumulative_data.splitlines()[0])
            self.assertIn("resource_type", cumulative_data)
            self.assertIn("security_domain", cumulative_data)
            self.assertTrue((root / "openalex-plots/07_classification_audit.svg").is_file())
            self.assertTrue((root / "openalex-plots/08_survey_sources.svg").is_file())
            self.assertTrue((root / "openalex-plots/09_evidence_escalation.svg").is_file())
            self.assertIn("denominator_scope", (root / "openalex-plots/data/09_evidence_escalation.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "openalex-plots/11_keyword_score_vs_decision.svg").is_file())
            self.assertTrue((root / "openalex-plots/12_why_not_included.svg").is_file())
            self.assertTrue((root / "openalex-plots/data/12_why_not_included_records.csv").is_file())
            paper_reason_header = (root / "openalex-plots/data/12_why_not_included_records.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("primary_reason", paper_reason_header)
            self.assertIn("how_reason_was_assigned", paper_reason_header)
            paper_score_header = (root / "openalex-plots/data/11_keyword_score_vs_decision.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("score_median", paper_score_header)
            self.assertIn("lower_whisker", paper_score_header)
            self.assertIn("upper_whisker", paper_score_header)
            self.assertIn("outlier_count", paper_score_header)
            self.assertIn("llm_decision", (root / "openalex-plots/data/11_keyword_score_vs_decision_records.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(stale.exists())
            architecture = build_architecture(root, ["svg"], 72)
            self.assertTrue(architecture)
            self.assertTrue((root / "pipeline_architecture.svg").is_file())
            self.assertTrue((root / "system_architecture.svg").is_file())
            self.assertFalse((root / "github-plots/pipeline_sequence.svg").exists())
            self.assertFalse((root / "openalex-plots/pipeline_sequence.svg").exists())


if __name__ == "__main__":
    unittest.main()
