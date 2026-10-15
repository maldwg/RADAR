from __future__ import annotations

import argparse
from collections import Counter
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import pandas as pd

from radar.cli import _evaluation_jobs, _execute_github, _execute_openalex, _execution_paths, _variant_paths, build_parser
from radar.common.config import Settings, load_settings, load_system_type_profile
from radar.common.csvio import read_csv
from tests.fixtures import write_csv as atomic_write_csv
from radar.common.evaluation import evaluate_results, load_rating_profile
from radar.common.ollama import decision_profile
from radar.github.schema import AUDIT_ROLES, CAPABILITIES, METHODS, OBSERVATION_OBJECTS, SCOPES, system_type_codes
from radar.openalex.schema import ARTIFACT_TYPES, DATA_FORMS, DOMAINS, ENVIRONMENTS, LABELING
from radar.plotting.evaluation import _recovered_taxonomy_accuracy, _reference_uncertainty, _taxonomy_accuracy_overview, build as build_evaluation_plots
from radar.plotting.compare import _reference_uncertainty_overview
from radar.plotting.common import configure_style
from radar.plotting.recovery import recovered_taxonomy_table
from radar.plotting.uncertainty import reference_uncertainty

from radar.common.provenance import audited_bundle
_recovered_taxonomy_accuracy = audited_bundle(_recovered_taxonomy_accuracy)
_reference_uncertainty = audited_bundle(_reference_uncertainty)
_taxonomy_accuracy_overview = audited_bundle(_taxonomy_accuracy_overview)
_reference_uncertainty_overview = audited_bundle(_reference_uncertainty_overview)



ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    """Exercise centrally rated evaluation for both discovery sources."""

    def test_reference_uncertainty_separates_attempts_errors_and_missing_references(self) -> None:
        """Count survey fallback attempts once and keep retrieval gaps out of the rate."""
        columns = ["classification_present", "actual_disposition", "status", "survey_reference_classification_present", "survey_reference_actual_disposition"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"classification_present": "true", "actual_disposition": disposition, "status": status}
                for disposition, status in [("include", "classified"), ("exclude", "classified"), ("uncertain", "uncertain"), ("uncertain", "error")]
            ]
            rows[0].update(survey_reference_classification_present="true", survey_reference_actual_disposition="uncertain")
            rows += [
                {"classification_present": "false", "survey_reference_classification_present": "true", "survey_reference_actual_disposition": disposition, "status": status}
                for disposition, status in [("uncertain", "survey_reference_classification_uncertain"), ("unavailable", "survey_reference_classification_error"), ("exclude", "survey_reference_classified_nonrelease")]
            ]
            rows += [{"classification_present": "false", "status": status} for status in ["survey_reference_not_retrieved", "survey_reference_retrieved_not_classified"]]
            atomic_write_csv(root / "instances.csv", rows, columns)
            counts = reference_uncertainty(root)
            self.assertEqual(counts["attempted"], 7)
            self.assertEqual(counts["uncertain"], 2)
            self.assertEqual(counts["error"], 2)
            self.assertEqual(counts["exclude"], 2)
            self.assertEqual(counts["accept"], 1)
            self.assertEqual(counts["not_attempted"], 2)
            self.assertAlmostEqual(counts["value"], 2 / 7)
            configure_style({"show_subcaptions": False})
            try:
                _reference_uncertainty(root, {"source": "openalex"}, root / "figures", ["svg"], 72)
                svg = (root / "figures/08_reference_uncertainty.svg").read_text()
                for label in ["28.6%", "2/7 attempts", "Uncertain decisions among attempted reference classifications (%)"]:
                    self.assertIn(label, svg)
            finally:
                configure_style()

    def test_reference_uncertainty_shows_no_attempts_as_unavailable_not_zero(self) -> None:
        """Keep missing classification attempts distinct from zero observed uncertainty."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for label, rows in {
                "Available": [{"classification_present": "true", "actual_disposition": "exclude", "status": "classified"}],
                "Missing": [{"classification_present": "false", "actual_disposition": "unavailable", "status": "not_retrieved"}],
            }.items():
                inputs[label] = root / label
                atomic_write_csv(inputs[label] / "instances.csv", rows, ["classification_present", "actual_disposition", "status"])
            self.assertIsNone(reference_uncertainty(inputs["Missing"])["value"])
            _reference_uncertainty_overview("github", inputs, root, ["svg"], 72)
            table = pd.read_csv(root / "data/02_reference_uncertainty.csv").set_index("classifier")
            self.assertEqual(table.loc["Available", "value"], 0)
            self.assertTrue(pd.isna(table.loc["Missing", "value"]))
            svg = (root / "02_reference_uncertainty.svg").read_text()
            for label in ["0.0%", "N/A", "0/1 attempts", "0/0 attempts"]:
                self.assertIn(label, svg)

    def test_recovered_taxonomy_uses_only_accepted_reference_positives(self) -> None:
        """Separate 72/80 labeling quality from 80/100 recovery and negative controls."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_csv(root / "field_metrics.csv", [
                {"field": field, "supported": supported, "comparisons": count}
                for field, supported, count in [
                    ("artifact_type", True, 101), ("domain", True, 101),
                    ("data_form", True, 100), ("labeling", False, 0),
                ]
            ], ["field", "supported", "comparisons"])
            instances = []
            for index in range(100):
                checks = {"artifact_type": index < 72, "domain": False}
                if index < 60:
                    checks["data_form"] = True
                instances.append({
                    "expected_disposition": "include",
                    "actual_disposition": "include" if index < 80 else "uncertain" if index < 90 else "exclude",
                    "field_results_json": json.dumps(checks),
                })
            instances.append({
                "expected_disposition": "exclude", "actual_disposition": "include",
                "field_results_json": json.dumps({"artifact_type": True, "domain": True}),
            })
            atomic_write_csv(root / "instances.csv", instances, [
                "expected_disposition", "actual_disposition", "field_results_json",
            ])
            table = recovered_taxonomy_table(root).set_index("field")
            self.assertEqual(table.loc["artifact_type", "comparisons"], 80)
            self.assertEqual(table.loc["artifact_type", "exact_matches"], 72)
            self.assertAlmostEqual(table.loc["artifact_type", "value"], .9)
            self.assertEqual(table.loc["domain", "value"], 0)
            self.assertEqual(table.loc["data_form", "comparisons"], 60)
            self.assertEqual(table.loc["data_form", "value"], 1)
            self.assertTrue(pd.isna(table.loc["labeling", "value"]))
            self.assertTrue((table["recovered"] == 80).all())
            self.assertTrue((table["expected_positive_entities"] == 100).all())
            self.assertTrue((table["recovery_rate"] == .8).all())
            configure_style({"show_subcaptions": False})
            try:
                _recovered_taxonomy_accuracy(root, {"source": "openalex"}, root, ["svg"], 72)
                svg = (root / "07_recovered_taxonomy_accuracy.svg").read_text()
                for label in ["90.0%", "0.0%", "N/A", "n=80", "n=60", "recovered 80/100"]:
                    self.assertIn(label, svg)
                self.assertNotIn("radar-subcaption", svg)
            finally:
                configure_style()

            # Empty recovery is undefined accuracy, even on a supported field.
            for row in instances:
                row["actual_disposition"] = "unavailable"
            atomic_write_csv(root / "instances.csv", instances, list(instances[0]))
            empty = recovered_taxonomy_table(root)
            self.assertTrue(empty["value"].isna().all())
            self.assertTrue((empty["comparisons"] == 0).all())
            _recovered_taxonomy_accuracy(root, {"source": "github"}, root, ["svg"], 72)
            self.assertIn("recovered 0/100", (root / "07_recovered_taxonomy_accuracy.svg").read_text())

    def test_taxonomy_overview_keeps_full_field_counts_and_unsupported_na(self) -> None:
        """Use rated comparisons rather than recovered items and retain unsupported fields."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_csv(root / "field_metrics.csv", [
                {"field": "artifact_type", "supported": "true", "comparisons": 3, "exact_matches": 1, "exact_accuracy": 1 / 3},
                {"field": "domain", "supported": "true", "comparisons": 3, "exact_matches": 0, "exact_accuracy": 0},
                {"field": "data_form", "supported": "false", "comparisons": 0, "exact_matches": 0, "exact_accuracy": ""},
            ], ["field", "supported", "comparisons", "exact_matches", "exact_accuracy"])
            _taxonomy_accuracy_overview(root, {"source": "openalex", "classifier_id": "ensemble"}, root, ["svg"], 72)
            rows = {row["field"]: row for row in read_csv(root / "data/06_taxonomy_accuracy.csv")}
            self.assertAlmostEqual(float(rows["artifact_type"]["percentage"]), 100 / 3)
            self.assertEqual(float(rows["domain"]["value"]), 0)
            self.assertEqual(rows["data_form"]["value"], "")
            svg = (root / "06_taxonomy_accuracy.svg").read_text()
            for label in ("33.3%", "0.0%", "N/A", "3 rated entities per field"):
                self.assertIn(label, svg)

    def _write_profile(self, path: Path, source: str, body: str) -> Path:
        """Write a valid evaluation profile with caller-supplied entity tables."""
        fields = (
            '["role", "observation_scopes", "observation_objects", "capabilities", "detection_methods", "primary_system_type", "system_types"]'
            if source == "github"
            else '["paper_role", "relevance_tier", "artifact_type", "domain", "data_form", "capture_environment", "labeling"]'
        )
        path.write_text(
            f'schema_version = "1.0"\nsource = "{source}"\nclassification_fields = {fields}\n\n{body}',
            encoding="utf-8",
        )
        return path

    def test_ghml23_profile_preserves_upstream_ground_truth_and_crosswalk(self) -> None:
        """Keep the upstream snapshot and explicit project-taxonomy crosswalk auditable."""
        path = ROOT / "config/studies/ids_repositories/evaluation/references.toml"
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        profile = load_rating_profile(path, "github")
        rows = raw["entities"]
        self.assertNotIn("entity_source", raw)
        self.assertEqual(len(rows), 120)
        self.assertEqual(len(profile.entities), 120)
        self.assertEqual(Counter(row["benchmark_category"] for row in rows), {
            "secumon": 100, "imageocr": 10, "pentest": 10,
        })
        self.assertEqual(Counter(row["decision"] for row in rows), {"accept": 71, "exclude": 47, "uncertain": 2})
        self.assertEqual(raw["crosswalk_version"], "ghml23-to-radar-v10")
        self.assertEqual(raw["taxonomy_revision"], "antivirus-file-scanning-v6")
        self.assertIn("what concrete evidence", raw["taxonomy_semantics"]["observation_objects"].casefold())
        self.assertIn("how", raw["taxonomy_semantics"]["detection_methods"].casefold())
        self.assertIn("not derived", raw["taxonomy_semantics"]["system_types"].casefold())
        self.assertEqual(profile.classification_fields, [
            "role", "observation_scopes", "observation_objects", "capabilities", "detection_methods",
            "primary_system_type", "system_types", "centralization",
        ])
        configured_types = set(system_type_codes(load_system_type_profile(load_settings(ROOT / "config/studies/ids_repositories/study.toml"), "github")))
        for row in rows:
            self.assertIn(row["role"], AUDIT_ROLES)
            self.assertLessEqual(set(row["observation_scopes"]), set(SCOPES))
            self.assertLessEqual(set(row["observation_objects"]), set(OBSERVATION_OBJECTS))
            self.assertLessEqual(set(row["capabilities"]), set(CAPABILITIES))
            self.assertLessEqual(set(row["detection_methods"]), set(METHODS))
            self.assertLessEqual(set(row["system_types"]), configured_types)
            for field in ("observation_scopes", "observation_objects", "detection_methods", "system_types"):
                self.assertFalse("unknown" in row[field] and len(row[field]) > 1, (row["repository"], field))
            self.assertFalse((set(row["observation_objects"]) - {"unknown"}) & set(row["detection_methods"]), row["repository"])
            self.assertIn(row["primary_system_type"], row["system_types"], row["repository"])
        by_repository = {row["repository"]: row for row in rows}
        for repository in ("airbnb/binaryalert", "cloudina/hawk", "Neo23x0/Loki"):
            self.assertEqual(by_repository[repository]["role"], "antivirus")
            self.assertEqual(by_repository[repository]["system_types"], ["AV"])
            self.assertEqual(by_repository[repository]["observation_objects"], ["file_analysis"])
            self.assertTrue(by_repository[repository]["annotation_sources"])
        self.assertIn("prevention", by_repository["cloudina/hawk"]["capabilities"])
        self.assertEqual(by_repository["Neo23x0/signature-base"]["role"], "rules_content")
        self.assertEqual(by_repository["Neo23x0/signature-base"]["system_types"], ["AV"])
        self.assertEqual(by_repository["ajaykumarkk/CyberDoc"]["role"], "demo")
        self.assertEqual(by_repository["ajaykumarkk/CyberDoc"]["system_types"], ["AV"])
        self.assertEqual(by_repository["0xN3utr0n/Kanis"]["observation_objects"], ["system_calls"])
        self.assertNotIn("system_calls", by_repository["0xN3utr0n/Kanis"]["detection_methods"])
        for repository in ("zeek/zeek", "arkime/arkime", "cisco/joy", "cisco/mercury", "USArmyResearchLab/Dshell"):
            self.assertEqual(by_repository[repository]["role"], "traffic_analyzer")
            self.assertEqual(by_repository[repository]["system_types"], ["NSM"])
        self.assertEqual(by_repository["zeek/zeek"]["observation_objects"], ["packet_analysis"])
        # Installation on an endpoint and firewall responses do not imply HIDS/HIPS.
        for repository, primary, types in (
            ("fail2ban/fail2ban", "IPS", ["IPS", "IDS"]),
            ("marshyski/sshwatch", "IPS", ["IPS", "IDS"]),
            ("austin-taylor/flare", "IDS", ["IDS"]),
            ("evilsocket/opensnitch", "IPS", ["IPS"]),
            ("crowdsecurity/crowdsec", "IPS", ["IPS", "IDS", "APIDS"]),
        ):
            with self.subTest(repository=repository):
                self.assertEqual(by_repository[repository]["primary_system_type"], primary)
                self.assertEqual(by_repository[repository]["system_types"], types)
                self.assertTrue(by_repository[repository]["annotation_sources"])
        self.assertEqual(by_repository["fail2ban/fail2ban"]["observation_scopes"], ["host_endpoint"])
        self.assertEqual(by_repository["evilsocket/opensnitch"]["observation_objects"], ["packet_analysis"])
        self.assertEqual(by_repository["mitre-attack/bzar"]["observation_objects"], ["packet_analysis"])
        self.assertEqual(by_repository["mitre-attack/bzar"]["system_types"], ["PIDS", "NIDS"])
        for repository in ("ossec/ossec-hids", "wazuh/wazuh", "Velocidex/velociraptor",
                           "ION28/BLUESPAWN", "ion-storm/sysmon-edr", "rabbitstack/fibratus"):
            self.assertIn("HIPS", by_repository[repository]["system_types"], repository)
        for repository in ("ait-aecid/logdata-anomaly-miner", "Qianlitp/WatchAD", "sans-blue-team/DeepBlueCLI"):
            self.assertEqual(by_repository[repository]["system_types"], ["HIDS"], repository)
        for repository in ("OISF/suricata", "snort3/snort3"):
            self.assertIn("NIPS", by_repository[repository]["system_types"], repository)
        self.assertEqual(by_repository["dreadl0ck/netcap"]["role"], "detection_framework")
        self.assertEqual({row["repository"] for row in rows if "flow_analysis" in row["observation_objects"]}, {
            "arkime/arkime", "austin-taylor/flare", "cisco/joy", "corelight/zeek2es", "Cyb3rWard0g/HELK",
            "dreadl0ck/netcap", "ntop/ntopng", "Security-Onion-Solutions/securityonion",
        })
        self.assertEqual(by_repository["ossec/ossec-hids"]["role"], "detection_framework")
        self.assertEqual(by_repository["wazuh/wazuh"]["role"], "detection_framework")
        self.assertEqual(by_repository["Security-Onion-Solutions/securityonion"]["role"], "detection_platform")
        self.assertEqual(by_repository["kismetwireless/kismet"]["primary_system_type"], "WIDS")
        self.assertNotIn("PIDS", by_repository["kismetwireless/kismet"]["system_types"])
        self.assertEqual(by_repository["cisco/mercury"]["primary_system_type"], "NSM")
        self.assertEqual({row["repository"] for row in rows if row["primary_system_type"] == "SM"}, {
            "Cyb3rWard0g/HELK", "droe/xnumon", "jmpsec/osctrl", "olafhartong/ThreatHunting",
        })
        self.assertEqual(by_repository["StamusNetworks/SELKS"]["role"], "support_tooling")
        self.assertEqual(by_repository["droe/xnumon"]["role"], "support_tooling")
        self.assertEqual(by_repository["BlueTeamLabs/sentinel-attack"]["decision"], "uncertain")
        self.assertEqual(by_repository["alphasoc/nfr"]["decision"], "uncertain")
        self.assertEqual(by_repository["OWASP/Nettacker"]["decision"], "exclude")
        self.assertEqual(by_repository["JaidedAI/EasyOCR"]["decision"], "exclude")

    def test_ashraf_2026_profile_preserves_requested_table_2_dataset_types(self) -> None:
        """Keep the selected survey artifacts source-traceable and the taxonomy exact."""
        path = ROOT / "config/studies/ids_datasets/evaluation/references.toml"
        default_settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        dataset_settings = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        reference_sets = default_settings.section("evaluation")["reference_sets"] + dataset_settings.section("evaluation")["reference_sets"]
        configured = {item["name"]: item for item in reference_sets}
        self.assertEqual(configured["ghml23"]["profile"], str(ROOT / "config/studies/ids_repositories/evaluation/references.toml"))
        self.assertEqual(configured["shield"]["profile"], str(path))
        self.assertEqual(default_settings.root / configured["shield"]["profile"], path)
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        rows = raw["entities"]
        self.assertEqual(raw["evaluation_unit"], "artifact")
        self.assertEqual(raw["source_claimed_dataset_count"], 124)
        self.assertEqual(raw["source_extracted_dataset_count"], 123)
        self.assertEqual(raw["source_missing_row_numbers"], [122])
        self.assertEqual(raw["selected_dataset_count"], 105)
        self.assertEqual(raw["excluded_dataset_count"], 18)
        self.assertEqual(raw["excluded_survey_categories"], ["Botnet", "IOV", "MANET network", "Satellites"])
        self.assertEqual(len(rows), 105)
        self.assertNotIn("centralization", raw["classification_fields"])
        self.assertTrue(all("centralization" not in row for row in rows))
        self.assertEqual(Counter(row["decision"] for row in rows), {"accept": 96, "accept_adjacent": 5, "exclude": 4})
        self.assertTrue({row["artifact_type"] for row in rows}.issubset(set(ARTIFACT_TYPES)))
        self.assertEqual({row["artifact_type"] for row in rows}, {"dataset"})
        requested_domains = {
            "general_network", "host_endpoint", "iot_iiot", "cloud", "mobile_android",
            "electrical_smartgrid", "uav", "vpn",
        }
        self.assertEqual(set(DOMAINS), requested_domains | {"unknown"})
        self.assertTrue({row["data_form"] for row in rows}.issubset(set(DATA_FORMS)))
        self.assertTrue({row["capture_environment"] for row in rows}.issubset(set(ENVIRONMENTS)))
        self.assertTrue({row["labeling"] for row in rows}.issubset(set(LABELING)))
        self.assertEqual({row["domain"] for row in rows}, requested_domains | {"unknown"})
        by_name = {row["artifact"]: row for row in rows}
        self.assertNotIn("Kitsune Network", by_name)
        self.assertEqual(by_name["ISOT-CID"]["domain"], "cloud")
        self.assertEqual(by_name["Android Malware"]["domain"], "mobile_android")
        self.assertEqual(by_name["UAV-NIDD"]["domain"], "uav")
        self.assertEqual(by_name["VPN-NONVPN"]["domain"], "vpn")
        self.assertEqual(by_name["CIC-IDS 2017"]["aliases"], ["CICIDS2017", "CIC-IDS2017"])
        self.assertEqual(by_name["TON-IOT"]["survey_category"], "IOT/IIOT")
        self.assertEqual(by_name["LBNL"]["survey_category"], "Electrical network")
        self.assertEqual(by_name["LBNL"]["domain"], "general_network")
        self.assertEqual(by_name["LBNL"]["decision"], "accept_adjacent")
        self.assertEqual(by_name["MAWI"]["data_form"], "packet")
        self.assertEqual(by_name["NSL-KDD"]["data_form"], "tabular_features")
        self.assertEqual(by_name["ADFA"]["data_form"], "host_system_call")
        self.assertEqual(by_name["OPCUA"]["domain"], "iot_iiot")
        self.assertEqual(by_name["OPCUA"]["capture_environment"], "synthetic_lab")
        self.assertEqual(by_name["CIC-IDS-2018"]["capture_environment"], "synthetic_lab")
        self.assertEqual(by_name["STPID"]["decision"], "exclude")
        self.assertEqual(by_name["STPID"]["domain"], "unknown")
        self.assertEqual(by_name["BLEBeacon"]["decision"], "exclude")
        self.assertEqual(by_name["IEEE-300-BUS POWER TEST SYSTEM"]["decision"], "exclude")
        self.assertIn("survey_reference", by_name["DARPA98"])
        profile = load_rating_profile(path, "openalex")
        self.assertEqual(len(profile.entities), 105)
        self.assertTrue(all("centralization" not in row for row in profile.entities.values()))

    def test_github_metrics_penalize_boundary_and_missing_results_transparently(self) -> None:
        """Verify operational, strict, and selective matrices use their documented denominators."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = (
                '[[entities]]\nid = "g1"\nname = "Correct detector"\ndecision = "accept"\n'
                'role = "native_detector"\nobservation_scopes = ["network"]\n'
                'capabilities = ["detection", "prevention"]\ndetection_methods = ["signature_based"]\n\n'
                '[[entities]]\nid = "g2"\ndecision = "exclude"\n\n'
                '[[entities]]\nid = "g3"\ndecision = "accept"\n\n'
                '[[entities]]\nid = "g4"\ndecision = "exclude"\n\n'
                '[[entities]]\nid = "g5"\ndecision = "accept"\n'
            )
            profile_path = self._write_profile(root / "github.toml", "github", body)
            classifications = root / "classifications.csv"
            atomic_write_csv(classifications, [
                {"github_repo_id": "g1", "decision": "accept", "role": "native_detector", "observation_scopes": "network", "capabilities": "prevention;detection", "detection_methods": "signature_based", 'classifier_supported_fields': '*'},
                {"github_repo_id": "g2", "decision": "accept", 'classifier_supported_fields': '*'},
                {"github_repo_id": "g3", "decision": "exclude", 'classifier_supported_fields': '*'},
                {"github_repo_id": "g4", "decision": "uncertain", 'classifier_supported_fields': '*'},
                {"github_repo_id": "unrated", "decision": "accept", 'classifier_supported_fields': '*'},
            ], ["github_repo_id", "decision", "role", "observation_scopes", "capabilities", "detection_methods"])
            candidates = root / "candidates.csv"
            atomic_write_csv(candidates, [{"github_repo_id": f"g{index}"} for index in range(1, 6)], ["github_repo_id"])

            result = evaluate_results(
                load_rating_profile(profile_path, "github"), classifications,
                root / "evaluation", candidates, confidence_level=.90,
            )
            metrics = result.metrics
            self.assertEqual(metrics["confidence_level"], .90)
            self.assertIn("detection_rate", metrics["confidence_intervals"])
            self.assertNotIn("confidence_intervals_95", metrics)
            self.assertEqual(metrics["operational"]["tp"], 1)
            self.assertEqual(metrics["operational"]["fp"], 1)
            self.assertEqual(metrics["operational"]["tn"], 1)
            self.assertEqual(metrics["operational"]["fn"], 2)
            self.assertAlmostEqual(metrics["operational"]["DR"], 1 / 3)
            self.assertAlmostEqual(metrics["operational"]["FPR"], 1 / 2)
            self.assertAlmostEqual(metrics["operational"]["FDR"], 1 / 2)
            self.assertAlmostEqual(metrics["operational"]["ACC"], 2 / 5)
            self.assertAlmostEqual(metrics["operational"]["PREC"], 1 / 2)
            self.assertEqual({key: metrics["strict"][key] for key in ("tp", "fp", "tn", "fn")}, {"tp": 1, "fp": 2, "tn": 0, "fn": 2})
            self.assertEqual({key: metrics["selective"][key] for key in ("tp", "fp", "tn", "fn")}, {"tp": 1, "fp": 1, "tn": 0, "fn": 1})
            self.assertEqual(metrics["status_counts"], {"classified": 3, "not_classified": 1, "uncertain": 1})
            self.assertAlmostEqual(metrics["retrieval_coverage"], 1.0)
            self.assertAlmostEqual(metrics["classification_coverage"], 4 / 5)
            self.assertAlmostEqual(metrics["definitive_coverage"], 3 / 5)
            self.assertAlmostEqual(metrics["abstention_rate"], 2 / 5)
            self.assertEqual(metrics["unrated_actual_entities"], 1)
            self.assertAlmostEqual(metrics["classification_field_accuracy"], 1.0)
            self.assertEqual(
                {item.name for item in (root / "evaluation").iterdir()},
                {
                    "metrics.json", "instances.csv", "decision_confusion.csv",
                    "field_metrics.csv", "field_label_metrics.csv", "README.md",
                    "reporting.sqlite3",
                },
            )
            self.assertNotIn("mean_jaccard", result.field_metrics[0])
            self.assertNotIn("label_precision", result.field_metrics[0])
            self.assertNotIn("label_recall", result.field_metrics[0])
            self.assertIn("label_f1", result.field_metrics[0])
            rows = {row["entity_id"]: row for row in read_csv(root / "evaluation/instances.csv")}
            self.assertEqual(rows["g5"]["status"], "not_classified")

    def test_openalex_metrics_compare_decisions_and_taxonomy_fields(self) -> None:
        """Verify OpenAlex uses stable work IDs and evaluates normalized field values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = (
                '[[entities]]\nopenalex_url = "https://openalex.org/W1/"\ndecision = "accept"\n'
                'paper_role = "original_release"\nrelevance_tier = "core"\nartifact_type = "dataset"\n'
                'domain = "general_network"\ndata_form = "packet"\ncapture_environment = "synthetic_lab"\nlabeling = "labeled"\n\n'
                '[[entities]]\nopenalex_url = "https://openalex.org/W2"\ndecision = "exclude"\n'
            )
            profile_path = self._write_profile(root / "openalex.toml", "openalex", body)
            actual = root / "classifications.csv"
            atomic_write_csv(actual, [
                {"openalex_id": "https://openalex.org/W1", "decision": "accept", "paper_role": "original_release", "relevance_tier": "core", "artifact_type": "dataset", "domain": "general_network", "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled", 'classifier_supported_fields': '*'},
                {"openalex_id": "https://openalex.org/W2", "decision": "exclude", 'classifier_supported_fields': '*'},
            ], ["openalex_id", "decision", "paper_role", "relevance_tier", "artifact_type", "domain", "data_form", "capture_environment", "labeling"])
            result = evaluate_results(load_rating_profile(profile_path, "openalex"), actual, root / "evaluation")
            self.assertEqual(result.metrics["operational"]["ACC"], 1.0)
            self.assertEqual(result.metrics["operational"]["DR"], 1.0)
            self.assertEqual(result.metrics["classification_field_accuracy"], 1.0)
            self.assertIsNone(result.metrics["retrieval_coverage"])
            persisted = json.loads((root / "evaluation/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["source"], "openalex")

    def test_openalex_artifact_profile_matches_names_and_aliases(self) -> None:
        """Evaluate dataset coverage independently of which paper represents the artifact."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "artifact-ratings.toml"
            profile_path.write_text(
                'schema_version = "1.0"\nsource = "openalex"\nevaluation_unit = "artifact"\n'
                'classification_fields = ["artifact_type", "domain"]\n\n'
                '[[entities]]\nartifact = "CIC-IDS 2017"\naliases = ["CICIDS2017"]\n'
                'decision = "accept"\nartifact_type = "dataset"\ndomain = "general_network"\n\n'
                '[[entities]]\nartifact = "Survey Named Dataset"\ndecision = "accept"\n'
                'artifact_type = "dataset"\ndomain = "host_endpoint"\n'
                'survey_reference_openalex_url = "https://openalex.org/W-reference"\n\n'
                '[[entities]]\nartifact = "Downstream Cited Dataset"\ndecision = "accept"\n'
                'artifact_type = "dataset"\ndomain = "general_network"\n'
                'survey_reference_openalex_url = "https://openalex.org/W-downstream"\n\n'
                '[[entities]]\nartifact = "Retrieved but unclassified"\ndecision = "accept"\n'
                'artifact_type = "dataset"\ndomain = "cloud"\n'
                'survey_reference_openalex_url = "https://openalex.org/W-candidate"\n\n'
                '[[entities]]\nartifact = "Not retrieved"\ndecision = "accept"\n'
                'artifact_type = "dataset"\ndomain = "uav"\n'
                'survey_reference_openalex_url = "https://openalex.org/W-absent"\n\n'
                '[[entities]]\nartifact = "Unresolved reference"\ndecision = "accept"\n'
                'artifact_type = "dataset"\ndomain = "vpn"\n\n'
                '[[entities]]\nartifact = "Uncertain cited release"\ndecision = "accept"\n'
                'artifact_type = "dataset"\ndomain = "iot_iiot"\n'
                'survey_reference_openalex_url = "https://openalex.org/W-uncertain"\n',
                encoding="utf-8",
            )
            actual = root / "classifications.csv"
            atomic_write_csv(actual, [
                {
                    "openalex_id": "W-version", "decision": "accept", "paper_role": "versioned_release",
                    "relevance_tier": "adjacent", "artifact_name": "CICIDS2017",
                    "artifact_display_name": "CICIDS2017", "artifact_aliases": "",
                    "artifact_type": "dataset", "domain": "general_network",
                 'classifier_supported_fields': '*'},
                {
                    "openalex_id": "W-original", "decision": "accept", "paper_role": "original_release",
                    "relevance_tier": "core", "artifact_name": "Canadian Institute for Cybersecurity IDS 2017",
                    "artifact_display_name": "CICIDS2017", "artifact_aliases": "CIC-IDS 2017",
                    "artifact_type": "dataset", "domain": "general_network",
                 'classifier_supported_fields': '*'},
                {
                    "openalex_id": "https://openalex.org/W-reference", "decision": "accept",
                    "paper_role": "original_release", "relevance_tier": "core",
                    "artifact_name": "Long descriptive source title", "artifact_display_name": "LDS",
                    "artifact_aliases": "", "artifact_type": "dataset", "domain": "host_endpoint",
                 'classifier_supported_fields': '*'},
                {
                    "openalex_id": "https://openalex.org/W-downstream", "decision": "exclude",
                    "paper_role": "downstream_application", "relevance_tier": "excluded",
                    "artifact_name": "", "artifact_display_name": "", "artifact_aliases": "",
                    "artifact_type": "unknown", "domain": "general_network",
                 'classifier_supported_fields': '*'},
                {
                    "openalex_id": "https://openalex.org/W-uncertain", "decision": "uncertain",
                    "paper_role": "uncertain", "relevance_tier": "uncertain",
                    "artifact_name": "", "artifact_display_name": "", "artifact_aliases": "",
                    "artifact_type": "unknown", "domain": "iot_iiot",
                 'classifier_supported_fields': '*'},
            ], [
                "openalex_id", "decision", "paper_role", "relevance_tier", "artifact_name",
                "artifact_display_name", "artifact_aliases", "artifact_type", "domain",
            ])
            candidates = root / "candidates.csv"
            atomic_write_csv(candidates, [
                {"openalex_id": "W-original"},
                {"openalex_id": "https://openalex.org/W-reference"},
                {"openalex_id": "https://openalex.org/W-downstream"},
                {"openalex_id": "https://openalex.org/W-candidate"},
                {"openalex_id": "https://openalex.org/W-uncertain"},
            ], ["openalex_id"])
            result = evaluate_results(
                load_rating_profile(profile_path, "openalex"), actual, root / "evaluation", candidates,
            )
            self.assertEqual(result.metrics["evaluation_unit"], "artifact")
            self.assertIn("false_discovery_rate", result.metrics["not_estimable_metrics"])
            self.assertIsNone(result.metrics["operational"]["FPR"])
            self.assertIsNone(result.metrics["operational"]["FDR"])
            self.assertIsNone(result.metrics["operational"]["ACC"])
            self.assertIsNone(result.metrics["confidence_intervals"]["false_discovery_rate"])
            self.assertEqual(result.metrics["operational"]["tp"], 2)
            self.assertEqual(result.metrics["operational"]["fn"], 5)
            self.assertEqual(result.metrics["status_counts"], {
                "classified": 2,
                "survey_reference_classification_uncertain": 1,
                "survey_reference_classified_nonrelease": 1,
                "survey_reference_not_retrieved": 1,
                "survey_reference_retrieved_not_classified": 1,
                "survey_reference_unresolved": 1,
            })
            self.assertAlmostEqual(result.metrics["classification_coverage"], 2 / 7)
            self.assertAlmostEqual(result.metrics["unresolved_artifact_rate"], 5 / 7)
            self.assertAlmostEqual(result.metrics["abstention_rate"], 1 / 7)
            self.assertIsNone(result.metrics["retrieval_coverage"])
            self.assertEqual(result.instances[0]["entity_id"], "W-original")
            self.assertEqual(result.instances[1]["match_method"], "accepted_survey_reference")

    def test_mixed_artifact_profile_scores_negative_controls_without_citation_fallback(self) -> None:
        """Negative artifacts make FPR estimable, but a cited paper alone is not an artifact match."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "ratings.toml"
            profile_path.write_text(
                'schema_version = "1.0"\nsource = "openalex"\nevaluation_unit = "artifact"\n'
                'classification_fields = ["artifact_type"]\n\n'
                '[[entities]]\nartifact = "GoodIDS"\ndecision = "accept"\nartifact_type = "dataset"\n\n'
                '[[entities]]\nartifact = "NoIDS"\ndecision = "exclude"\nartifact_type = "dataset"\n'
                'survey_reference_openalex_url = "https://openalex.org/W-citation"\n\n'
                '[[entities]]\nartifact = "BadIDS"\ndecision = "exclude"\nartifact_type = "dataset"\n',
                encoding="utf-8",
            )
            actual = root / "classifications.csv"
            atomic_write_csv(actual, [
                {"openalex_id": "https://openalex.org/W-good", "artifact_name": "GoodIDS", "decision": "accept", "artifact_type": "dataset", 'classifier_supported_fields': '*', 'artifact_display_name': "GoodIDS", 'resource_name': "GoodIDS"},
                {"openalex_id": "https://openalex.org/W-citation", "artifact_name": "Other Dataset", "decision": "accept", "artifact_type": "dataset", 'classifier_supported_fields': '*', 'artifact_display_name': "Other Dataset", 'resource_name': "Other Dataset"},
                {"openalex_id": "https://openalex.org/W-bad", "artifact_name": "BadIDS", "decision": "accept", "artifact_type": "dataset", 'classifier_supported_fields': '*', 'artifact_display_name': "BadIDS", 'resource_name': "BadIDS"},
            ], ["openalex_id", "artifact_name", "decision", "artifact_type"])
            result = evaluate_results(load_rating_profile(profile_path, "openalex"), actual, root / "evaluation")
            self.assertEqual(result.metrics["benchmark_scope"], "mixed_artifact_reference_set")
            self.assertEqual(result.metrics["not_estimable_metrics"], [])
            self.assertEqual(result.metrics["operational"]["tp"], 1)
            self.assertEqual(result.metrics["operational"]["fp"], 1)
            self.assertEqual(result.metrics["operational"]["tn"], 1)
            self.assertEqual(result.metrics["operational"]["FPR"], .5)
            self.assertEqual(result.metrics["classification_coverage"], 1.0)
            self.assertEqual(result.metrics["classification_field_comparisons"], 1)
            self.assertIsNotNone(result.metrics["confidence_intervals"]["false_positive_rate"])
            self.assertNotEqual(result.instances[1]["match_method"], "accepted_survey_reference")

    def test_readable_selectors_resolve_and_ambiguous_titles_fail(self) -> None:
        """Resolve repository names safely and reject non-unique paper titles."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_rating_profile(self._write_profile(root / "github.toml", "github", '[[entities]]\nrepository = "Owner/Detector"\ndecision = "accept"\n'), "github")
            actual = root / "github.csv"
            atomic_write_csv(actual, [{"github_repo_id": "42", "full_name": "owner/detector", "decision": "accept", 'classifier_supported_fields': '*'}], ["github_repo_id", "full_name", "decision"])
            candidates = root / "github-candidates.csv"
            atomic_write_csv(candidates, [{"github_repo_id": "42", "full_name": "Owner/Detector"}], ["github_repo_id", "full_name"])
            result = evaluate_results(profile, actual, root / "github-evaluation", candidates)
            self.assertEqual(result.instances[0]["entity_id"], "42")
            self.assertEqual(result.instances[0]["match_field"], "repository")

            title_profile = load_rating_profile(self._write_profile(root / "openalex.toml", "openalex", '[[entities]]\ntitle = "Repeated title"\ndecision = "exclude"\n'), "openalex")
            duplicate_actual = root / "openalex.csv"
            atomic_write_csv(duplicate_actual, [{"openalex_id": "W1", "title": "Repeated title", "decision": "exclude", 'classifier_supported_fields': '*'}, {"openalex_id": "W2", "title": "repeated  title", "decision": "exclude", 'classifier_supported_fields': '*'}], ["openalex_id", "title", "decision"])
            with self.assertRaisesRegex(ValueError, "ambiguous evaluation selector"):
                evaluate_results(title_profile, duplicate_actual, root / "openalex-evaluation")

    def test_evaluation_plot_bundle_visualizes_decisions_fields_and_coverage(self) -> None:
        """Render approachable decision, metadata, confusion, and coverage figures."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = (
                '[[entities]]\nid = "g1"\ndecision = "accept"\nrole = "native_detector"\n'
                'observation_scopes = ["network"]\nobservation_objects = ["packet_analysis"]\n'
                'capabilities = ["detection", "prevention"]\n'
                'detection_methods = ["signature_based"]\nprimary_system_type = "NIDS"\n'
                'system_types = ["NIDS"]\n\n'
                '[[entities]]\nid = "g2"\ndecision = "exclude"\nrole = "out_of_scope"\n'
                'observation_scopes = ["unknown"]\nobservation_objects = ["unknown"]\n'
                'capabilities = []\ndetection_methods = ["unknown"]\n'
                'primary_system_type = "unknown"\nsystem_types = ["unknown"]\n'
            )
            profile = load_rating_profile(self._write_profile(root / "github.toml", "github", body), "github")
            classifications = root / "classifications.csv"
            atomic_write_csv(classifications, [
                {"github_repo_id": "g1", "decision": "accept", "role": "native_detector", "observation_scopes": "network", "observation_objects": "packet_analysis", "capabilities": "detection", "detection_methods": "signature_based", "primary_system_type": "NIDS", "system_types": "NIDS", 'classifier_supported_fields': '*'},
                {"github_repo_id": "g2", "decision": "uncertain", "role": "uncertain", "observation_scopes": "unknown", "observation_objects": "unknown", "capabilities": "", "detection_methods": "unknown", "primary_system_type": "unknown", "system_types": "unknown", 'classifier_supported_fields': '*'},
            ], ["github_repo_id", "decision", "role", "observation_scopes", "observation_objects", "capabilities", "detection_methods", "primary_system_type", "system_types"])
            candidates = root / "candidates.csv"
            atomic_write_csv(candidates, [{"github_repo_id": "g1"}, {"github_repo_id": "g2"}], ["github_repo_id"])
            output_dir = root / "evaluation"
            result = evaluate_results(profile, classifications, output_dir, candidates)
            self.assertEqual(set(result.metrics["taxonomy_field_metrics"]), set(profile.classification_fields))
            self.assertEqual(result.metrics["taxonomy_field_metrics"]["observation_objects"]["comparisons"], 2)
            self.assertIn("packet_analysis", result.metrics["taxonomy_field_metrics"]["observation_objects"]["per_label"])
            self.assertIn("nids", result.metrics["taxonomy_field_metrics"]["system_types"]["per_label"])
            outputs = build_evaluation_plots(output_dir, {"formats": ["svg"], "dpi": 72})
            self.assertEqual([str(output.relative_to(output_dir)) for output in outputs], [
                "figures/01_decision_views.svg", "figures/02_decision_confusion.svg",
                "figures/03_taxonomy_fields.svg", "figures/04_taxonomy_categories.svg",
                "figures/05_pipeline_coverage.svg",
                "figures/06_taxonomy_accuracy.svg",
                "figures/07_recovered_taxonomy_accuracy.svg",
                "figures/08_reference_uncertainty.svg",
            ])
            self.assertTrue(all(output.is_file() and output.stat().st_size > 1000 for output in outputs))
            self.assertEqual(
                {item.name for item in (output_dir / "figures/data").iterdir()},
                {"01_decision_views.csv", "02_decision_confusion.csv", "03_taxonomy_fields.csv", "04_taxonomy_categories.csv", "05_coverage_rates.csv", "05_status_counts.csv", "06_taxonomy_accuracy.csv", "07_recovered_taxonomy_accuracy.csv", "08_reference_uncertainty.csv"},
            )
            readme = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("Metadata field accuracy", readme)
            self.assertIn("Interpreting the three views", readme)
            self.assertIn("definitive coverage and abstention rate", readme)
            self.assertIn("figures/03_taxonomy_fields.svg", readme)
            self.assertIn("figures/04_taxonomy_categories.svg", readme)
            field_svg = (output_dir / "figures/03_taxonomy_fields.svg").read_text(encoding="utf-8")
            self.assertIn("Observation object", field_svg)
            self.assertIn("Primary security system type", field_svg)
            self.assertIn("Security system types", field_svg)
            self.assertIn("Label F1", field_svg)
            self.assertNotIn("Jaccard", field_svg)
            accuracy_rows = read_csv(output_dir / "figures/data/06_taxonomy_accuracy.csv")
            for row in accuracy_rows:
                self.assertAlmostEqual(float(row["value"]), int(row["exact_matches"]) / int(row["comparisons"]))
            self.assertIn("Correct labels across the reference inventory (%)", (output_dir / "figures/06_taxonomy_accuracy.svg").read_text())
            recovered_rows = read_csv(output_dir / "figures/data/07_recovered_taxonomy_accuracy.csv")
            for row in recovered_rows:
                self.assertEqual(int(row["recovered"]), 1)
                self.assertEqual(int(row["comparisons"]), 1)
            conditional = {row["field"]: row for row in recovered_rows}
            self.assertEqual(float(conditional["role"]["value"]), 1)
            self.assertEqual(float(conditional["capabilities"]["value"]), 0)
            category_data = read_csv(output_dir / "figures/data/04_taxonomy_categories.csv")
            self.assertIn({"field": "observation_objects", "label": "packet_analysis"}, [
                {"field": row["field"], "label": row["label"]} for row in category_data
            ])
            self.assertIn({"field": "system_types", "label": "nids"}, [
                {"field": row["field"], "label": row["label"]} for row in category_data
            ])

    def test_rating_profiles_reject_duplicate_or_invalid_entities(self) -> None:
        """Reject ambiguous ground truth before producing misleading metrics."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = self._write_profile(root / "duplicate.toml", "github", '[[entities]]\nid = "1"\ndecision = "accept"\n[[entities]]\nid = "1"\ndecision = "exclude"\n')
            with self.assertRaisesRegex(ValueError, "duplicate evaluation entity"):
                load_rating_profile(duplicate, "github")
            invalid = self._write_profile(root / "invalid.toml", "openalex", '[[entities]]\nid = "W1"\ndecision = "maybe"\n')
            with self.assertRaisesRegex(ValueError, "unsupported decision"):
                load_rating_profile(invalid, "openalex")

    def test_cli_exposes_evaluate_with_candidate_and_rating_overrides(self) -> None:
        """Expose a source-local evaluation command without requiring a pipeline rerun."""
        args = build_parser().parse_args([
            "github", "evaluate", "--input", "actual.csv", "--output", "report",
            "--candidates", "candidates.csv", "--ratings", "ratings.toml",
        ])
        self.assertEqual(args.action, "evaluate")
        self.assertEqual(args.input, Path("actual.csv"))
        self.assertEqual(args.output, Path("report"))
        self.assertEqual(args.candidates, Path("candidates.csv"))
        self.assertEqual(args.ratings, Path("ratings.toml"))

    def test_openalex_run_schedules_only_the_dataset_evaluation_profile(self) -> None:
        """Use only the configured survey-derived dataset benchmark in normal evaluation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_profile = self._write_profile(
                root / "openalex_ashraf_2026.toml", "openalex",
                '[[entities]]\nid = "W2"\ndecision = "accept"\n',
            )
            settings = self._settings(root, root / "github.toml", dataset_profile)
            args = argparse.Namespace(action="run", ratings=None, output=None)

            jobs = _evaluation_jobs(settings, "openalex", args, root / "evaluation/openalex")

            self.assertEqual(jobs, [(dataset_profile, root / "evaluation/openalex", "shield")])
            args.action = "evaluate"
            self.assertEqual(_evaluation_jobs(settings, "openalex", args, root / "evaluation/openalex"), jobs)

    def test_multiple_reference_sets_are_isolated_and_do_not_change_decision_fingerprint(self) -> None:
        """Keep reference selection outside classification semantics and output collisions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_profile = self._write_profile(
                root / "github-one.toml", "github", "[[entities]]\nid = \"1\"\ndecision = \"accept\"\n",
            )
            second_profile = self._write_profile(
                root / "github-two.toml", "github", "[[entities]]\nid = \"2\"\ndecision = \"exclude\"\n",
            )
            openalex_profile = self._write_profile(
                root / "openalex.toml", "openalex", "[[entities]]\nid = \"W1\"\ndecision = \"accept\"\n",
            )
            settings = self._settings(root, github_profile, openalex_profile)
            before = decision_profile("fixture-model", "version", settings.section("ollama"), {"prompt": "fixed"})
            settings.section("evaluation")["reference_sets"].append({
                "name": "second set", "source": "github", "profile": str(second_profile),
            })
            after = decision_profile("fixture-model", "version", settings.section("ollama"), {"prompt": "fixed"})
            jobs = _evaluation_jobs(
                settings, "github", argparse.Namespace(action="run", ratings=None, output=None),
                root / "evaluation/github",
            )
            self.assertEqual(before, after)
            self.assertEqual(jobs, [
                (github_profile, root / "evaluation/github/ghml23", "ghml23"),
                (second_profile, root / "evaluation/github/second-set", "second-set"),
            ])

    def _settings(self, root: Path, github_ratings: Path, openalex_ratings: Path) -> Settings:
        """Create a single-model offline settings fixture for CLI evaluation stages."""
        return Settings(root, {
            "paths": {
                "interim_dir": str(root / "interim"), "processed_dir": str(root / "processed"),
                "plots_dir": str(root / "plots"), "runs_dir": str(root / "runs"),
                "evaluation_dir": str(root / "evaluation"),
                "github_cache": str(root / "github.sqlite3"), "openalex_cache": str(root / "openalex.sqlite3"),
                "github_terms": str(ROOT / "config/studies/ids_repositories/search/github.toml"), "openalex_terms": str(ROOT / "config/studies/ids_datasets/search/openalex.toml"),
                "github_prompts": str(ROOT / "config/studies/ids_repositories/prompts/classification.toml"),
                "github_screening": str(ROOT / "config/studies/ids_repositories/screening.toml"),
                "openalex_screening": str(ROOT / "config/studies/ids_datasets/screening.toml"),
                "github_taxonomy": str(ROOT / "config/studies/ids_repositories/taxonomy.toml"),
                "openalex_taxonomy": str(ROOT / "config/studies/ids_datasets/taxonomy.toml"), "openalex_prompts": str(ROOT / "config/studies/ids_datasets/prompts/classification.toml"),
                "github_system_types": str(ROOT / "config/studies/ids_repositories/taxonomy.toml"),
            },
            "logging": {"progress_every": 25},
            "ollama": {"models": ["fixture-model"], "temperature": 0.0, "seed": 42},
            "ensemble": {"enabled": False, "models": [], "threshold": 0.5},
            "evaluation": {
                "enabled": False,
                "reference_sets": [
                    {"name": "ghml23", "source": "github", "profile": str(github_ratings)},
                    {"name": "shield", "source": "openalex", "profile": str(openalex_ratings)},
                ],
            },
            "github": {"collection_limit": 10, "classification_limit": 10, "borderline_threshold": 4, "reconcile_with_llm": False},
            "openalex": {"collection_limit": 10, "classification_limit": 10, "borderline_threshold": 4, "reconcile_with_llm": False},
            "plots": {"formats": ["svg"], "dpi": 72, "top_n": 5},
        })

    def test_both_cli_evaluation_stages_write_metrics_and_manifest_counts(self) -> None:
        """Run both source evaluation commands offline and record their confusion counts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_profile = self._write_profile(root / "github.toml", "github", '[[entities]]\nid = "1"\ndecision = "accept"\n')
            openalex_profile = self._write_profile(root / "openalex.toml", "openalex", '[[entities]]\nid = "W1"\ndecision = "exclude"\n')
            settings = self._settings(root, github_profile, openalex_profile)
            github_actual = root / "github-actual.csv"
            github_candidates = root / "github-candidates.csv"
            atomic_write_csv(github_actual, [{"github_repo_id": "1", "decision": "accept", 'classifier_supported_fields': '*'}], ["github_repo_id", "decision"])
            atomic_write_csv(github_candidates, [{"github_repo_id": "1"}], ["github_repo_id"])
            openalex_actual = root / "openalex-actual.csv"
            openalex_candidates = root / "openalex-candidates.csv"
            atomic_write_csv(openalex_actual, [{"openalex_id": "W1", "decision": "exclude", 'classifier_supported_fields': '*'}], ["openalex_id", "decision"])
            atomic_write_csv(openalex_candidates, [{"openalex_id": "W1"}], ["openalex_id"])

            common = {"limit": None, "collect_limit": None, "classify_limit": None, "classify_all": False, "resume": True, "dry_run": False}
            github_counts = _execute_github(argparse.Namespace(action="evaluate", input=github_actual, output=root / "github-report", candidates=github_candidates, **common), settings)
            openalex_counts = _execute_openalex(argparse.Namespace(action="evaluate", input=openalex_actual, output=root / "openalex-report", candidates=openalex_candidates, **common), settings)
            self.assertEqual(github_counts["evaluation_tp_fixture-model_ghml23"], 1)
            self.assertEqual(openalex_counts["evaluation_tn_fixture-model_shield"], 1)
            self.assertTrue((root / "github-report/metrics.json").is_file())
            self.assertTrue((root / "openalex-report/metrics.json").is_file())
            self.assertTrue((root / "github-report/figures/03_taxonomy_fields.svg").is_file())
            self.assertTrue((root / "openalex-report/figures/05_pipeline_coverage.svg").is_file())
            manifests = list((root / "runs").glob("*.json"))
            self.assertEqual(len(manifests), 2)

    def test_multi_model_and_ensemble_evaluations_are_isolated_for_both_sources(self) -> None:
        """Evaluate every configured model and ensemble in separate source directories."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_profile = self._write_profile(root / "github.toml", "github", '[[entities]]\nid = "1"\ndecision = "accept"\n')
            openalex_profile = self._write_profile(root / "openalex.toml", "openalex", '[[entities]]\nid = "W1"\ndecision = "exclude"\n')
            settings = self._settings(root, github_profile, openalex_profile)
            settings.values["ollama"]["models"] = ["model-a", "model-b"]
            settings.values["ensemble"] = {"enabled": True, "models": [], "threshold": 0.5}
            models = ["model-a", "model-b"]
            github_candidates = root / "interim/github/candidates.csv"
            openalex_candidates = root / "interim/openalex/paper_candidates.csv"
            atomic_write_csv(github_candidates, [{"github_repo_id": "1"}], ["github_repo_id"])
            atomic_write_csv(openalex_candidates, [{"openalex_id": "W1"}], ["openalex_id"])
            for model, decision in zip(models, ("accept", "exclude")):
                target = _execution_paths(settings, "github", models)[model]["classifications"]
                atomic_write_csv(target, [{"github_repo_id": "1", "decision": decision, 'classifier_supported_fields': '*'}], ["github_repo_id", "decision"])
            github_ensemble = _variant_paths(settings, "github", Path("ensemble"))["classifications"]
            atomic_write_csv(github_ensemble, [{"github_repo_id": "1", "decision": "uncertain", 'classifier_supported_fields': '*'}], ["github_repo_id", "decision"])
            for model, decision in zip(models, ("exclude", "accept")):
                target = _execution_paths(settings, "openalex", models)[model]["classifications"]
                atomic_write_csv(target, [{"openalex_id": "W1", "decision": decision, 'classifier_supported_fields': '*'}], ["openalex_id", "decision"])
            openalex_ensemble = _variant_paths(settings, "openalex", Path("ensemble"))["classifications"]
            atomic_write_csv(openalex_ensemble, [{"openalex_id": "W1", "decision": "uncertain", 'classifier_supported_fields': '*'}], ["openalex_id", "decision"])

            common = {"action": "evaluate", "limit": None, "collect_limit": None, "classify_limit": None, "classify_all": False, "resume": True, "dry_run": False, "input": None, "output": None, "candidates": None}
            github_counts = _execute_github(argparse.Namespace(**common), settings)
            openalex_counts = _execute_openalex(argparse.Namespace(**common), settings)
            self.assertEqual(github_counts["evaluation_tp_model-a_ghml23"], 1)
            self.assertEqual(github_counts["evaluation_fn_model-b_ghml23"], 1)
            self.assertEqual(openalex_counts["evaluation_tn_model-a_shield"], 1)
            self.assertEqual(openalex_counts["evaluation_fp_model-b_shield"], 1)
            for source in ("github", "openalex"):
                paths = _execution_paths(settings, source, models)
                for model in models:
                    self.assertTrue((paths[model]["evaluation"] / "metrics.json").is_file())
                    self.assertEqual(
                        paths[model]["evaluation"].relative_to(root / "evaluation" / source),
                        Path("classifiers/llm") / model,
                    )
                self.assertTrue((_variant_paths(settings, source, Path("ensemble"))["evaluation"] / "metrics.json").is_file())
                self.assertTrue((root / "evaluation" / source / "README.md").is_file())
                self.assertTrue((root / "evaluation" / source / "comparison/00_metric_overview.svg").is_file())


if __name__ == "__main__":
    unittest.main()
