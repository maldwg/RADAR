from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from tests.fixtures import fixture_collection
from tests.fixtures import SYSTEM_TYPE_PROFILE
from typing import Any
from unittest.mock import Mock, patch

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import load_prompt_profile, load_settings, load_system_type_profile, load_term_profile
from ids_meta.github.pipeline import classify, export, prefilter, prompt
from ids_meta.github.reconcile import reconcile
from ids_meta.github.schema import CANDIDATE_COLUMNS, CLASSIFICATION_COLUMNS, RESPONSE_SCHEMA, ROLES, normalize_response, response_schema, system_type_codes
from tests.fixtures import write_csv as atomic_write_csv
from ids_meta.common.text import stable_hash
from ids_meta.github.collect import GitHubClient, _enrich, collect, search_shards


ROOT = Path(__file__).resolve().parents[1]


class FakeGitHubModel:
    model = "fixture-model"

    def __init__(self) -> None:
        """Initialize the fixture model's invocation counter."""
        self.calls = 0

    def generate(self, _prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
        """Return a deterministic accepted native-detector response."""
        self.calls += 1
        return {
            "role": "native_detector",
            "observation_scopes": ["network"], "observation_objects": ["packet_analysis"],
            "capabilities": ["detection"], "detection_methods": ["signature_based"],
            "primary_system_type": "NIDS", "system_types": ["NIDS"],
            "independent_fork": False, "direct_self_description": True,
            "runtime_observation": True, "reproducible_execution": True, "offline_dataset_only": False, "implements_detection_engine": True, "requires_external_detector": False, "integrates_multiple_detectors": True, "cohesive_framework_architecture": False, "research_basis": True, "upstream_full_name": "",
            "primary_reason": "The project directly describes a runnable network detector.",
            "evidence": [{"source": "README", "quote": "intruson detction system analyzes network traffic"}],
        }


class GitHubPipelineTests(unittest.TestCase):
    def test_client_paces_every_request_and_search_bucket(self) -> None:
        """Verify core and search pacing intervals are both enforced."""
        client = GitHubClient(request_pause_seconds=1.0, search_pause_seconds=2.2)
        client._last_request_at = 10.0
        client._last_search_at = 10.0
        with patch("ids_meta.github.collect.time.monotonic", side_effect=[10.5, 12.2]), patch("ids_meta.github.collect.time.sleep") as sleep:
            client._wait_before_request(True)
        self.assertEqual(sleep.call_count, 1)
        self.assertAlmostEqual(sleep.call_args.args[0], 1.7)
        self.assertEqual(client._last_request_at, 12.2)

    def test_rate_limit_wait_does_not_consume_request_retries(self) -> None:
        """Verify quota waits do not exhaust transient-failure retry attempts."""
        client = GitHubClient(token="fixture", request_pause_seconds=0, search_pause_seconds=0)
        limited = Mock(
            status_code=403,
            headers={
                "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "110",
                "X-RateLimit-Resource": "core", "X-RateLimit-Limit": "5000", "X-RateLimit-Used": "5000",
            },
            text="API rate limit exceeded",
        )
        success = Mock(
            status_code=200,
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Reset": "3700", "X-RateLimit-Resource": "core"},
            text="{}",
        )
        client.session.get = Mock(side_effect=[limited, success])
        with patch.object(client, "_wait_before_request"), patch("ids_meta.github.collect.time.time", return_value=100), patch("ids_meta.github.collect.time.sleep") as sleep:
            response = client.request("/repos/example/project")
        self.assertIs(response, success)
        self.assertEqual(client.session.get.call_count, 2)
        sleep.assert_called_once_with(12.0)

    def test_adaptive_pacing_spreads_remaining_quota(self) -> None:
        """Verify adaptive pacing distributes calls over the reset window."""
        client = GitHubClient(token=None, request_pause_seconds=2, search_pause_seconds=3)
        response = Mock(headers={"X-RateLimit-Remaining": "59", "X-RateLimit-Reset": "3700", "X-RateLimit-Resource": "core"})
        with patch("ids_meta.github.collect.time.time", return_value=100):
            client._observe_rate_limit(response)
        self.assertGreater(client._adaptive_core_pause, 60)

    def test_all_meaningful_roles_can_enter_combined_catalogue(self) -> None:
        """Verify every supported meaningful role can be accepted."""
        self.assertEqual(ROLES, ("native_detector", "classifier_engine", "detection_framework", "detection_platform", "traffic_analyzer", "antivirus", "research_project", "demo", "rules_content", "support_tooling"))
        self.assertNotIn("confidence", RESPONSE_SCHEMA["properties"])
        self.assertNotIn("runnable_detector", RESPONSE_SCHEMA["properties"])
        self.assertNotIn("derived_labels", CLASSIFICATION_COLUMNS)
        self.assertNotIn("role_confidence", CLASSIFICATION_COLUMNS)
        config = {"require_verified_quote": True}
        for role in ROLES:
            with self.subTest(role=role):
                decision = normalize_response({
                    "role": role, "observation_scopes": ["unknown"],
                    "observation_objects": ["file_analysis"] if role == "antivirus" else ["unknown"],
                    "capabilities": ["detection"], "detection_methods": ["unknown"],
                    "primary_system_type": "unknown", "system_types": ["unknown"],
                    "independent_fork": False, "direct_self_description": True,
                    "runtime_observation": True, "reproducible_execution": True, "offline_dataset_only": False, "implements_detection_engine": True, "requires_external_detector": False, "integrates_multiple_detectors": True, "cohesive_framework_architecture": role == "detection_framework", "research_basis": True, "upstream_full_name": "",
                    "primary_reason": "fixture",
                }, [{"source": "README", "quote": "fixture"}], config, False, system_type_profile=SYSTEM_TYPE_PROFILE)
                self.assertEqual(decision["decision"], "accept")

    def test_traffic_analyzer_accepts_offline_inspection_without_detection_engine(self) -> None:
        """Accept a runnable forensic analyzer while retaining the execution gate."""
        raw = {
            "role": "traffic_analyzer", "observation_scopes": ["network"],
            "observation_objects": ["packet_analysis"], "capabilities": ["monitoring"],
            "detection_methods": ["unknown"], "primary_system_type": "NSM", "system_types": ["NSM"],
            "direct_self_description": True, "reproducible_execution": True,
            "runtime_observation": False, "offline_dataset_only": True,
            "implements_detection_engine": False,
        }
        evidence = [{"source": "README", "quote": "Inspect a saved packet capture with analyze -r trace.pcap."}]
        accepted = normalize_response(raw, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(accepted["decision"], "accept")
        self.assertEqual(accepted["primary_system_type"], "NSM")
        self.assertEqual(accepted["classification_validation_flags"], "")
        rejected = normalize_response({**raw, "reproducible_execution": False}, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(rejected["decision"], "uncertain")
        self.assertEqual(rejected["classification_validation_flags"], "reproducible_execution_missing")

    def test_antivirus_accepts_offline_and_external_scanners_with_file_detection(self) -> None:
        """File scanning needs execution and file detection, independently of live IDS gates."""
        raw = {
            "role": "antivirus", "observation_scopes": ["host_endpoint"],
            "observation_objects": ["file_analysis"], "capabilities": ["detection"],
            "detection_methods": ["signature_based"], "primary_system_type": "AV", "system_types": ["AV"],
            "direct_self_description": True, "reproducible_execution": True,
            "runtime_observation": False, "offline_dataset_only": True,
            "implements_detection_engine": False, "requires_external_detector": True,
        }
        evidence = [{"source": "README", "quote": "Run scan uploads/ to scan files with ClamAV."}]
        accepted = normalize_response(raw, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(accepted["decision"], "accept")
        self.assertEqual(accepted["system_types"], "AV")
        self.assertEqual(accepted["observation_objects"], "file_analysis")
        self.assertEqual(accepted["classification_validation_flags"], "")
        for change, flag in (
            ({"reproducible_execution": False}, "reproducible_execution_missing"),
            ({"observation_objects": ["log_analysis"]}, "file_analysis_missing"),
            ({"capabilities": ["monitoring"]}, "detection_capability_missing"),
        ):
            with self.subTest(flag=flag):
                result = normalize_response({**raw, **change}, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
                self.assertEqual(result["decision"], "uncertain")
                self.assertIn(flag, result["classification_validation_flags"])
        unknown = normalize_response({**raw, "primary_system_type": "unknown", "system_types": ["unknown"]}, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(unknown["system_types"], "unknown")

    def test_native_detector_requires_independent_runtime_engine_evidence(self) -> None:
        """Verify static or externally delegated implementations cannot pass as native detectors."""
        decision = normalize_response({
            "role": "native_detector", "observation_scopes": ["network"], "observation_objects": ["unknown"],
            "capabilities": ["detection"], "detection_methods": ["ml_based"],
            "primary_system_type": "NIDS", "system_types": ["NIDS"],
            "independent_fork": False, "direct_self_description": True,
            "runtime_observation": False, "reproducible_execution": False,
            "offline_dataset_only": True, "implements_detection_engine": False, "requires_external_detector": False, "integrates_multiple_detectors": False, "cohesive_framework_architecture": False, "research_basis": False, "upstream_full_name": "", "primary_reason": "fixture",
        }, [{"source": "README", "quote": "evaluated on a static dataset"}], {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(decision["decision"], "uncertain")
        self.assertEqual(
            decision["classification_validation_flags"],
            "runtime_observation_missing;reproducible_execution_missing;offline_dataset_only;detection_engine_implementation_missing",
        )

    def test_cohesive_agent_manager_system_is_a_framework_not_native_detector(self) -> None:
        """Keep OSSEC- and Wazuh-style distributed systems on the framework side of the boundary."""
        evidence = [{"source": "README", "quote": "full platform with manager agents log analysis alerting and active response"}]
        raw = {
            "role": "native_detector", "observation_scopes": ["host_endpoint"], "observation_objects": ["log_analysis"],
            "capabilities": ["detection", "prevention"], "detection_methods": ["rule_based"],
            "primary_system_type": "HIDS", "system_types": ["HIDS", "HIPS"],
            "independent_fork": False, "direct_self_description": True,
            "runtime_observation": True, "reproducible_execution": True, "offline_dataset_only": False,
            "implements_detection_engine": True, "requires_external_detector": False,
            "integrates_multiple_detectors": False, "cohesive_framework_architecture": True,
            "research_basis": False, "upstream_full_name": "", "primary_reason": "fixture",
        }
        native = normalize_response(raw, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(native["decision"], "uncertain")
        self.assertIn("framework_architecture_conflict", native["classification_validation_flags"])
        framework = normalize_response({**raw, "role": "detection_framework"}, evidence, {}, False, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(framework["decision"], "accept")
        self.assertEqual(framework["classification_validation_flags"], "")

    def test_derivative_fork_requires_independence_evidence(self) -> None:
        """Verify derivative forks need evidence of independent significance."""
        decision = normalize_response({
            "role": "native_detector", "observation_scopes": ["network"], "observation_objects": ["packet_analysis"],
            "capabilities": ["detection"], "detection_methods": ["signature_based"],
            "primary_system_type": "NIDS", "system_types": ["NIDS"],
            "independent_fork": False, "direct_self_description": True,
            "runtime_observation": True, "reproducible_execution": True, "offline_dataset_only": False, "implements_detection_engine": True, "requires_external_detector": False, "integrates_multiple_detectors": True, "cohesive_framework_architecture": False, "research_basis": True, "upstream_full_name": "", "primary_reason": "fixture",
        }, [{"source": "README", "quote": "fixture"}], {}, True, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(decision["decision"], "uncertain")

    def test_search_uses_native_best_match_without_sort_override(self) -> None:
        """Verify GitHub controls repository relevance ordering directly."""
        client = GitHubClient(request_pause_seconds=0, search_pause_seconds=0)
        with patch.object(client, "get_json", return_value={"items": []}) as get_json:
            client.search("intrusion detection", 2, 50)
        params = get_json.call_args.args[1]
        self.assertEqual(params, {"q": "intrusion detection", "page": 2, "per_page": 50})

    def test_search_supports_explicit_sort_and_order(self) -> None:
        """Pass supported non-relevance ordering directly to GitHub repository search."""
        client = GitHubClient(request_pause_seconds=0, search_pause_seconds=0)
        with patch.object(client, "get_json", return_value={"items": []}) as get_json:
            client.search("intrusion detection", 2, 50, "updated", "asc")
        self.assertEqual(get_json.call_args.args[1], {
            "q": "intrusion detection", "page": 2, "per_page": 50,
            "sort": "updated", "order": "asc",
        })

    def test_search_is_sharded_below_github_cap(self) -> None:
        """Verify overflowing GitHub searches split into retrievable shards."""
        class SearchClient:
            def search(self, query: str, page: int, per_page: int, sort="best_match", order="desc") -> dict[str, Any]:
                """Return an overflowing parent interval and bounded child intervals."""
                # The full interval overflows; recursively split intervals fit.
                total = 1500 if "2008-01-01..2008-01-04" in query else 1
                return {"total_count": total, "incomplete_results": False, "items": [{"id": query}] if page == 1 and total == 1 else []}

        from datetime import date
        shards = list(search_shards(SearchClient(), "intrusion detection", date(2008, 1, 1), date(2008, 1, 4), 100))  # type: ignore[arg-type]
        self.assertGreaterEqual(len(shards), 2)
        self.assertTrue(all(not incomplete for _shard, _page, incomplete in shards))

    def test_collection_balances_best_match_across_queries(self) -> None:
        """Verify one early query cannot consume the complete collection budget."""
        class Client:
            authenticated = True
            def __init__(self) -> None:
                """Record the exact queries sent to GitHub search."""
                self.search_queries: list[str] = []
            def search(self, query: str, _page: int, _per_page: int, sort="best_match", order="desc") -> dict[str, Any]:
                """Return one native best-match result for each query."""
                self.search_queries.append(query)
                base = query.split(" stars:", 1)[0]
                repo_id = 1 if base == "first" else 2
                return {"total_count": 1, "items": [{"id": repo_id, "full_name": "example/" + base}]}
            def get_json(self, path: str, _params: dict[str, Any] | None = None) -> Any:
                """Return repository or empty commit metadata."""
                if path.endswith("/commits"):
                    return []
                name = path.rsplit("/repos/", 1)[-1]
                return {"id": 1 if name.endswith("first") else 2, "full_name": name, "owner": {"login": "example"}}
            def raw_readme(self, _full_name: str) -> str:
                """Return minimal repository evidence."""
                return "intrusion detection system"
            def docs(self, _full_name: str, _max_files: int) -> str:
                """Return no documentation fixture."""
                return ""
        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            with PipelineCache(Path(directory) / "cache.sqlite3") as cache:
                counts = collect(
                    {"created_from": "2020-01-01", "per_page": 100, "progress_every": 10, "search_scan_multiplier": 1},
                    {"search_queries": ["first", "second"]}, cache, limit=2, client=client, resume=False,
                )
                active = cache.get_record("github.collection", "latest")
            self.assertEqual(counts["discovered"], 2)
            self.assertEqual(set(active["repository_ids"]), {"1", "2"})
            self.assertEqual(active["ordering"], "github_best_match")
            self.assertEqual(client.search_queries, ["first stars:>=100", "second stars:>=100"])
            self.assertEqual(
                active["search_strategy"],
                f"github_native_best_match_v5_stars_100_cap_6_queries_{stable_hash('first', 'second')}",
            )

    def test_collection_caps_each_query_proportionally(self) -> None:
        """Verify query discovery oversamples the final collection by a configured factor."""
        class Client:
            authenticated = True

            def __init__(self) -> None:
                """Record search calls for the bounded query."""
                self.search_queries: list[str] = []

            def search(self, query: str, _page: int, _per_page: int, sort="best_match", order="desc") -> dict[str, Any]:
                """Return more repositories than one query may contribute."""
                self.search_queries.append(query)
                return {
                    "total_count": 150,
                    "items": [{"id": index, "full_name": f"example/repo-{index}"} for index in range(1, 151)],
                }

            def get_json(self, path: str, _params: dict[str, Any] | None = None) -> Any:
                """Return repository or empty commit metadata."""
                if path.endswith("/commits"):
                    return []
                name = path.rsplit("/repos/", 1)[-1]
                return {"id": int(name.rsplit("-", 1)[-1]), "full_name": name}

            def raw_readme(self, _full_name: str) -> str:
                """Return minimal repository evidence."""
                return "intrusion detection system"

            def docs(self, _full_name: str, _max_files: int) -> str:
                """Return no documentation fixture."""
                return ""

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            with PipelineCache(Path(directory) / "cache.sqlite3") as cache:
                counts = collect(
                    {"per_page": 100, "progress_every": 100, "query_limit_factor": 2},
                    {"search_queries": ["intrusion detection"]}, cache, limit=50, client=client, resume=False,
                )
                active = cache.get_record("github.collection", "latest")
                search_count = sum(1 for _ in cache.records("github.search_candidate"))
            self.assertEqual(search_count, 100)
            self.assertEqual(active["sample_size_per_query"], 100)
            self.assertEqual(counts["discovered"], 50)
            self.assertEqual(len(active["repository_ids"]), 50)
            self.assertEqual(client.search_queries, ["intrusion detection stars:>=100"])

    def test_resumed_collection_never_exceeds_repository_limit(self) -> None:
        """Verify a full cached pool does not trigger an extra discovery request."""
        class NoCallClient:
            authenticated = True

            def __init__(self) -> None:
                """Initialize the unexpected-search counter."""
                self.search_calls = 0

            def search(self, _query: str, _page: int, _per_page: int, sort="best_match", order="desc") -> dict[str, Any]:
                """Record an API search that should not occur in this fixture."""
                self.search_calls += 1
                return {"total_count": 0, "items": []}

        with tempfile.TemporaryDirectory() as directory:
            fake = NoCallClient()
            with PipelineCache(Path(directory) / "cache.sqlite3") as cache:
                for repo_id in ("1", "2"):
                    item = {"id": int(repo_id), "full_name": f"example/repo-{repo_id}"}
                    cache.put_record("github.search_candidate", repo_id, {
                        "item": item, "matched_queries": ["query"],
                        "search_strategy": f"github_native_best_match_v5_stars_100_cap_6_queries_{stable_hash('query')}",
                    }, "now")
                    cache.put_record("github.repository", repo_id, {"repository": item, "matched_queries": ["query"]}, "now")
                cache.put_record("github.collection", "latest", {
                    "search_strategy": f"github_native_best_match_v5_stars_100_cap_6_queries_{stable_hash('query')}",
                    "search_complete": True, "query_results_scanned": {"query": 2},
                }, "now")
                counts = collect(
                    {"created_from": "2020-01-01", "per_page": 100, "progress_every": 10},
                    {"search_queries": ["query"]}, cache, limit=2, client=fake, resume=True,  # type: ignore[arg-type]
                )
            self.assertEqual(counts["discovered"], 2)
            self.assertEqual(fake.search_calls, 0)

    def test_system_types_are_selected_by_the_llm_not_derived(self) -> None:
        """Preserve model-selected IDS types even when another dimension differs."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        profile = load_system_type_profile(settings, "github")
        schema = response_schema(profile)
        self.assertEqual(schema["properties"]["system_types"]["items"]["enum"][-1], "unknown")
        self.assertIn("DBIDS", schema["properties"]["primary_system_type"]["enum"])
        normalized = normalize_response({
            "role": "native_detector", "observation_scopes": ["network"],
            "observation_objects": ["log_analysis"], "capabilities": ["detection"],
            "detection_methods": ["rule_based"], "primary_system_type": "DBIDS",
            "system_types": ["DBIDS"], "independent_fork": False,
            "direct_self_description": True, "runtime_observation": True,
            "reproducible_execution": True, "offline_dataset_only": False,
            "implements_detection_engine": True, "requires_external_detector": False,
            "integrates_multiple_detectors": False, "cohesive_framework_architecture": False,
            "research_basis": False, "upstream_full_name": "", "primary_reason": "fixture",
        }, [{"source": "README", "quote": "fixture"}], {}, False, profile)
        self.assertEqual(normalized["primary_system_type"], "DBIDS")
        self.assertEqual(normalized["system_types"], "DBIDS")
        normalized_unknown_list = normalize_response({
            "role": "uncertain", "observation_scopes": ["unknown"],
            "observation_objects": ["unknown"], "capabilities": [],
            "detection_methods": ["unknown"], "primary_system_type": "HIDS",
            "system_types": ["unknown"],
        }, [], {"require_verified_quote": False}, False, profile)
        self.assertEqual(normalized_unknown_list["system_types"], "HIDS")

    def test_configured_system_types_extend_the_llm_schema(self) -> None:
        """Add a configured LLM output type without changing Python enums."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        profile = load_system_type_profile(settings, "github")
        extended = {"types": {**profile["types"], "CLOUDIDS": {
            "name": "Cloud intrusion detection system", "description": "Fixture extension",
        }}}
        self.assertIn("CLOUDIDS", system_type_codes(extended))
        self.assertIn("CLOUDIDS", response_schema(extended)["properties"]["system_types"]["items"]["enum"])
        rendered = prompt([], load_prompt_profile(settings, "github"), extended)
        for code, definition in extended["types"].items():
            self.assertIn(f"- {code} ({definition['name']}): {definition['description']}", rendered)
        self.assertNotIn("${system_types}", rendered)

    def test_general_ids_types_are_preserved_for_specific_observation_scopes(self) -> None:
        """Keep service-log and flow detectors general despite endpoint/network scopes."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        profile = load_system_type_profile(settings, "github")
        for scope, objects, capabilities, primary, types in (
            ("host_endpoint", ["log_analysis"], ["detection", "prevention"], "IPS", ["IPS", "IDS"]),
            ("network", ["flow_analysis", "log_analysis"], ["detection"], "IDS", ["IDS"]),
            ("network", ["packet_analysis"], ["detection", "prevention"], "IPS", ["IPS"]),
        ):
            with self.subTest(scope=scope, objects=objects):
                normalized = normalize_response({
                    "role": "native_detector", "observation_scopes": [scope],
                    "observation_objects": objects, "capabilities": capabilities,
                    "detection_methods": ["rule_based"], "primary_system_type": primary,
                    "system_types": types, "implements_detection_engine": True,
                    "runtime_observation": True, "reproducible_execution": True,
                }, [], {"require_verified_quote": False}, False, profile)
                self.assertEqual(normalized["primary_system_type"], primary)
                self.assertEqual(normalized["system_types"], ";".join(types))

    def test_security_monitoring_type_does_not_require_detection_or_alerting(self) -> None:
        """Preserve a monitoring-only SM annotation through the configured schema."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        profile = load_system_type_profile(settings, "github")
        for selected in (profile,):
            with self.subTest(configured=selected is not None):
                self.assertIn("SM", response_schema(selected)["properties"]["primary_system_type"]["enum"])
                normalized = normalize_response({
                    "role": "support_tooling", "observation_scopes": ["host_endpoint"],
                    "observation_objects": ["log_analysis"], "capabilities": ["monitoring"],
                    "detection_methods": ["unknown"], "primary_system_type": "SM", "system_types": ["SM"],
                    "direct_self_description": True, "reproducible_execution": True,
                    "implements_detection_engine": False,
                }, [{"source": "README", "quote": "Aggregates endpoint audit logs for security monitoring."}],
                    {}, False, selected)
                self.assertEqual(normalized["decision"], "accept")
                self.assertEqual(normalized["system_types"], "SM")
                self.assertEqual(normalized["capabilities"], "monitoring")
                self.assertEqual(normalized["detection_methods"], "unknown")

    def test_enrichment_records_exact_latest_commit_time(self) -> None:
        """Verify enrichment preserves the latest commit's exact timestamp."""
        class Client:
            def get_json(self, path: str, _params: dict[str, Any] | None = None):
                """Return fixture repository or latest-commit API metadata."""
                if path.endswith("/commits"):
                    return [{
                        "sha": "abc", "html_url": "https://github.invalid/commit/abc",
                        "commit": {"committer": {"date": "2025-05-06T07:08:09Z"}},
                    }]
                return {"full_name": "example/project", "parent": None}

            def raw_readme(self, _full_name: str) -> str:
                """Return fixture README evidence."""
                return "README"

            def docs(self, _full_name: str, _max_files: int) -> str:
                """Return fixture documentation evidence."""
                return "docs"

        payload = _enrich(Client(), {"full_name": "example/project"}, 1)  # type: ignore[arg-type]
        self.assertEqual(payload["last_commit_at"], "2025-05-06T07:08:09Z")
        self.assertEqual(payload["last_commit_sha"], "abc")

    def test_uncertain_classification_retries_with_more_docs(self) -> None:
        """Verify unresolved repository evidence triggers one broader docs fetch and retry."""
        class Model:
            model = "fixture-model"
            def __init__(self) -> None:
                """Initialize the model call count."""
                self.calls = 0
            def generate(self, prompt: str, _schema: dict[str, Any]) -> dict[str, Any]:
                """Produce a below-threshold split before the expanded-evidence restart."""
                self.calls += 1
                expanded = "decisive runnable detector documentation" in prompt
                accepted = expanded or self.calls <= 3
                return {
                    "role": "native_detector" if accepted else "out_of_scope",
                    "observation_scopes": ["network"], "observation_objects": ["packet_analysis"],
                    "capabilities": ["detection"], "detection_methods": ["unknown"],
                    "primary_system_type": "NIDS" if accepted else "unknown",
                    "system_types": ["NIDS"] if accepted else ["unknown"], "independent_fork": False,
                    "direct_self_description": accepted, "runtime_observation": accepted,
                    "reproducible_execution": accepted, "offline_dataset_only": False,
                    "implements_detection_engine": accepted, "requires_external_detector": False,
                    "integrates_multiple_detectors": False, "cohesive_framework_architecture": False, "research_basis": False,
                    "upstream_full_name": "", "primary_reason": "fixture",
                    "evidence": [{"source": "Documentation" if expanded else "README",
                                  "quote": "decisive runnable detector documentation" if expanded else "ambiguous project"}],
                }

        class Enricher:
            def enrich(self, _payload: dict[str, Any]) -> dict[str, str]:
                """Return the decisive expanded documentation fixture."""
                return {"docs_text": "decisive runnable detector documentation"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified = root / "candidates.csv", root / "classified.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update({"github_repo_id": "1", "full_name": "example/ambiguous",
                        "prefilter_band": "candidate", "prefilter_score": "7", "evidence_hash": "old"})
            atomic_write_csv(candidates, [row], CANDIDATE_COLUMNS)
            model = Model()
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("github.repository", "1", {
                    "repository": {"full_name": "example/ambiguous"}, "readme_text": "ambiguous project",
                }, "now")
                decisions = classify(candidates, classified, cache, {"require_verified_quote": True},
                                     generator=model, enricher=Enricher(),
                                     prompt_config=load_prompt_profile(load_settings(ROOT / "config/studies/ids_repositories/study.toml"), "github"),
                                     consecutive_runs=5, consensus_threshold=.8, system_type_profile=SYSTEM_TYPE_PROFILE)
            self.assertEqual(model.calls, 10, decisions)
            self.assertEqual(decisions[0]["decision"], "accept")
            self.assertEqual(decisions[0]["ensemble_vote_status"], "threshold_met")
            self.assertEqual(decisions[0]["evidence_expansion_required"], "true")
            self.assertEqual(decisions[0]["evidence_expansion_attempted"], "true")
            self.assertEqual(decisions[0]["evidence_expansion_improved"], "true")
            self.assertEqual(decisions[0]["evidence_expansion_resolved"], "true")

    def test_offline_pipeline_and_resume(self) -> None:
        """Verify the offline GitHub pipeline exports and resumes decisions."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        profile = load_term_profile(settings, "github")
        prompts = load_prompt_profile(settings, "github")
        system_types = load_system_type_profile(settings, "github")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified, reconciled, catalogue = root / "candidates.csv", root / "classified.csv", root / "reconciled.csv", root / "repositories.csv"
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("github.repository", "101", {
                    "repository": {
                        "id": 101, "node_id": "R_101", "full_name": "example/network-guard", "name": "network-guard",
                        "owner": {"login": "example"}, "html_url": "https://github.invalid/example/network-guard",
                        "description": "A network intruson detction system", "topics": ["network-security"],
                        "license": {"spdx_id": "MIT"}, "language": "Python", "fork": False, "archived": False,
                        "stargazers_count": 0, "forks_count": 0, "open_issues_count": 0,
                        "created_at": "2024-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
                        "pushed_at": "2025-01-01T00:00:00Z", "default_branch": "main",
                    },
                    "readme_text": "This intruson detction system analyzes network traffic. Install and run the sensor.",
                    "docs_text": "", "last_commit_sha": "abc", "parent_full_name": "", "matched_queries": ["intrusion detection system"],
                }, "2026-01-01T00:00:00+00:00")
                rows = prefilter(settings.section("github"), profile, fixture_collection(cache, 'github'), candidates, "fixture-run")
                self.assertEqual(rows[0]["prefilter_band"], "candidate")
                model = FakeGitHubModel()
                first = classify(candidates, classified, cache, settings.section("ollama"), generator=model, prompt_config=prompts, system_type_profile=system_types, consecutive_runs=3, consensus_threshold=.66)
                self.assertEqual(first[0]["decision"], "accept")
                self.assertTrue(first[0]["consensus_field_audit"])
                self.assertEqual(first[0]["consensus_validation_flags"], "")
                self.assertEqual(first[0]["system_types"], "NIDS")
                self.assertEqual(first[0]["primary_system_type"], "NIDS")
                self.assertIn("Network intrusion detection system", first[0]["system_type_names"])
                classify(candidates, classified, cache, settings.section("ollama"), generator=model, prompt_config=prompts, system_type_profile=system_types, consecutive_runs=3, consensus_threshold=.66)
                self.assertEqual(model.calls, 3, "matching evidence/profile should resume without another LLM call")
                reconcile(classified, reconciled, cache, settings.section("ollama"), settings.section("github"), dry_run=True)
                final = export(reconciled, catalogue, system_types)
                self.assertEqual(len(final), 1)
                self.assertEqual(final[0]["role"], "native_detector")
                self.assertEqual(final[0]["repository_relationship"], "canonical")
                self.assertEqual(final[0]["canonical_resolution"], "singleton_project")
                self.assertEqual(final[0]["canonical_margin"], "")
                self.assertEqual(final[0]["canonical_confidence"], "")
                self.assertIn("margin n/a", final[0]["canonical_reason"])
                self.assertNotIn("role=", final[0]["canonical_score_breakdown"])

    def test_classification_limit_includes_cached_ranked_rows(self) -> None:
        """Verify resume keeps the same bounded score-ranked shortlist."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates, classified = root / "candidates.csv", root / "classified.csv"
            rows = []
            with PipelineCache(root / "cache.sqlite3") as cache:
                for repo_id, score in (("1", "7"), ("2", "12"), ("3", "9")):
                    row = {column: "" for column in CANDIDATE_COLUMNS}
                    row.update({
                        "github_repo_id": repo_id, "full_name": f"example/repo-{repo_id}",
                        "prefilter_band": "candidate", "prefilter_score": score,
                        "evidence_hash": f"evidence-{repo_id}",
                    })
                    rows.append(row)
                    cache.put_record("github.repository", repo_id, {
                        "repository": {"full_name": f"example/repo-{repo_id}"},
                        "readme_text": "fixture evidence", "docs_text": "",
                    }, "2026-01-01T00:00:00+00:00")
                atomic_write_csv(candidates, rows, CANDIDATE_COLUMNS)
                first = classify(candidates, classified, cache, {}, limit=2, dry_run=True, system_type_profile=SYSTEM_TYPE_PROFILE)
                second = classify(candidates, classified, cache, {}, limit=1, dry_run=True, system_type_profile=SYSTEM_TYPE_PROFILE)
            self.assertEqual([row["github_repo_id"] for row in first], ["2", "3"])
            self.assertEqual([row["github_repo_id"] for row in second], ["2"])

    def test_project_reconciliation_is_order_independent(self) -> None:
        """Verify canonical repository selection is independent of input order."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def row(repo_id: str, name: str) -> dict[str, str]:
                """Build one accepted repository classification fixture."""
                value = {column: "" for column in CLASSIFICATION_COLUMNS}
                value.update({
                    "github_repo_id": repo_id, "full_name": f"example/{name}", "owner": "example", "name": name,
                    "decision": "accept", "role": "native_detector", "fork": "false",
                    "direct_self_description": "true", "last_commit_sha": "sha", "stars": "0",
                 'classifier_supported_fields': '*'})
                return value
            source_rows = [row("102", "network-guard-docker"), row("101", "network-guard")]
            selections = []
            with PipelineCache(root / "cache.sqlite3") as cache:
                for index, ordered in enumerate((source_rows, list(reversed(source_rows)))):
                    source, output = root / f"source-{index}.csv", root / f"output-{index}.csv"
                    atomic_write_csv(source, ordered, CLASSIFICATION_COLUMNS)
                    result = reconcile(source, output, cache, settings.section("ollama"), settings.section("github"), dry_run=True)
                    accepted = [item for item in result if item.get("decision") == "accept"]
                    selections.append({item["canonical_repo_id"] for item in accepted})
                    relationships = {item["github_repo_id"]: item["repository_relationship"] for item in accepted}
                    self.assertEqual(relationships, {"101": "canonical", "102": "deployment_variant"})
            self.assertEqual(selections, [{"101"}, {"101"}])

    def test_verified_upstream_variant_joins_base_without_becoming_canonical(self) -> None:
        """Verify a cross-owner deployment variant resolves to its explicit upstream base."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.csv", root / "output.csv"
            base = {column: "" for column in CLASSIFICATION_COLUMNS}
            base.update({
                "github_repo_id": "1", "full_name": "OISF/suricata", "owner": "OISF", "name": "suricata",
                "decision": "accept", "role": "native_detector", "fork": "false",
                "direct_self_description": "true", "last_commit_sha": "sha",
             'classifier_supported_fields': '*'})
            variant = {column: "" for column in CLASSIFICATION_COLUMNS}
            variant.update({
                "github_repo_id": "2", "full_name": "third/ansible-suricata", "owner": "third", "name": "ansible-suricata",
                "decision": "accept", "role": "support_tooling", "fork": "false",
                "direct_self_description": "true", "upstream_full_name": "OISF/suricata", "last_commit_sha": "sha",
             'classifier_supported_fields': '*'})
            atomic_write_csv(source, [variant, base], CLASSIFICATION_COLUMNS)
            with PipelineCache(root / "cache.sqlite3") as cache:
                result = reconcile(source, output, cache, settings.section("ollama"), settings.section("github"), dry_run=True)
            values = {row["github_repo_id"]: row for row in result}
            self.assertEqual(values["1"]["repository_relationship"], "canonical")
            self.assertEqual(values["2"]["repository_relationship"], "deployment_variant")
            self.assertEqual(values["2"]["canonical_full_name"], "OISF/suricata")
            self.assertEqual(values["1"]["canonical_margin"], "")
            self.assertNotIn("-9", values["1"]["canonical_reason"])

    def test_canonical_selection_does_not_prefer_one_repository_role(self) -> None:
        """Let lineage evidence beat role labels when selecting a project family base."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.csv", root / "output.csv"
            native = {column: "" for column in CLASSIFICATION_COLUMNS}
            native.update({
                "github_repo_id": "1", "full_name": "example/detector-core", "owner": "example", "name": "detector-core",
                "decision": "accept", "role": "native_detector", "fork": "false",
                "direct_self_description": "true", "last_commit_sha": "sha",
             'classifier_supported_fields': '*'})
            framework = {column: "" for column in CLASSIFICATION_COLUMNS}
            framework.update({
                "github_repo_id": "2", "full_name": "example/framework-suite", "owner": "example", "name": "framework-suite",
                "decision": "accept", "role": "detection_framework", "fork": "false",
                "direct_self_description": "true", "last_commit_sha": "sha",
             'classifier_supported_fields': '*'})
            atomic_write_csv(source, [native, framework], CLASSIFICATION_COLUMNS)
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("github.repository", "1", {"readme_text": "Canonical project: https://github.com/example/framework-suite", "docs_text": ""}, "now")
                cache.put_record("github.repository", "2", {"readme_text": "", "docs_text": ""}, "now")
                result = reconcile(source, output, cache, settings.section("ollama"), settings.section("github"), dry_run=True)
            by_id = {row["github_repo_id"]: row for row in result}
            self.assertEqual(by_id["1"]["canonical_repo_id"], "2")
            self.assertEqual(by_id["2"]["repository_relationship"], "canonical")
            self.assertNotIn("role=", by_id["2"]["canonical_score_breakdown"])

    def test_derivative_only_family_reports_missing_base(self) -> None:
        """Verify a lone packaging repository is not called canonical."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.csv", root / "output.csv"
            row = {column: "" for column in CLASSIFICATION_COLUMNS}
            row.update({
                "github_repo_id": "2", "full_name": "third/ansible-suricata", "owner": "third", "name": "ansible-suricata",
                "decision": "accept", "role": "support_tooling", "fork": "false", "direct_self_description": "true",
             'classifier_supported_fields': '*'})
            atomic_write_csv(source, [row], CLASSIFICATION_COLUMNS)
            result = reconcile(source, output, None, settings.section("ollama"), settings.section("github"), dry_run=True)
            self.assertEqual(result[0]["repository_relationship"], "base_repository_missing")
            self.assertEqual(result[0]["canonical_repo_id"], "")
            self.assertEqual(result[0]["canonical_margin"], "")



if __name__ == "__main__":
    unittest.main()
