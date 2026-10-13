"""Exercise proportional discovery budgets using offline API fixtures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ids_meta.cli import _pipeline_limits, build_parser
from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError
from ids_meta.common.limits import optional_limit, query_result_limit
from ids_meta.github.collect import collect as collect_github, search_best_matches
from ids_meta.openalex.collect import OpenAlexClient, collect as collect_openalex, sample_query_results


class RepositoryClient:
    """Serve a finite search result stream without contacting GitHub."""

    authenticated = True

    def __init__(self, count: int = 150):
        """Initialize fixture repositories and observed search requests."""
        self.items = [{"id": index, "full_name": f"fixture/repo-{index}"} for index in range(1, count + 1)]
        self.calls: list[tuple[str, int]] = []

    def search(self, query, page, per_page, sort="best_match", order="desc"):
        """Return an actual page slice in native search order."""
        self.calls.append((query, page))
        return {"total_count": len(self.items), "items": self.items[(page - 1) * per_page:page * per_page]}


class PaperClient:
    """Serve direct papers, a survey, and its references without network calls."""

    def __init__(self, count: int = 150, references: int = 0):
        """Record pagination bounds and configure the reference inventory."""
        self.count = count
        self.references = references
        self.calls: list[tuple[str, int, int | None]] = []

    def search(self, query, per_page, max_pages, sort="best_match", order="desc"):
        """Honor the supplied page cap so hidden truncation is detectable."""
        self.calls.append((query, per_page, max_pages))
        if query == "survey":
            yield {"id": "S", "title": "Survey of datasets", "type": "review",
                   "referenced_works": [f"R{index}" for index in range(self.references)]}
        else:
            count = self.count if max_pages is None else min(self.count, per_page * max_pages)
            for index in range(count):
                yield {"id": f"W{index}", "title": f"Dataset release paper {index}"}

    def works_by_ids(self, ids):
        """Resolve every requested reference to a usable release paper."""
        for key in ids:
            yield {"id": key, "title": f"Dataset reference release {key}"}


class QueryLimitTests(unittest.TestCase):
    """Verify bounded and unbounded discovery share one limit policy."""

    def test_limit_policy_uses_larger_finite_limit_and_rounds_up(self):
        """Avoid starving collection while permitting fractional multipliers."""
        for settings, collection, expected in [
            ({}, 1000, 3000),
            ({"classification_limit": 1000}, None, 3000),
            ({"classification_limit": 100}, 1000, 3000),
            ({"classification_limit": 1000}, 100, 3000),
            ({"classification_limit": "all"}, 1000, 3000),
            ({"query_limit_factor": 1.5}, 3, 5),
            ({"query_limit_factor": .5}, 3, 2),
            ({}, None, None),
            ({"classification_limit": "all"}, None, None),
        ]:
            with self.subTest(settings=settings, collection=collection):
                self.assertEqual(query_result_limit(settings, collection), expected)

    def test_route_factors_inherit_or_override_common_factor(self):
        """Keep survey and reference budgets proportional and independently adjustable."""
        settings = {"query_limit_factor": 4, "survey_query_limit_factor": 2}
        self.assertEqual(query_result_limit(settings, 10, "survey_query_limit_factor"), 20)
        self.assertEqual(query_result_limit(settings, 10, "reference_limit_factor"), 40)
        self.assertIsNone(query_result_limit(settings, None, "survey_query_limit_factor"))

    def test_invalid_factors_and_limits_fail_explicitly(self):
        """Reject zero, fractional record limits, booleans, and nonfinite factors."""
        for value in (0, -1, 1.5, True, "garbage"):
            with self.subTest(limit=value), self.assertRaises(ConfigError):
                optional_limit(value, "collection_limit")
        for value in (0, -1, float("inf"), float("nan"), True, "garbage"):
            with self.subTest(factor=value), self.assertRaises(ConfigError):
                query_result_limit({"query_limit_factor": value}, None)

    def test_unlimited_config_and_cli_override_finite_defaults(self):
        """Ensure missing limits and explicit all flags introduce no hidden defaults."""
        parser = build_parser()
        args = parser.parse_args(["github", "run", "--collect-all", "--classify-all"])
        self.assertEqual(_pipeline_limits(args, {"collection_limit": 1000, "classification_limit": 100}), (None, None))
        args = parser.parse_args(["openalex", "collect"])
        self.assertEqual(_pipeline_limits(args, {}), (None, None))
        self.assertEqual(_pipeline_limits(args, {"collection_limit": "all", "classification_limit": "all"}), (None, None))
        args = parser.parse_args(["openalex", "collect", "--collect-all", "--classify-limit", "1000"])
        self.assertEqual(_pipeline_limits(args, {}), (None, 1000))
        with self.assertRaises(SystemExit):
            parser.parse_args(["github", "collect", "--collect-all", "--collect-limit", "10"])

    def test_github_oversamples_every_query_and_resumes_completed_scan(self):
        """Fetch three times the collection cap per query, then reuse the complete pool."""
        client = RepositoryClient()
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / "cache.sqlite3") as cache:
            for item in client.items:
                cache.put_record("github.repository", str(item["id"]), {"repository": item}, "now")
            profile = {"search_queries": ["first", "second"]}
            collect_github({}, profile, cache, 10, client=client)
            self.assertEqual(sum(1 for _ in cache.records("github.search_candidate")), 30)
            active = cache.get_record("github.collection", "latest")
            self.assertEqual(len(active["repository_ids"]), 10)
            self.assertEqual(active["query_results_scanned"], {"first": 30, "second": 30})
            self.assertEqual(len(client.calls), 2)
            collect_github({}, profile, cache, 10, client=client)
            self.assertEqual(len(client.calls), 2)
            collect_github({"query_limit_factor": 4}, profile, cache, 10, client=client)
            self.assertEqual(len(client.calls), 4)
            self.assertEqual(cache.get_record("github.collection", "latest")["query_results_scanned"],
                             {"first": 40, "second": 40})

    def test_github_unlimited_paginates_past_old_default_and_records_null_cap(self):
        """Exhaust all reachable pages when neither pipeline limit is numeric."""
        client = RepositoryClient()
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / "cache.sqlite3") as cache:
            for item in client.items:
                cache.put_record("github.repository", str(item["id"]), {"repository": item}, "now")
            counts = collect_github({"query_sample_size": 1}, {"search_queries": ["query"]}, cache, client=client)
            active = cache.get_record("github.collection", "latest")
            self.assertEqual(counts["discovered"], 150)
            self.assertIsNone(active["sample_size_per_query"])
            self.assertEqual(active["query_results_scanned"], {"query": 150})
            self.assertEqual([page for _query, page in client.calls], [1, 2])
            self.assertTrue(all("created:" in query for query, _page in client.calls))

    def test_github_small_pages_do_not_impose_ten_page_cap(self):
        """Retrieve a full native window even with fewer than 100 items per page."""
        client = RepositoryClient(1000)
        pages = list(search_best_matches(client, "query", 50))
        self.assertEqual(len(pages), 20)
        self.assertEqual(sum(len(page["items"]) for _shard, page, _incomplete in pages), 1000)

    def test_openalex_derived_cap_overrides_old_page_cap(self):
        """Derive enough pages to reach the factor cap before final selection."""
        client = PaperClient()
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / "cache.sqlite3") as cache:
            counts = collect_openalex({"max_surveys": 0, "per_page": 10, "max_pages_per_query": 1,
                                      "query_limit_factor": 3, "retrieval_minimum_score": 0},
                                     {"direct_queries": ["first", "second"]}, cache, 40, client)
            active = cache.get_record("openalex.collection", "latest")
            self.assertEqual(counts["direct_results_scanned"], 240)
            self.assertEqual(counts["discovered"], 40)
            self.assertEqual(active["sample_size_per_query"], 120)
            self.assertEqual(client.calls, [("first", 10, 12), ("second", 10, 12)])

    def test_openalex_unlimited_queries_and_references_have_no_hidden_budget(self):
        """Retain the complete fixture, including references beyond old absolute bounds."""
        client = PaperClient(150, references=310)
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / "cache.sqlite3") as cache:
            counts = collect_openalex({"max_surveys": 1, "per_page": 10, "max_pages_per_query": 1,
                                      "query_sample_size": 1, "reference_sample_size": 1,
                                      "retrieval_minimum_score": 0},
                                     {"survey_queries": ["survey"], "direct_queries": ["direct"]}, cache, client=client)
            active = cache.get_record("openalex.collection", "latest")
            self.assertEqual(counts["direct_results_scanned"], 150)
            self.assertEqual(counts["reference_results_scanned"], 310)
            self.assertEqual(counts["discovered"], 460)
            self.assertIsNone(active["sample_size_per_query"])
            self.assertIsNone(active["reference_sample_size"])
            self.assertEqual(client.calls, [("survey", 10, None), ("direct", 10, None)])

    def test_openalex_route_factors_bound_surveys_and_reference_traversal(self):
        """Apply distinct factors without using an absolute survey or reference budget."""
        client = PaperClient(150, references=20)
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / "cache.sqlite3") as cache:
            counts = collect_openalex({"classification_limit": 2, "query_limit_factor": 3,
                                      "survey_query_limit_factor": 2, "reference_limit_factor": 4,
                                      "per_page": 1, "max_surveys": 1, "retrieval_minimum_score": 0},
                                     {"survey_queries": ["survey"], "direct_queries": ["direct"]},
                                     cache, client=client)
            active = cache.get_record("openalex.collection", "latest")
            self.assertEqual(counts["direct_results_scanned"], 6)
            self.assertEqual(counts["reference_results_scanned"], 8)
            self.assertEqual(counts["discovered"], 14)
            self.assertEqual(active["survey_sample_size_per_query"], 4)
            self.assertEqual(active["reference_sample_size"], 8)
            self.assertEqual(client.calls, [("survey", 1, 4), ("direct", 1, 6)])

    def test_openalex_uncapped_cursor_search_exhausts_and_detects_cycles(self):
        """Read beyond 20 pages and reject a repeated cursor rather than looping forever."""
        client = OpenAlexClient(request_pause_seconds=0)
        payloads = [{"results": [{"id": f"W{index}"}], "meta": {"next_cursor": str(index)}} for index in range(25)]
        payloads.append({"results": [], "meta": {"next_cursor": "25"}})
        with patch.object(client, "get", side_effect=payloads) as get:
            results, scanned, exhausted = sample_query_results(client, ["query"], None, 1, None)
        self.assertEqual(len(results), 25)
        self.assertEqual(scanned, {"query": 25})
        self.assertEqual(exhausted, {"query"})
        self.assertEqual(get.call_count, 26)
        with patch.object(client, "get", return_value={"results": [{"id": "W"}], "meta": {"next_cursor": "*"}}):
            with self.assertRaisesRegex(RuntimeError, "repeated search cursor"):
                list(client.search("query", 100, None))
