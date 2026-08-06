#!/usr/bin/env python3
"""
GitHub crawler for public IDS, IPS, and related intrusion-detection repositories.

The crawler is intentionally recall-friendly: it does not require exact IDS
topics or names. Instead, it uses multiple GitHub search seeds, then scores
repositories based on repository metadata, topics, README text, and curated
security product terms.

Examples:
  python data-gathering/github-crawler.py
  python data-gathering/github-crawler.py --min-stars 200 --max-repos 75
  python data-gathering/github-crawler.py --extra-seed "industrial intrusion detection"
  python data-gathering/github-crawler.py --readme-mode never --include-research
  python data-gathering/github-crawler.py --anonymous-friendly --max-repos 20

Authentication:
  A token is optional. Without one, the crawler switches to much slower pacing
  so it stays within GitHub's public API limits instead of relying on a PAT.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import http.client
import json
import math
import os
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_limit()


API_ROOT = "https://api.github.com"
SEARCH_REPOSITORIES_URL = f"{API_ROOT}/search/repositories"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 5
EVALUATED_CACHE_VERSION = "1"
TRANSIENT_NETWORK_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    http.client.HTTPException,
    socket.timeout,
    ssl.SSLError,
)

EVALUATED_CACHE_COLUMNS = [
    "cache_version",
    "evaluated_at",
    "full_name",
    "accepted",
    "error",
    "matched_queries",
    "repo_json",
    "readme_text",
    "docs_text",
    "docs_urls",
    "score_card_json",
    "score",
    "primary_family",
    "type_description",
    "repo_role",
    "repo_kind",
    "base_identity",
    "base_detector_status",
    "base_detector_reason",
    "base_detector_score",
    "canonical_product_repo",
    "html_url",
    "description",
    "topics",
    "stars",
    "forks",
    "open_issues",
    "language",
    "archived",
    "updated_at",
    "pushed_at",
    "last_commit_sha",
]


DEFAULT_SEEDS = [
    "intrusion detection system",
    "intrusion prevention system",
    "network intrusion detection",
    "host intrusion detection",
    "host intrusion prevention",
    "wireless intrusion detection",
    "wireless intrusion prevention",
    "provenance intrusion detection",
    "database intrusion detection",
    "database intrusion prevention",
    "collaborative intrusion detection",
    "distributed intrusion detection",
    "network detection and response",
    "network security monitoring",
    "malicious traffic detection",
    "network traffic analysis",
    "security monitoring",
    "threat hunting",
    "threat detection",
    "network sensor",
    "iot intrusion detection",
    "iiot intrusion detection",
    "ics intrusion detection",
    "scada intrusion detection",
    "anomaly based intrusion detection",
    "signature based intrusion detection",
    "machine learning intrusion detection",
    "deep learning intrusion detection",
    "log based intrusion detection",
    "HIDS",
    "NIDS",
    "CIDS",
    "DIDS",
    "NDR",
    "EDR",
    "XDR",
    "Network Detection Response",
    "Endpoint Detection Response",
    "Extended Detection Response",
    "Intrusion",
    "Attack detection",
    # Some repos like Snort do not enrich anything with metadata so it is undiscoverable by any generic search
    # Hence we include the most famous tools as seeds also to broaden the recall. This is alright as many other tools/classifier reference these or build on these.
    "zeek",
    "suricata",
    "snort3",
    "wazuh",
    "ossec",
    "database proxy intrusion detection",
    "IDS",
    "IPS",
    "Machine Learning IDS",
    "Machine Learning Intrusion Detection",
    "Network Anomaly detection",
    "Log anomaly detection",
    "behavior anomaly detection" 
]


CORE_SIGNAL_WEIGHTS = {
    "intrusion detection": 8,
    "intrusion prevention": 8,
    "network intrusion detection": 8,
    "host intrusion detection": 8,
    "wireless intrusion detection": 8,
    "collaborative intrusion detection": 8,
    "distributed intrusion detection": 8,
    "network detection and response": 8,
    "network security monitoring": 6,
    "malicious traffic detection": 8,
    "network traffic analysis": 5,
    "security monitoring": 2,
    "threat hunting": 2,
    "threat detection": 2,
    "deep packet inspection": 6,
    "packet inspection": 6,
    "network sensor": 5,
    "malicious traffic": 6,
    "ids": 7,
    "ips": 7,
    "nids": 7,
    "hids": 7,
    "wids": 7,
    "cids": 7,
    "dids": 7,
    "ndr": 7,
    "nsm": 7,
    "xdr": 7,
    "iot intrusion detection": 7,
    "iiot intrusion detection": 7,
    "ics intrusion detection": 7,
    "scada intrusion detection": 7,
}


DEFAULT_PRODUCT_SIGNALS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "product_signals.csv",
)

SOLUTION_CATEGORIES = (
    "IDS",
    "IPS",
    "HIDS",
    "NIDS",
    "WIDS",
    "PIDS",
    "NIPS",
    "HIPS",
    "DBIDS",
    "DBIPS",
    "WIPS",
    "UNKNOWN",
)


CATEGORY_KEYWORDS = {
    "IDS": {
        "intrusion detection system": 6,
        "intrusion detection": 5,
        "ids": 4,
        "detect intrusions": 4,
        "attack detection": 3,
        "threat detection": 2,
        "malware detection": 2,
        "tampering detection": 3,
    },
    "IPS": {
        "intrusion prevention system": 6,
        "intrusion prevention": 5,
        "ips": 4,
        "packet blocking": 4,
        "block attacks": 4,
        "default deny": 2,
        "firewall": 1,
    },
    "HIDS": {
        "host intrusion detection system": 7,
        "host based intrusion detection": 7,
        "host intrusion detection": 6,
        "hids": 5,
        "endpoint detection": 3,
        "file integrity monitoring": 3,
        "runtime security": 4,
        "runtime application self protection": 4,
        "rasp": 4,
        "remote attestation": 4,
        "attestation": 3,
        "tampering detection": 3,
        "root detection": 3,
        "frida detection": 3,
        "app shielding": 3,
        "application shielding": 3,
        "endpoint security": 3,
        "edr": 3,
        "linux host": 2,
        "ebpf": 3,
        "kprobe": 3,
        "kprobes": 3,
        "kernel hook": 3,
        "execve hook": 3,
        "file hook": 3,
        # "ossec": 4,
        # "wazuh": 4,
        # "falco": 4,
        # "aide": 4,
        # "osquery": 3,
        "sysmon": 3,
    },
    "NIDS": {
        "network intrusion detection system": 7,
        "network intrusion detection": 7,
        "network security monitoring": 3,
        "network traffic analysis": 5,
        "network analysis framework": 4,
        "network sensor": 4,
        "network monitor": 4,
        "network monitoring": 3,
        "nids": 5,
        "nsm": 3,
        "malicious traffic detection": 7,
        "malicious traffic": 5,
        "traffic detection": 4,
        "packet capture": 3,
        "packet analyzer": 3,
        "packet analyser": 3,
        "pcap analyzer": 3,
        "pcap analyser": 3,
        "pcap parser": 3,
        "wireshark": 2,
        "pcap": 3,
        # "suricata": 4,
        # "snort": 4,
        # "zeek": 4,
    },
    "WIDS": {
        "wireless intrusion detection system": 7,
        "wireless intrusion detection": 6,
        "wireless ids": 5,
        "wids": 5,
        "wifi intrusion": 4,
        "wi fi": 2,
        "802 11": 3,
        "wireless": 2,
    },
    "PIDS": {
        "provenance based intrusion detection": 7,
        "provenance intrusion detection": 6,
        "provenance based ids": 5,
        "pids": 5,
        "process provenance": 4,
        "system provenance": 4,
    },
    "NIPS": {
        "network intrusion prevention system": 8,
        "network intrusion prevention": 7,
        "nips": 6,
        "network ips": 5,
        "inline network": 4,
        # "suricata ips": 4,
        # "snort ips": 4,
    },
    "HIPS": {
        "host intrusion prevention system": 8,
        "host intrusion prevention": 7,
        "host based intrusion prevention": 7,
        "hips": 6,
        "endpoint prevention": 4,
        "endpoint security": 3,
        "access control": 3,
        "egress filtering": 3,
        "app shielding": 3,
        "application shielding": 3,
        "runtime application self protection": 4,
        "rasp": 4,
        "runtime enforcement": 4,
        "default deny": 3,
    },
    "DBIDS": {
        "database intrusion detection": 8,
        "database ids": 6,
        "database proxy": 4,
        "database security": 4,
        "honeypot record": 5,
        "honeypots": 4,
        "sql injection detection": 5,
    },
    "DBIPS": {
        "database intrusion prevention": 8,
        "database ips": 6,
        "sql injection prevention": 6,
        "database firewall": 5,
        "query blocking": 4,
    },
    "WIPS": {
        "wireless intrusion prevention system": 8,
        "wireless intrusion prevention": 7,
        "wireless ips": 6,
        "wips": 6,
        "wifi prevention": 4,
    },
}


DETECTION_TYPE_KEYWORDS = {
    "signature-based": (
        "signature based",
        "signature-based",
        "signature",
        "signatures",
        "rules",
        "ruleset",
        "rule set",
        "rule pack",
        "detection rules",
        "alert rules",
        "snort rules",
        "suricata rules",
        "yara",
        "sigma",
        "blocklist",
        "blacklist",
        "ioc",
        "iocs",
        "indicator of compromise",
        "indicators of compromise",
        "known bad",
        "malware trail",
        "malware trails",
        "threat rules",
        "malicious traffic detection",
        "attack detection",
        "threat detection",
    ),
    "anomaly-based": (
        "anomaly",
        "anomalies",
        "anomaly based",
        "anomaly-based",
        "anomaly detection",
        "behavioral",
        "behavioural",
        "baseline",
        "outlier",
        "statistical",
        "beacon detection",
    ),
    "query-based": (
        "query based",
        "query-based",
        "queries",
        "query pack",
        "query language",
        "osquery",
        "kql",
        "kusto",
        "sql query",
        "detection query",
        "native queries",
        "translated queries",
    ),
    "integrity-based": (
        "file integrity",
        "file integrity monitoring",
        "integrity monitoring",
        "integrity check",
        "integrity checks",
        "attestation",
        "remote attestation",
        "secure boot",
        "secureboot",
        "verified boot",
        "verifiedboot",
        "tampering detection",
        "root detection",
        "hooking detection",
        "hook",
        "hooks",
        "kprobe",
        "kprobes",
        "kernel hook",
        "connect hook",
        "create file hook",
        "dns query hook",
        "execve hook",
        "load lkm hook",
        "anti rootkit",
        "rootkit",
        "process injection",
        "porcess injection",
        "frida detection",
        "app shielding",
        "application shielding",
        "runtime application self protection",
    ),
    "threat-intel-based": (
        "threat intelligence",
        "threat intel",
        "osint",
        "indicator of compromise",
        "indicators of compromise",
        "ioc",
        "iocs",
        "misp",
        "urlhaus",
        "malware bazaar",
        "malwarebazaar",
        "threatfox",
        "threat rules",
        "certificate transparency",
        "certstream",
    ),
    "packet-analysis": (
        "packet analysis",
        "packet",
        "packet analyzer",
        "packet analyser",
        "packet capture",
        "pcap analysis",
        "pcap analyzer",
        "pcap analyser",
        "pcap parser",
        "network traffic analysis",
        "network analysis",
        "network monitoring",
        "traffic detection",
        "malicious traffic detection",
        "malicious traffic",
        "wireshark",
    ),
    "DNN-based": (
        "dnn",
        "deep neural network",
        "deep neural networks",
        "deep learning",
        "cnn",
        "lstm",
        "transformer",
        "autoencoder",
    ),
    "ML-based": (
        "machine learning",
        "ml based",
        "ml-based",
        "classifier",
        "classification",
        "random forest",
        "svm",
        "xgboost",
        "k means",
        "sklearn",
    ),
    "Log-based": (
        "log based",
        "log-based",
        "log analysis",
        "log analyzer",
        "loganalyzer",
        "logs",
        "event log",
        "event logs",
        "events",
        "audit log",
        "audit logs",
        "audit trail",
        "audit trails",
        "security event",
        "siem",
        "sysmon",
        "evtx",
        "http intrusion detection",
        "access log",
        "access logs",
        "request log",
        "request logs",
        "analyze logs",
        "log analyzer",
        "threat analyzer",
        "security audit",
        "audit",
        "ebpf-sec",
    ),
    "ANN-based": (
        "ann",
        "artificial neural network",
        "neural network",
        "neural networks",
    ),
    "rule-based": (
        "rule based",
        "rule-based",
        "policy based",
        "policy-based",
        "default deny",
        "iptables",
        "nftables",
        "firewall rules",
        "firewalld",
        "filter",
        "packet filter",
        "ddos mitigation",
        "anti ddos",
        "ipset",
        "ufw",
        "egress filtering",
        "bouncer",
        "ban hosts",
        "ban management",
        "threat rules",
        "rules",
        "blocking",
        "access control",
    ),
    "honeypot-based": (
        "honeypot",
        "honeypots",
        "honeytoken",
        "honeypot record",
    ),
}


DETECTION_TYPE_ORDER = (
    "signature-based",
    "anomaly-based",
    "query-based",
    "integrity-based",
    "threat-intel-based",
    "packet-analysis",
    "DNN-based",
    "ML-based",
    "Log-based",
    "ANN-based",
    "rule-based",
    "honeypot-based",
)


ADJACENT_ONLY_DETECTION_TYPES = {
    "packet-analysis",
    "threat-intel-based",
}

RISKY_STANDALONE_DETECTION_TYPES = ADJACENT_ONLY_DETECTION_TYPES | {
    "integrity-based",
}

RASP_REVIEW_KEYWORDS = (
    "rasp",
    "runtime application self protection",
    "app shielding",
    "application shielding",
)

STRONG_STANDALONE_DETECTOR_KEYWORDS = (
    "intrusion detection system",
    "intrusion prevention system",
    "host intrusion detection",
    "host based intrusion detection",
    "network intrusion detection",
    "wireless intrusion detection",
    "database intrusion detection",
    "provenance intrusion detection",
    "network detection and response",
    "malicious traffic detection",
    "attack detection",
)


PUBLIC_ROLE_LABELS = {
    "deployable_ids": "Deployable IDS",
    "classifier_model": "Classifier/model",
    "platform_framework": "Platform/framework",
    "rules_content": "Rules/content",
    "support_tooling": "Support tooling",
    "research_demo": "Research/demo",
}

PUBLIC_ROLE_QUALITY = {
    "deployable_ids": 5,
    "platform_framework": 4,
    "classifier_model": 3,
    "rules_content": 2,
    "support_tooling": 1,
    "research_demo": 0,
}


DOCS_URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


REPO_KIND_KEYWORDS = {
    "framework": (
        "framework",
        "toolkit",
        "library",
        "sdk",
        "frontend",
        "workbench",
        "network analysis framework",
        "traffic analysis framework",
    ),
    "classifier": (
        "classifier",
        "classification",
        "machine learning model",
        "deep learning model",
        "neural network model",
    ),
    "dataset": (
        "dataset",
        "data set",
        "cicids",
        "kdd cup",
        "unsw nb15",
        "bot iot",
        "training data",
        "pcap dataset",
    ),
    "paper": (
        "paper",
        "survey",
        "publication",
        "journal",
        "conference",
        "thesis",
        "dissertation",
        "manuscript",
    ),
    "list": (
        "awesome",
        "curated list",
        "list of",
        "resource list",
    ),
    "tutorial": (
        "tutorial",
        "notebook",
        "colab",
        "course project",
        "assignment",
        "exercise",
        "demo only",
    ),
    "tooling": (
        "ansible",
        "terraform",
        "helm chart",
        "kustomize",
        "docker compose",
        "docker",
        "puppet",
        "chef cookbook",
        "installer",
        "deployment",
        "package",
        "packages",
        "plugin",
        "integration",
        "connector",
        "bouncer",
        "mcp server",
        "dashboard",
        "gui",
        "management ui",
        "documentation",
    ),
    "ruleset": (
        "ruleset",
        "rule set",
        "signatures",
        "signature pack",
        "detection rules",
        "suricata rules",
        "snort rules",
        "sigma rules",
        "yara rules"
    ),
}


EXPLICIT_BASE_RELATION_KEYWORDS = (
    "based on",
    "based",
    "built on",
    "built with",
    "built around",
    "powered by",
    "on top of",
    "derived from",
    "wraps",
    "wrapper",
    "uses",
    "using",
    "includes",
    "integrates",
)


SUPPORT_DESCRIPTOR_KEYWORDS = (
    (
        "distribution",
        (
            "distribution",
            "distro",
            "iso",
            "appliance",
        ),
    ),
    (
        "suite/wrapper",
        (
            "tool suite",
            "all in one",
            "all-in-one",
            "wrapper",
        ),
    ),
    (
        "management UI",
        (
            "management ui",
            "management interface",
            "dashboard",
            "dashboards",
            "gui",
            "user interface",
            "frontend",
            "web interface",
            "case management",
        ),
    ),
    (
        "ruleset",
        (
            "ruleset",
            "rule set",
            "rule pack",
            "detection rules",
            "alert rules",
            "suricata rules",
            "suricata rule",
            "snort rules",
            "snort rule",
            "sigma rules",
            "sigma rule",
            "wazuh rules",
            "signatures",
        ),
    ),
    (
        "deployment/support tooling",
        (
            "ansible",
            "terraform",
            "helm chart",
            "kustomize",
            "docker compose",
            "docker",
            "installer",
            "deployment",
            "package",
            "packages",
        ),
    ),
    (
        "integration",
        (
            "integration",
            "connector",
            "plugin",
            "bouncer",
            "mcp server",
            "model context protocol",
            "translation",
            "translator",
            "bridge",
        ),
    ),
    (
        "framework/toolkit",
        (
            "framework",
            "toolkit",
            "library",
            "sdk",
            "workbench",
            "network analysis framework",
            "traffic analysis framework",
        ),
    ),
    (
        "threat-hunting content",
        (
            "playbook",
            "hunting campaign",
            "threat hunting",
            "detection engineering",
            "incident response project",
            "security query",
            "queries",
        ),
    ),
    (
        "packet-analysis tool",
        (
            "packet analyzer",
            "packet analyser",
            "pcap analyzer",
            "pcap analyser",
            "pcap parser",
            "packet capture",
            "wireshark",
        ),
    ),
    (
        "testing/simulation tool",
        (
            "testing tool",
            "testing tools",
            "simulator",
            "simulation",
            "flightsim",
            "test traffic",
            "traffic generator",
            "attack simulation",
        ),
    ),
    (
        "documentation/support content",
        (
            "documentation",
            "reference",
            "guide",
            "training",
            "classroom",
            "course",
            "tutorial",
            "learn",
            "beginner",
            "beginners",
            "projects for beginners",
            "90 days",
            "lab",
        ),
    ),
)


GENERIC_ENABLING_LIBRARY_KEYWORDS = (
    "api",
    "library",
    "sdk",
    "development kit",
    "packet filtering library",
    "low level network packet",
    "network packet interception",
)


BUILD_AS_USE_CASE_KEYWORDS = (
    "ideal for building",
    "for building custom",
    "building custom",
    "build custom",
    "can build",
    "can be used to build",
    "use it to build",
)


BASE_DETECTOR_SELF_DESCRIPTION_KEYWORDS = (
    "intrusion detection system",
    "intrusion prevention system",
    "network detection and response",
    "network security monitoring",
    "network intrusion detection",
    "host intrusion detection",
    "wireless intrusion detection",
    "database intrusion detection",
    "provenance intrusion detection",
    "malicious traffic detection",
    "attack detection",
    "threat detection",
    "runtime security",
)


BASE_DETECTOR_RUNTIME_KEYWORDS = (
    "agent",
    "sensor",
    "daemon",
    "engine",
    "monitor",
    "analyzer",
    "analyser",
    "detector",
    "prevention",
    "alerting",
    "real time",
    "real-time",
    "packet inspection",
    "deep packet inspection",
    "packet capture",
    "network traffic analysis",
    "flow analysis",
    "file integrity monitoring",
    "log analysis",
    "runtime enforcement",
)


FRAMEWORK_ROLE_KEYWORDS = (
    "framework",
    "toolkit",
    "library",
    "sdk",
    "workbench",
    "network analysis framework",
    "traffic analysis framework",
    "packet analysis framework",
    "dataset creation toolkit",
    "build detectors",
    "building detectors",
    "build intrusion detection",
    "building intrusion detection",
)


CLASSIFIER_ROLE_KEYWORDS = (
    "classifier",
    "classification",
    "machine learning model",
    "deep learning model",
    "neural network model",
    "trained model",
    "training model",
    "training script",
    "notebook",
    "jupyter",
    "colab",
    "accuracy",
    "f1 score",
)


FULL_SOLUTION_ROLE_KEYWORDS = (
    "daemon",
    "agent",
    "sensor",
    "engine",
    "service",
    "platform",
    "appliance",
    "security suite",
    "prevention system",
    "runtime security",
    "real time",
    "real-time",
    "alerting",
    "monitoring agent",
    "network monitor",
    "host monitor",
)


BASE_HARD_NEGATIVE_KINDS = {
    "dataset",
    "paper",
    "list",
    "tutorial",
    "ruleset",
}


BASE_HARD_NEGATIVE_DESCRIPTORS = {
    "generic enabling library",
    "management UI",
    "ruleset",
    "integration",
    "testing/simulation tool",
    "documentation/support content",
    "threat-hunting content",
}


BASE_DERIVATIVE_DESCRIPTORS = BASE_HARD_NEGATIVE_DESCRIPTORS | {
    "distribution",
    "suite/wrapper",
    "deployment/support tooling",
}

MIN_PARSEABLE_ENGLISH_WORDS = 10
MIN_PARSEABLE_ENGLISH_UNIQUE_WORDS = 7
MIN_PARSEABLE_ENGLISH_LETTERS = 45
MIN_CJK_DOMINANCE_CHARS = 20
MIN_ENGLISH_SHARE_WHEN_CJK_PRESENT = 0.55

CJK_CHAR_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)
ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}")


NEGATIVE_SIGNAL_WEIGHTS = {
    "survey": 5,
    "benchmark": 4,
    "awesome": 5,
    "curated list": 5,
    "tutorial": 4,
    "notebook": 4,
    "course project": 4,
    "assignment": 4,
    "archived": 2,
    "deprecated": 4,
    "unmaintained": 4,
}


RESEARCH_KINDS = {"dataset", "paper", "list", "tutorial"}


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    max_repos: int
    min_stars: int
    min_score: int
    max_pages_per_query: int
    per_page: int
    candidate_multiplier: int
    include_forks: bool
    include_archived: bool
    include_research: bool
    readme_mode: str
    docs_mode: str
    max_docs_pages: int
    include_support_tools: bool
    pushed_within_days: int | None
    out: str
    token: str | None
    extra_seeds: list[str]
    replace_default_seeds: bool
    pause_seconds: float | None
    search_pause_seconds: float | None
    core_pause_seconds: float | None
    anonymous_friendly: bool
    checkpoint_every: int
    checkpoint_out: str | None
    evaluated_cache_in: str | None
    evaluated_cache_out: str | None
    cache_only: bool
    product_signals_path: str | None
    refresh_commit_sha: bool


@dataclass
class SearchCandidate:
    item: dict[str, Any]
    matched_queries: set[str] = field(default_factory=set)


def normalize_text(*parts: str) -> str:
    joined = " ".join(part for part in parts if part)
    lowered = joined.lower()
    cleaned = []
    previous_was_space = False
    for char in lowered:
        if char.isalnum():
            cleaned.append(char)
            previous_was_space = False
        else:
            if not previous_was_space:
                cleaned.append(" ")
            previous_was_space = True
    return "".join(cleaned).strip()


PHRASE_FUZZY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "based",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}

TOKEN_CANONICAL_OVERRIDES = {
    "analysed": "analyze",
    "analyser": "analyze",
    "analysers": "analyze",
    "analyses": "analyze",
    "analysis": "analyze",
    "analytics": "analyze",
    "analyze": "analyze",
    "analyzed": "analyze",
    "analyzer": "analyze",
    "analyzers": "analyze",
    "analyzes": "analyze",
    "analyzing": "analyze",
    "analysing": "analyze",
    "attacks": "attack",
    "blocked": "block",
    "blocking": "block",
    "blocks": "block",
    "databases": "database",
    "db": "database",
    "detected": "detect",
    "detecting": "detect",
    "detection": "detect",
    "detections": "detect",
    "detector": "detect",
    "detectors": "detect",
    "detects": "detect",
    "devices": "device",
    "edr": "host",
    "endpoint": "host",
    "endpoints": "host",
    "events": "event",
    "files": "file",
    "hosts": "host",
    "inspect": "inspect",
    "inspected": "inspect",
    "inspection": "inspect",
    "inspector": "inspect",
    "intrusions": "intrusion",
    "logged": "log",
    "logging": "log",
    "logs": "log",
    "monitored": "monitor",
    "monitoring": "monitor",
    "monitors": "monitor",
    "networks": "network",
    "packets": "packet",
    "prevented": "prevent",
    "preventing": "prevent",
    "prevention": "prevent",
    "prevents": "prevent",
    "protected": "protect",
    "protecting": "protect",
    "protection": "protect",
    "protects": "protect",
    "queries": "query",
    "rules": "rule",
    "signatures": "signature",
    "threats": "threat",
    "wifi": "wireless",
}


def canonical_token(token: str) -> str:
    token = TOKEN_CANONICAL_OVERRIDES.get(token, token)
    if token in TOKEN_CANONICAL_OVERRIDES:
        return TOKEN_CANONICAL_OVERRIDES[token]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and token not in {"ids", "ips", "nids", "hids", "wids"}:
        return token[:-1]
    return token


@lru_cache(maxsize=8192)
def normalized_token_tuple(text: str) -> tuple[str, ...]:
    return tuple(normalize_text(text).split())


@lru_cache(maxsize=8192)
def canonical_token_tuple(text: str) -> tuple[str, ...]:
    return tuple(canonical_token(token) for token in normalized_token_tuple(text))


@lru_cache(maxsize=8192)
def canonical_token_set(text: str) -> frozenset[str]:
    return frozenset(canonical_token_tuple(text))


@lru_cache(maxsize=8192)
def phrase_concepts(phrase: str) -> tuple[str, ...]:
    tokens = normalized_token_tuple(phrase)
    filtered = [
        canonical_token(token)
        for token in tokens
        if token not in PHRASE_FUZZY_STOPWORDS
    ]
    return tuple(filtered or (canonical_token(token) for token in tokens))


def concepts_within_window(text_tokens: tuple[str, ...], required_concepts: frozenset[str]) -> bool:
    if not required_concepts:
        return False
    window_size = min(14, max(6, len(required_concepts) + 5))
    positions = [
        (index, token)
        for index, token in enumerate(text_tokens)
        if token in required_concepts
    ]
    if len(positions) < len(required_concepts):
        return False

    counts: dict[str, int] = defaultdict(int)
    left = 0
    for right, (right_index, token) in enumerate(positions):
        counts[token] += 1
        while right_index - positions[left][0] + 1 > window_size:
            left_token = positions[left][1]
            counts[left_token] -= 1
            if counts[left_token] <= 0:
                del counts[left_token]
            left += 1
        if required_concepts.issubset(counts):
            return True
    return False


def english_parseability(raw_text: str) -> dict[str, Any]:
    cjk_chars = CJK_CHAR_RE.findall(raw_text or "")
    english_words = ENGLISH_WORD_RE.findall(raw_text or "")
    unique_words = {
        word.casefold().strip("._-+#")
        for word in english_words
        if len(word.strip("._-+#")) >= 3
    }
    latin_letters = sum(1 for char in raw_text or "" if ("a" <= char.lower() <= "z"))
    cjk_count = len(cjk_chars)
    english_share = latin_letters / max(1, latin_letters + cjk_count)
    enough_english = (
        len(english_words) >= MIN_PARSEABLE_ENGLISH_WORDS
        and len(unique_words) >= MIN_PARSEABLE_ENGLISH_UNIQUE_WORDS
        and latin_letters >= MIN_PARSEABLE_ENGLISH_LETTERS
    )
    cjk_dominated = cjk_count >= MIN_CJK_DOMINANCE_CHARS and english_share < MIN_ENGLISH_SHARE_WHEN_CJK_PRESENT
    parseable = enough_english and not cjk_dominated
    if not enough_english:
        reason = (
            "insufficient English evidence "
            f"({len(english_words)} words, {len(unique_words)} unique, {latin_letters} Latin letters)"
        )
    elif cjk_dominated:
        reason = f"CJK-dominated text ({cjk_count} CJK chars, English share {english_share:.2f})"
    else:
        reason = "parseable English evidence"
    return {
        "english_parseable": parseable,
        "english_parseability_reason": reason,
        "english_word_count": len(english_words),
        "unique_english_word_count": len(unique_words),
        "latin_letter_count": latin_letters,
        "cjk_char_count": cjk_count,
        "english_latin_cjk_share": round(english_share, 4),
    }


def contains_exact_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    haystack = f" {text} "
    needle = f" {normalized_phrase} "
    return needle in haystack


def contains_phrase(text: str, phrase: str, *, fuzzy: bool = True) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    if contains_exact_phrase(text, normalized_phrase):
        return True
    if not fuzzy:
        return False

    concepts = phrase_concepts(normalized_phrase)
    if not concepts:
        return False

    if len(concepts) == 1:
        concept = concepts[0]
        if len(concept) < 4:
            return False
        return concept in canonical_token_set(text)

    required = frozenset(concepts)
    text_concepts = canonical_token_set(text)
    if not required.issubset(text_concepts):
        return False

    return concepts_within_window(canonical_token_tuple(text), required)


def format_seed_for_search(seed: str) -> str:
    seed = seed.strip()
    return seed


def parse_iso8601(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def lifecycle_status(repo: dict[str, Any]) -> str:
    if repo.get("archived"):
        return "archived"
    pushed_at = parse_iso8601(repo.get("pushed_at"))
    if pushed_at is None:
        return "unknown"
    age_days = (datetime.now(timezone.utc) - pushed_at).days
    if age_days <= 365:
        return "active"
    if age_days <= 730:
        return "stale"
    return "inactive"


def github_headers(token: str | None) -> dict[str, str]:
    resolved_token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-ids-crawler",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"
    return headers


def has_github_token(headers: dict[str, str]) -> bool:
    return "Authorization" in headers


def effective_search_pause(config: Config, authenticated: bool) -> float:
    if config.search_pause_seconds is not None:
        return config.search_pause_seconds
    if config.pause_seconds is not None:
        return config.pause_seconds
    if authenticated or not config.anonymous_friendly:
        return 0.2
    return 7.0


def effective_core_pause(config: Config, authenticated: bool) -> float:
    if config.core_pause_seconds is not None:
        return config.core_pause_seconds
    if config.pause_seconds is not None:
        return config.pause_seconds
    if authenticated or not config.anonymous_friendly:
        return 0.2
    return 70.0


def build_search_query(seed: str, config: Config) -> str:
    parts = [format_seed_for_search(seed), "is:public", f"stars:>={config.min_stars}"]
    if not config.include_forks:
        parts.append("fork:false")
    if not config.include_archived:
        parts.append("archived:false")
    if config.pushed_within_days is not None:
        minimum_push_date = (
            datetime.now(timezone.utc) - timedelta(days=config.pushed_within_days)
        ).strftime("%Y-%m-%d")
        parts.append(f"pushed:>={minimum_push_date}")
    return " ".join(parts)


def github_get_json(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    pause_seconds: float = 0.0,
) -> Any:
    encoded_params = urllib.parse.urlencode(params or {}, doseq=True)
    request_url = f"{url}?{encoded_params}" if encoded_params else url
    backoff_seconds = 1.0

    for attempt in range(1, REQUEST_RETRIES + 1):
        request = urllib.request.Request(request_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read()
                if pause_seconds:
                    time.sleep(pause_seconds)
                return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {403, 429}:
                retry_after = exc.headers.get("Retry-After")
                remaining = exc.headers.get("X-RateLimit-Remaining")
                reset = exc.headers.get("X-RateLimit-Reset")
                if retry_after:
                    wait_seconds = max(1, int(retry_after))
                elif remaining == "0" and reset:
                    wait_seconds = max(1, int(reset) - int(time.time()) + 2)
                else:
                    wait_seconds = int(math.ceil(backoff_seconds))
                print(
                    "Rate limit or access throttle on "
                    f"{request_url}. Waiting {wait_seconds}s before retrying. "
                    "If this run is interrupted during enrichment, fetched/evaluated cache "
                    "checkpoints may exist, but the compact accepted CSV is written only "
                    "after the final batch-classification phase.",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60.0)
                continue
            if 500 <= exc.code < 600 and attempt < REQUEST_RETRIES:
                print(
                    f"GitHub server error {exc.code} on {request_url}. Retrying in {backoff_seconds:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60.0)
                continue
            raise RuntimeError(f"GitHub API request failed ({exc.code}) for {request_url}: {body}") from exc
        except TRANSIENT_NETWORK_ERRORS as exc:
            if attempt >= REQUEST_RETRIES:
                raise RuntimeError(f"Network error while requesting {request_url}: {exc}") from exc
            print(
                f"Network error on {request_url}. Retrying in {backoff_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 60.0)

    raise RuntimeError(f"GitHub API request failed after {REQUEST_RETRIES} attempts: {request_url}")


def fetch_repository(full_name: str, headers: dict[str, str], pause_seconds: float) -> dict[str, Any]:
    return github_get_json(
        f"{API_ROOT}/repos/{full_name}",
        headers=headers,
        pause_seconds=pause_seconds,
    )


def fetch_last_commit_sha(
    full_name: str,
    default_branch: str | None,
    headers: dict[str, str],
    pause_seconds: float,
) -> str:
    if not default_branch:
        return ""
    encoded_ref = urllib.parse.quote(default_branch, safe="")
    try:
        payload = github_get_json(
            f"{API_ROOT}/repos/{full_name}/commits/{encoded_ref}",
            headers=headers,
            pause_seconds=pause_seconds,
        )
    except RuntimeError as exc:
        print(f"Last commit SHA fetch skipped for {full_name}: {exc}", file=sys.stderr)
        return ""
    sha = str(payload.get("sha") or "").strip().lower() if isinstance(payload, dict) else ""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return ""
    return sha


def ensure_last_commit_sha(
    repo: dict[str, Any],
    headers: dict[str, str],
    pause_seconds: float,
) -> dict[str, Any]:
    if repo.get("last_commit_sha"):
        return repo
    full_name = repo.get("full_name") or ""
    if not full_name:
        return repo
    sha = fetch_last_commit_sha(
        full_name,
        repo.get("default_branch"),
        headers=headers,
        pause_seconds=pause_seconds,
    )
    if not sha:
        return repo
    enriched_repo = dict(repo)
    enriched_repo["last_commit_sha"] = sha
    return enriched_repo


def fetch_readme_text(full_name: str, headers: dict[str, str], pause_seconds: float) -> str:
    try:
        payload = github_get_json(
            f"{API_ROOT}/repos/{full_name}/readme",
            headers=headers,
            pause_seconds=pause_seconds,
        )
    except RuntimeError as exc:
        if "(404)" in str(exc):
            return ""
        print(f"README fetch skipped for {full_name}: {exc}", file=sys.stderr)
        return ""

    content = payload.get("content")
    if not content:
        return ""
    try:
        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:
        return ""
    return decoded[:20000]


def clean_docs_url(url: str) -> str:
    return url.rstrip(".,;:")


def likely_docs_url(url: str, repo: dict[str, Any]) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    full_name = (repo.get("full_name") or "").lower()
    repo_name = (repo.get("name") or "").lower()
    if "api.github.com" in host or "raw.githubusercontent.com" in host:
        return False
    if "github.com" in host and full_name and full_name not in f"{host}{path}":
        return False
    docs_markers = (
        "readthedocs",
        "gitbook",
        "docs",
        "documentation",
        "github.io",
        "pages.dev",
        "netlify.app",
        "vercel.app",
    )
    if any(marker in host or marker in path for marker in docs_markers):
        return True
    if repo_name and repo_name in host and not host.endswith("github.com"):
        return True
    return False


def discover_docs_urls(repo: dict[str, Any], readme_text: str, max_urls: int) -> list[str]:
    urls: list[str] = []
    homepage = clean_docs_url(repo.get("homepage") or "")
    if homepage and likely_docs_url(homepage, repo):
        urls.append(homepage)
    for raw_url in DOCS_URL_RE.findall(readme_text or ""):
        url = clean_docs_url(raw_url)
        if likely_docs_url(url, repo) and url not in urls:
            urls.append(url)
        if len(urls) >= max_urls:
            break
    return urls[:max_urls]


def html_to_text(html_payload: str) -> str:
    html_payload = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html_payload)
    html_payload = re.sub(r"(?is)<[^>]+>", " ", html_payload)
    html_payload = html.unescape(html_payload)
    return normalize_text(html_payload)[:30000]


def fetch_docs_text(url: str, pause_seconds: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "User-Agent": "github-ids-crawler",
        },
    )
    backoff_seconds = 1.0
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read(250000)
                if pause_seconds:
                    time.sleep(pause_seconds)
                break
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < REQUEST_RETRIES:
                print(
                    f"Docs fetch transient HTTP {exc.code} for {url}. Retrying in {backoff_seconds:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30.0)
                continue
            print(f"Docs fetch skipped for {url}: HTTP {exc.code}", file=sys.stderr)
            return ""
        except TRANSIENT_NETWORK_ERRORS as exc:
            if attempt < REQUEST_RETRIES:
                print(
                    f"Docs fetch network error for {url}: {exc}. Retrying in {backoff_seconds:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30.0)
                continue
            print(f"Docs fetch skipped for {url}: {exc}", file=sys.stderr)
            return ""
        except Exception as exc:
            print(f"Docs fetch skipped for {url}: {exc}", file=sys.stderr)
            return ""
    else:
        return ""
    text = payload.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in text[:500].lower():
        return html_to_text(text)
    return normalize_text(text)[:30000]


def infer_repo_kind(normalized_metadata_text: str) -> str:
    for kind, keywords in REPO_KIND_KEYWORDS.items():
        for keyword in keywords:
            if contains_phrase(normalized_metadata_text, keyword):
                return kind
    return "system"


def meaningful_hits(hits: list[str]) -> list[str]:
    return [
        hit
        for hit in hits
        if hit not in {"ids", "ips", "nids", "hids", "wids", "cids", "dids", "ndr", "nsm", "xdr"}
    ]


def ordered_unique(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_load(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def split_signal_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(";") if item.strip())


def load_product_signals(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        print("Product-signal registry disabled.", file=sys.stderr)
        return {}
    if not os.path.exists(path):
        print(f"Product-signal registry not found, continuing without it: {path}", file=sys.stderr)
        return {}

    signals: dict[str, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            product = normalize_text(row.get("name") or "")
            if not product:
                continue
            try:
                weight = int(row.get("weight") or 1)
            except ValueError:
                weight = 1
            signals[product] = {
                "display_name": (row.get("display_name") or product.title()).strip(),
                "weight": weight,
                "families": split_signal_list(row.get("families") or ""),
                "upstreams": tuple(upstream.lower() for upstream in split_signal_list(row.get("upstreams") or "")),
                "type_hints": split_signal_list(row.get("type_hints") or ""),
            }

    print(f"Loaded {len(signals)} product signals from {path}", file=sys.stderr)
    return signals


def contains_any_phrase(normalized_text: str, phrases: tuple[str, ...]) -> bool:
    return any(contains_phrase(normalized_text, phrase) for phrase in phrases)


def matching_phrases(normalized_text: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if contains_phrase(normalized_text, phrase)]


def product_display_name(product: str, product_signals: dict[str, dict[str, Any]]) -> str:
    return product_signals.get(product, {}).get("display_name") or product.title()


def noncanonical_product_hits(
    repo: dict[str, Any],
    product_hits: list[str],
    product_signals: dict[str, dict[str, Any]],
) -> list[str]:
    full_name = (repo.get("full_name") or "").lower()
    products = []
    for product in product_hits:
        upstreams = product_signals.get(product, {}).get("upstreams", ())
        if full_name in upstreams:
            continue
        products.append(product_display_name(product, product_signals))
    return ordered_unique(products)


def product_label(products: list[str], max_products: int = 3) -> str:
    if not products:
        return ""
    shown = products[:max_products]
    if len(products) > max_products:
        shown.append("others")
    return "/".join(shown)


def has_base_detector_runtime_text(normalized_text: str) -> bool:
    if not (
        contains_any_phrase(normalized_text, BASE_DETECTOR_SELF_DESCRIPTION_KEYWORDS)
        and contains_any_phrase(normalized_text, BASE_DETECTOR_RUNTIME_KEYWORDS)
    ):
        return False
    return not (
        contains_any_phrase(normalized_text, GENERIC_ENABLING_LIBRARY_KEYWORDS)
        and contains_any_phrase(normalized_text, BUILD_AS_USE_CASE_KEYWORDS)
    )


def support_descriptor(normalized_text: str, repo_kind: str) -> str:
    if repo_kind == "ruleset":
        return "ruleset"
    if repo_kind == "classifier":
        return "classifier/prototype"
    if (
        repo_kind == "framework"
        and contains_any_phrase(normalized_text, GENERIC_ENABLING_LIBRARY_KEYWORDS)
        and contains_any_phrase(normalized_text, BUILD_AS_USE_CASE_KEYWORDS)
    ):
        return "generic enabling library"

    embedded_runtime_feature_descriptors = {
        "management UI",
        "ruleset",
        "deployment/support tooling",
        "integration",
        "packet-analysis tool",
        "documentation/support content",
    }
    base_runtime_system = repo_kind == "system" and has_base_detector_runtime_text(normalized_text)
    for descriptor, keywords in SUPPORT_DESCRIPTOR_KEYWORDS:
        if base_runtime_system and descriptor in embedded_runtime_feature_descriptors:
            continue
        if contains_any_phrase(normalized_text, keywords):
            return descriptor
    if repo_kind == "tooling":
        return "deployment/support tooling"
    if repo_kind == "framework":
        return "framework/toolkit"
    return ""


def infer_product_relation_descriptors(
    repo: dict[str, Any],
    product_hits: list[str],
    repo_kind: str,
    normalized_text: str,
    product_signals: dict[str, dict[str, Any]],
) -> list[str]:
    products = noncanonical_product_hits(repo, product_hits, product_signals)
    if not products:
        return []

    base_label = product_label(products)
    descriptor = support_descriptor(normalized_text, repo_kind)
    explicit_base_relation = contains_any_phrase(normalized_text, EXPLICIT_BASE_RELATION_KEYWORDS)

    if descriptor in {"distribution", "suite/wrapper", "framework/toolkit"}:
        return [f"{base_label}-based {descriptor}"]
    if descriptor in {"management UI", "ruleset", "deployment/support tooling", "integration"}:
        return [f"{base_label} {descriptor}"]
    if descriptor:
        return [f"{base_label}-related {descriptor}"]
    if explicit_base_relation:
        return [f"{base_label}-based"]
    return [f"{base_label}-related support tool"]


def infer_support_type_descriptors(
    normalized_text: str,
    repo_kind: str,
    repo_role: str,
    primary_family: str,
) -> list[str]:
    descriptors = []
    descriptor = support_descriptor(normalized_text, repo_kind)
    if descriptor:
        descriptors.append(descriptor)
    elif repo_role == "framework":
        descriptors.append("framework/toolkit")
    elif repo_role == "classifier":
        descriptors.append("classifier/prototype")
    if primary_family != "UNKNOWN" and not descriptors:
        descriptors.append(f"{primary_family} tool (method unclear)")
    if not descriptors and contains_phrase(normalized_text, "threat detection"):
        descriptors.append("threat-detection support tool")
    if not descriptors and contains_phrase(normalized_text, "threat hunting"):
        descriptors.append("threat-hunting support tool")
    return descriptors


def type_description(
    detection_types: list[str],
    product_relation_descriptors: list[str],
    support_type_descriptors: list[str],
) -> str:
    known_detection_types = [label for label in detection_types if label != "UNKNOWN"]
    parts = list(known_detection_types)
    parts.extend(product_relation_descriptors)
    if not parts:
        parts.extend(support_type_descriptors)
    return "; ".join(ordered_unique(parts)) or "UNKNOWN"


def infer_solution_category(category_scores: dict[str, int], normalized_text: str) -> str:
    scores = defaultdict(int, category_scores)

    if scores["NIDS"] >= 4 and scores["IPS"] >= 4:
        scores["NIPS"] += min(scores["NIDS"], scores["IPS"]) + 2
    if scores["HIDS"] >= 4 and scores["IPS"] >= 4:
        scores["HIPS"] += min(scores["HIDS"], scores["IPS"]) + 2
    if scores["WIDS"] >= 4 and scores["IPS"] >= 4:
        scores["WIPS"] += min(scores["WIDS"], scores["IPS"]) + 2
    if scores["DBIDS"] >= 4 and scores["IPS"] >= 4:
        scores["DBIPS"] += min(scores["DBIDS"], scores["IPS"]) + 2

    for category in ("NIPS", "HIPS", "WIPS", "DBIPS"):
        if contains_phrase(normalized_text, category.lower()):
            scores[category] += 6

    candidates = [
        (category, scores[category])
        for category in SOLUTION_CATEGORIES
        if category != "UNKNOWN" and scores[category] > 0
    ]
    if not candidates:
        return "UNKNOWN"

    category_priority = {
        "DBIPS": 0,
        "DBIDS": 1,
        "WIPS": 2,
        "WIDS": 3,
        "HIPS": 4,
        "HIDS": 5,
        "NIPS": 6,
        "NIDS": 7,
        "PIDS": 8,
        "IPS": 9,
        "IDS": 10,
    }
    category, score = sorted(
        candidates,
        key=lambda item: (-item[1], category_priority.get(item[0], 99), item[0]),
    )[0]
    return category if score >= 3 else "UNKNOWN"


def infer_detection_types(
    normalized_text: str,
    product_hits: list[str],
    product_signals: dict[str, dict[str, Any]],
) -> list[str]:
    hits: list[str] = []
    for label in DETECTION_TYPE_ORDER:
        if any(contains_phrase(normalized_text, phrase) for phrase in DETECTION_TYPE_KEYWORDS[label]):
            hits.append(label)

    for product in product_hits:
        for hint in product_signals.get(product, {}).get("type_hints", ()):
            if hint in DETECTION_TYPE_ORDER and hint not in hits:
                hits.append(hint)

    if "DNN-based" in hits and "ANN-based" in hits:
        hits.remove("ANN-based")
    if "DNN-based" in hits and "ML-based" not in hits:
        hits.append("ML-based")
    return [label for label in DETECTION_TYPE_ORDER if label in hits] or ["UNKNOWN"]


def infer_repo_role(repo: dict[str, Any], repo_kind: str, detection_types: list[str], normalized_text: str) -> str:
    if repo_kind == "framework":
        return "framework"
    if repo_kind == "classifier":
        return "classifier"
    if has_base_detector_runtime_text(normalized_text):
        return "full"
    if any(
        contains_phrase(normalized_text, phrase)
        for phrase in FRAMEWORK_ROLE_KEYWORDS
    ):
        return "framework"
    classifier_signals = {"DNN-based", "ML-based", "ANN-based"}
    if (
        classifier_signals.intersection(detection_types)
        or contains_any_phrase(normalized_text, CLASSIFIER_ROLE_KEYWORDS)
    ) and not contains_any_phrase(
        normalized_text,
        FULL_SOLUTION_ROLE_KEYWORDS,
    ):
        return "classifier"
    return "full"


TOOL_IDENTITY_STOP_WORDS = {
    "ids",
    "ips",
    "hids",
    "nids",
    "wids",
    "intrusion",
    "detection",
    "prevention",
    "system",
    "security",
    "tool",
    "tools",
    "project",
    "framework",
    "engine",
    "agent",
    "sensor",
    "docker",
    "helm",
    "ansible",
    "rules",
    "ruleset",
}

REPO_FAMILY_VARIANT_WORDS = TOOL_IDENTITY_STOP_WORDS | {
    "android",
    "api",
    "app",
    "archive",
    "archived",
    "chart",
    "charts",
    "cli",
    "community",
    "compose",
    "connector",
    "deploy",
    "deployment",
    "docs",
    "documentation",
    "examples",
    "extension",
    "extensions",
    "flutter",
    "frontend",
    "gui",
    "install",
    "installer",
    "integration",
    "ios",
    "k8s",
    "kubernetes",
    "legacy",
    "library",
    "old",
    "package",
    "packages",
    "plugin",
    "plugins",
    "python",
    "sdk",
    "server",
    "ui",
    "web",
}

REPO_NONMAIN_VARIANT_WORDS = {
    "android",
    "ansible",
    "api",
    "app",
    "archive",
    "archived",
    "chart",
    "charts",
    "cli",
    "compose",
    "connector",
    "deploy",
    "deployment",
    "deprecated",
    "docs",
    "documentation",
    "docker",
    "examples",
    "extension",
    "extensions",
    "flutter",
    "frontend",
    "gui",
    "helm",
    "install",
    "installer",
    "integration",
    "ios",
    "k8s",
    "kubernetes",
    "legacy",
    "library",
    "old",
    "package",
    "packages",
    "plugin",
    "plugins",
    "python",
    "sdk",
    "ui",
    "web",
}

GENERIC_SINGLE_TOKEN_FAMILY_WORDS = {
    "agent",
    "api",
    "app",
    "core",
    "data",
    "docs",
    "monitor",
    "network",
    "server",
    "service",
    "web",
}


def identity_words(value: str) -> list[str]:
    return [word for word in normalize_text(value).split() if word]


def normalized_tool_identity(value: str) -> str:
    text = normalize_text(value)
    words = [word for word in text.split() if word not in TOOL_IDENTITY_STOP_WORDS]
    return "-".join(words) or text.replace(" ", "-")


def repo_family_identity(repo_name: str) -> str:
    words = [
        word
        for word in identity_words(repo_name)
        if word not in REPO_FAMILY_VARIANT_WORDS
    ]
    if len(words) >= 2:
        return "-".join(words)
    if len(words) == 1 and len(words[0]) >= 4 and words[0] not in GENERIC_SINGLE_TOKEN_FAMILY_WORDS:
        return words[0]
    return normalized_tool_identity(repo_name)


def repo_nonmain_variant_penalty(row: dict[str, Any]) -> int:
    full_name = (row.get("full_name") or "").lower()
    repo_name = full_name.split("/", 1)[-1]
    words = set(identity_words(repo_name))
    penalty = len(words.intersection(REPO_NONMAIN_VARIANT_WORDS))
    if str(row.get("archived") or "").lower() == "true":
        penalty += 2
    return penalty


def repo_nonmain_variant_penalty_for_repo(repo: dict[str, Any]) -> int:
    repo_name = (repo.get("name") or (repo.get("full_name") or "").split("/")[-1]).lower()
    words = set(identity_words(repo_name))
    penalty = len(words.intersection(REPO_NONMAIN_VARIANT_WORDS))
    if repo.get("archived"):
        penalty += 2
    if repo.get("fork"):
        penalty += 2
    return penalty


def infer_main_repository_signal(repo: dict[str, Any], score_card: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    variant_penalty = repo_nonmain_variant_penalty_for_repo(repo)

    if not repo.get("fork"):
        score += 2
        reasons.append("not a fork")
    if not repo.get("archived"):
        score += 1
        reasons.append("not archived")
    if variant_penalty == 0:
        score += 2
        reasons.append("main-like repository name")
    else:
        score -= variant_penalty
        reasons.append(f"side-variant name penalty {variant_penalty}")

    if score_card.get("repo_kind") == "system":
        score += 2
        reasons.append("system repository")
    elif score_card.get("repo_kind") in BASE_HARD_NEGATIVE_KINDS:
        score -= 3
        reasons.append(f"{score_card.get('repo_kind')} repository")
    elif score_card.get("repo_kind") == "tooling":
        score -= 2
        reasons.append("tooling repository")

    role = score_card.get("role")
    if role == "deployable_ids":
        score += 3
        reasons.append("deployable detector role")
    elif role == "platform_framework":
        score += 1
        reasons.append("platform/framework role")
    elif role in {"rules_content", "support_tooling", "research_demo"}:
        score -= 2
        reasons.append(f"{role} role")

    if score_card.get("base_detector_status") == "base":
        score += 2
        reasons.append("base detector evidence")
    elif score_card.get("base_detector_status") == "support":
        score -= 1
        reasons.append("support repository evidence")

    stars = int(repo.get("stargazers_count") or 0)
    if stars >= 500:
        score += 1
        reasons.append("popular repository")
    if stars >= 2000:
        score += 1
        reasons.append("very popular repository")

    pushed_at = parse_iso8601(repo.get("pushed_at"))
    if pushed_at is not None:
        age_days = (datetime.now(timezone.utc) - pushed_at).days
        if age_days <= 730:
            score += 1
            reasons.append("recently maintained")
        elif age_days > 1825:
            score -= 1
            reasons.append("long inactive")

    return {
        "main_repository": score >= 6,
        "main_repository_score": score,
        "main_repository_reason": "; ".join(ordered_unique(reasons)),
    }


def repo_owner_and_name(repo: dict[str, Any]) -> tuple[str, str]:
    full_name = (repo.get("full_name") or "").lower()
    if "/" in full_name:
        owner, name = full_name.split("/", 1)
        return owner, name
    return "", (repo.get("name") or full_name).lower()


def upstream_owners(product: str, product_signals: dict[str, dict[str, Any]]) -> set[str]:
    owners: set[str] = set()
    for upstream in product_signals.get(product, {}).get("upstreams", ()):
        if "/" in upstream:
            owners.add(upstream.split("/", 1)[0].lower())
    return owners


def repo_name_matches_product(repo_name: str, product: str, product_signals: dict[str, dict[str, Any]]) -> bool:
    repo_identity = normalized_tool_identity(repo_name)
    product_identity = normalized_tool_identity(product_display_name(product, product_signals))
    return (
        repo_identity == product_identity
        or repo_identity.startswith(f"{product_identity}-")
        or repo_identity.endswith(f"-{product_identity}")
    )


def infer_base_identity(
    repo: dict[str, Any],
    product_hits: list[str],
    canonical_product_repo: bool,
    product_signals: dict[str, dict[str, Any]],
) -> str:
    full_name = (repo.get("full_name") or "").lower()
    owner, repo_name = repo_owner_and_name(repo)
    if canonical_product_repo:
        for product in product_hits:
            upstreams = product_signals.get(product, {}).get("upstreams", ())
            if full_name in upstreams:
                return normalized_tool_identity(product_display_name(product, product_signals))
    for product in product_hits:
        if owner in upstream_owners(product, product_signals) and repo_name_matches_product(
            repo_name,
            product,
            product_signals,
        ):
            return normalized_tool_identity(product_display_name(product, product_signals))
    return normalized_tool_identity(repo.get("name") or repo.get("full_name") or "")


def infer_base_detector_status(
    repo: dict[str, Any],
    score_card: dict[str, Any],
) -> tuple[str, str, int]:
    normalized_text = score_card["normalized_text"]
    repo_kind = score_card["repo_kind"]
    repo_role = score_card["repo_role"]
    descriptor = support_descriptor(normalized_text, repo_kind)
    self_descriptions = matching_phrases(normalized_text, BASE_DETECTOR_SELF_DESCRIPTION_KEYWORDS)
    runtime_markers = matching_phrases(normalized_text, BASE_DETECTOR_RUNTIME_KEYWORDS)
    explicit_base_relation = contains_any_phrase(normalized_text, EXPLICIT_BASE_RELATION_KEYWORDS)

    base_score = 0
    reasons: list[str] = []
    if score_card["meaningful_core_hits"]:
        base_score += 3
        reasons.append("direct IDS/IPS wording")
    if score_card["primary_family"] != "UNKNOWN":
        base_score += 2
        reasons.append(f"{score_card['primary_family']} category evidence")
    if repo_role == "full":
        base_score += 2
        reasons.append("runtime/full-solution role")
    elif repo_role == "framework":
        base_score += 1
        reasons.append("framework role")
    elif repo_role == "classifier":
        base_score -= 2
        reasons.append("classifier/prototype role")
    if self_descriptions:
        base_score += min(3, len(self_descriptions))
        reasons.append("self-described detector")
    if runtime_markers:
        base_score += min(3, len(runtime_markers))
        reasons.append("runtime detector component")

    if repo_kind in BASE_HARD_NEGATIVE_KINDS:
        return "support", f"{repo_kind} repository", base_score

    if (
        descriptor == "threat-hunting content"
        and repo_role == "full"
        and base_score >= 8
        and self_descriptions
        and runtime_markers
    ):
        return "base", "; ".join(ordered_unique(reasons)), base_score

    if descriptor in BASE_HARD_NEGATIVE_DESCRIPTORS:
        return "support", descriptor, base_score

    if descriptor in {"distribution", "suite/wrapper"}:
        return "support", descriptor, base_score

    if (
        descriptor == "deployment/support tooling"
        and repo_kind == "tooling"
        and base_score < 8
    ):
        return "support", descriptor, base_score

    if (
        score_card["product_relation_descriptors"]
        and descriptor in BASE_DERIVATIVE_DESCRIPTORS
        and base_score < 8
    ):
        return "support", "; ".join(score_card["product_relation_descriptors"]), base_score

    if (
        score_card["product_relation_descriptors"]
        and explicit_base_relation
        and base_score < 7
    ):
        return "support", "; ".join(score_card["product_relation_descriptors"]), base_score

    if repo_role == "classifier" and base_score < 8:
        return "support", "classifier/prototype", base_score

    if base_score >= 6:
        return "base", "; ".join(ordered_unique(reasons)), base_score

    return "support", "insufficient standalone detector evidence", base_score


def known_detection_type_set(score_card: dict[str, Any]) -> set[str]:
    return {
        label
        for label in score_card.get("detection_types", [])
        if label and label != "UNKNOWN"
    }


def method_unclear_only(score_card: dict[str, Any]) -> bool:
    if known_detection_type_set(score_card):
        return False
    descriptors = score_card.get("support_type_descriptors") or []
    return bool(descriptors) and all(
        descriptor.endswith("tool (method unclear)")
        for descriptor in descriptors
    )


def adjacent_method_only(score_card: dict[str, Any]) -> bool:
    methods = known_detection_type_set(score_card)
    return bool(methods) and methods.issubset(ADJACENT_ONLY_DETECTION_TYPES)


def integrity_method_only(score_card: dict[str, Any]) -> bool:
    return known_detection_type_set(score_card) == {"integrity-based"}


def rasp_like(score_card: dict[str, Any]) -> bool:
    return contains_any_phrase(score_card.get("normalized_text", ""), RASP_REVIEW_KEYWORDS)


def has_non_risky_detector_method(score_card: dict[str, Any]) -> bool:
    return bool(known_detection_type_set(score_card) - RISKY_STANDALONE_DETECTION_TYPES)


def has_strong_standalone_detector_wording(score_card: dict[str, Any]) -> bool:
    normalized_text = score_card.get("normalized_text", "")
    return contains_any_phrase(normalized_text, STRONG_STANDALONE_DETECTOR_KEYWORDS)


def passes_integrity_review(score_card: dict[str, Any]) -> bool:
    if not has_strong_standalone_detector_wording(score_card):
        return False
    return int(score_card.get("base_detector_score") or 0) >= 8


def high_confidence_standalone_detector(score_card: dict[str, Any]) -> bool:
    if score_card.get("repo_kind") != "system":
        return False
    if score_card.get("repo_role") != "full":
        return False
    if score_card.get("primary_family") == "UNKNOWN":
        return False
    if not score_card.get("english_parseable", False):
        return False
    if int(score_card.get("base_detector_score") or 0) < 8:
        return False
    return has_strong_standalone_detector_wording(score_card) or bool(score_card.get("meaningful_core_hits"))


def deployable_detector_evidence(score_card: dict[str, Any]) -> bool:
    if score_card.get("primary_family") == "UNKNOWN":
        return False
    if not score_card.get("english_parseable", False):
        return False
    if score_card.get("base_detector_status") != "base":
        return False
    if int(score_card.get("base_detector_score") or 0) < 8:
        return False
    return has_strong_standalone_detector_wording(score_card) or bool(score_card.get("meaningful_core_hits"))


def strong_runtime_detector_evidence(score_card: dict[str, Any]) -> bool:
    if score_card.get("repo_kind") != "system":
        return False
    if score_card.get("repo_role") != "full":
        return False
    if score_card.get("base_detector_status") != "base":
        return False
    if score_card.get("primary_family") == "UNKNOWN":
        return False
    if not score_card.get("english_parseable", False):
        return False
    if int(score_card.get("base_detector_score") or 0) < 10:
        return False
    return has_strong_standalone_detector_wording(score_card) or bool(score_card.get("meaningful_core_hits"))


def public_role_result(role: str, confidence: str, evidence: str) -> dict[str, str]:
    return {
        "role": role,
        "role_label": PUBLIC_ROLE_LABELS[role],
        "role_confidence": confidence,
        "role_evidence": evidence,
    }


def infer_public_repo_role(repo: dict[str, Any], score_card: dict[str, Any]) -> dict[str, str]:
    normalized_text = score_card.get("normalized_text", "")
    repo_kind = score_card.get("repo_kind") or "system"
    repo_role = score_card.get("repo_role") or "full"
    descriptor = support_descriptor(normalized_text, repo_kind)
    full_name = (repo.get("full_name") or "").lower()
    language = (repo.get("language") or "").lower()
    detection_types = set(score_card.get("detection_types") or [])

    if "security-onion" in full_name or "securityonion" in full_name:
        return public_role_result("platform_framework", "high", "known wrapper/distribution project")

    if repo_kind in RESEARCH_KINDS:
        return public_role_result("research_demo", "high", f"{repo_kind} repository")

    if descriptor == "ruleset" or repo_kind == "ruleset":
        return public_role_result("rules_content", "high", "ruleset/content repository")

    if descriptor in {"distribution", "suite/wrapper"}:
        return public_role_result("platform_framework", "high", descriptor)

    if descriptor in {
        "deployment/support tooling",
        "integration",
        "generic enabling library",
        "packet-analysis tool",
        "testing/simulation tool",
        "documentation/support content",
    }:
        return public_role_result("support_tooling", "high", descriptor)

    if strong_runtime_detector_evidence(score_card):
        evidence = score_card.get("base_detector_reason") if score_card.get("base_detector_status") == "base" else ""
        return public_role_result(
            "deployable_ids",
            "high",
            evidence or "strong runtime detector evidence",
        )

    if deployable_detector_evidence(score_card):
        return public_role_result(
            "deployable_ids",
            "high",
            score_card.get("base_detector_reason") or "standalone runtime detector evidence",
        )

    classifier_signals = {"DNN-based", "ML-based", "ANN-based"}
    if (
        repo_role == "classifier"
        or repo_kind == "classifier"
        or classifier_signals.intersection(detection_types)
        or contains_any_phrase(normalized_text, CLASSIFIER_ROLE_KEYWORDS)
    ):
        return public_role_result("classifier_model", "high", "classifier/model or training evidence")

    if descriptor == "framework/toolkit" or repo_kind == "framework" or repo_role == "framework":
        return public_role_result("platform_framework", "medium", descriptor or "framework/toolkit evidence")

    if descriptor == "threat-hunting content":
        return public_role_result("rules_content", "medium", descriptor)

    if descriptor == "management UI":
        return public_role_result("support_tooling", "high", descriptor)

    if language == "jupyter notebook":
        return public_role_result("research_demo", "medium", "notebook/research artifact")

    if score_card.get("base_detector_status") == "base":
        return public_role_result(
            "deployable_ids",
            "medium",
            score_card.get("base_detector_reason") or "base detector evidence",
        )

    return public_role_result("support_tooling", "low", "insufficient standalone deployable IDS evidence")


def passes_rasp_review(score_card: dict[str, Any]) -> bool:
    if not has_strong_standalone_detector_wording(score_card):
        return False
    return has_non_risky_detector_method(score_card)


def method_review_flags(score_card: dict[str, Any]) -> list[str]:
    flags = []
    if adjacent_method_only(score_card):
        flags.append("adjacent-method-only")
    if method_unclear_only(score_card):
        flags.append("method-unclear-only")
    if integrity_method_only(score_card):
        flags.append("integrity-method-only")
    if rasp_like(score_card):
        flags.append("rasp-like")
    return flags


def score_repository(
    repo: dict[str, Any],
    matched_queries: set[str],
    readme_text: str,
    docs_text: str = "",
    product_signals: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    product_signals = {}
    topics = repo.get("topics") or []
    description = repo.get("description") or ""
    full_name = (repo.get("full_name") or "").lower()
    raw_text = " ".join(
        part
        for part in (
            repo.get("name") or "",
            description,
            " ".join(topics),
            readme_text,
            docs_text,
        )
        if part
    )
    parseability = english_parseability(raw_text)
    normalized_metadata_text = normalize_text(
        repo.get("name") or "",
        description,
        " ".join(topics),
    )
    normalized_text = normalize_text(raw_text)

    score = 0
    core_hits: list[str] = []
    metadata_core_hits: list[str] = []
    product_hits: list[str] = []
    metadata_product_hits: list[str] = []
    negative_hits: list[str] = []
    category_scores: dict[str, int] = defaultdict(int)
    canonical_product_repo = False

    for phrase, weight in CORE_SIGNAL_WEIGHTS.items():
        if contains_phrase(normalized_text, phrase):
            score += weight
            core_hits.append(phrase)
        if contains_phrase(normalized_metadata_text, phrase):
            metadata_core_hits.append(phrase)

    for product, metadata in product_signals.items():
        if contains_phrase(normalized_text, product, fuzzy=False):
            score += metadata["weight"]
            product_hits.append(product)
            for family in metadata["families"]:
                category_scores[family] += metadata["weight"]
        if contains_phrase(normalized_metadata_text, product, fuzzy=False):
            metadata_product_hits.append(product)
        if full_name in metadata.get("upstreams", ()):
            canonical_product_repo = True

    for family, keywords in CATEGORY_KEYWORDS.items():
        best_family_keyword_weight = 0
        for keyword, weight in keywords.items():
            if contains_phrase(normalized_text, keyword):
                best_family_keyword_weight = max(best_family_keyword_weight, weight)
        if best_family_keyword_weight:
            category_scores[family] += best_family_keyword_weight

    for keyword, penalty in NEGATIVE_SIGNAL_WEIGHTS.items():
        if contains_phrase(normalized_text, keyword):
            score -= penalty
            negative_hits.append(keyword)

    query_boost = min(len(matched_queries), 5)
    score += query_boost

    stars = int(repo.get("stargazers_count") or 0)
    if stars >= 100:
        score += 1
    if stars >= 500:
        score += 1
    if stars >= 2000:
        score += 1
    if stars >= 10000:
        score += 1

    repo_kind = infer_repo_kind(normalized_metadata_text)
    if repo_kind == "system":
        score += 2
    elif repo_kind == "ruleset":
        score += 1
    elif repo_kind in {"framework", "classifier"}:
        score += 1
    elif repo_kind == "tooling":
        score -= 2
    else:
        score -= 5

    meaningful_core_hits = meaningful_hits(core_hits)
    meaningful_metadata_core_hits = meaningful_hits(metadata_core_hits)

    matched_families = [
        family
        for family, family_score in sorted(
            category_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if family in SOLUTION_CATEGORIES and family != "UNKNOWN" and family_score >= 4
    ]
    primary_family = infer_solution_category(category_scores, normalized_text)
    detection_types = infer_detection_types(normalized_text, product_hits, product_signals)
    repo_role = infer_repo_role(repo, repo_kind, detection_types, normalized_text)
    product_relation_descriptors = infer_product_relation_descriptors(
        repo,
        product_hits,
        repo_kind,
        normalized_text,
        product_signals,
    )
    support_type_descriptors = infer_support_type_descriptors(
        normalized_text,
        repo_kind,
        repo_role,
        primary_family,
    )
    inferred_type_description = type_description(
        detection_types,
        product_relation_descriptors,
        support_type_descriptors,
    )
    if primary_family != "UNKNOWN":
        score += 2
    if repo_role == "full":
        score += 1
    if product_relation_descriptors:
        score += 2
    elif detection_types == ["UNKNOWN"] and support_type_descriptors:
        score += 1

    evidence = []
    for item in core_hits[:5]:
        evidence.append(item)
    for item in product_hits[:5]:
        if item not in evidence:
            evidence.append(item)
    for item in sorted(matched_queries)[:5]:
        if item not in evidence:
            evidence.append(item)

    result = {
        "normalized_text": normalized_text,
        "score": score,
        "core_hits": core_hits,
        "metadata_core_hits": metadata_core_hits,
        "meaningful_core_hits": meaningful_core_hits,
        "meaningful_metadata_core_hits": meaningful_metadata_core_hits,
        "product_hits": product_hits,
        "metadata_product_hits": metadata_product_hits,
        "canonical_product_repo": canonical_product_repo,
        "negative_hits": negative_hits,
        "primary_family": primary_family,
        "matched_families": matched_families,
        "detection_types": detection_types,
        "product_relation_descriptors": product_relation_descriptors,
        "support_type_descriptors": support_type_descriptors,
        "type_description": inferred_type_description,
        "repo_role": repo_role,
        "repo_kind": repo_kind,
        "evidence": evidence,
    }
    result.update(parseability)
    base_status, base_reason, base_score = infer_base_detector_status(repo, result)
    result["base_detector_status"] = base_status
    result["base_detector_reason"] = base_reason
    result["base_detector_score"] = base_score
    result["base_identity"] = infer_base_identity(repo, product_hits, canonical_product_repo, product_signals)
    result["method_review_flags"] = method_review_flags(result)
    result.update(infer_public_repo_role(repo, result))
    result.update(infer_main_repository_signal(repo, result))
    return result


def should_fetch_readme(
    repo: dict[str, Any],
    provisional_score: dict[str, Any],
    config: Config,
    authenticated: bool,
) -> bool:
    if config.readme_mode == "always":
        return True
    if config.readme_mode == "never":
        return False
    return True


def accepted_repository(
    repo: dict[str, Any],
    score_card: dict[str, Any],
    config: Config,
) -> bool:
    high_confidence_detector = high_confidence_standalone_detector(score_card)
    if int(repo.get("stargazers_count") or 0) < config.min_stars:
        return False
    if repo.get("fork") and not config.include_forks:
        return False
    if repo.get("archived") and not config.include_archived and not high_confidence_detector:
        return False
    if config.pushed_within_days is not None:
        pushed_at = parse_iso8601(repo.get("pushed_at"))
        if pushed_at is None:
            return False
        max_age = timedelta(days=config.pushed_within_days)
        if datetime.now(timezone.utc) - pushed_at > max_age:
            return False
    if (
        not config.include_research
        and score_card["repo_kind"] in RESEARCH_KINDS
        and score_card["repo_role"] != "classifier"
    ):
        return False
    if score_card["score"] < config.min_score:
        return False
    if not score_card.get("english_parseable", False):
        return False
    if not config.include_support_tools and score_card["base_detector_status"] != "base":
        return False
    if not score_card["core_hits"]:
        return False
    if (
        score_card["base_detector_status"] != "base"
        and not score_card["meaningful_core_hits"]
    ):
        return False
    if adjacent_method_only(score_card) and not high_confidence_detector:
        return False
    if method_unclear_only(score_card) and not high_confidence_detector:
        return False
    if rasp_like(score_card) and not passes_rasp_review(score_card):
        return False
    if integrity_method_only(score_card) and not passes_integrity_review(score_card):
        return False
    if score_card["primary_family"] == "UNKNOWN" and score_card["repo_role"] == "full":
        return False
    if (
        score_card["repo_kind"] == "tooling"
        and not score_card["meaningful_core_hits"]
        and not score_card["product_relation_descriptors"]
    ):
        return False
    return True


def csv_row(
    repo: dict[str, Any],
    score_card: dict[str, Any],
    matched_queries: set[str],
) -> dict[str, Any]:
    row = {
        "_score": score_card["score"],
        "full_name": repo.get("full_name") or "",
        "type": score_card["type_description"],
        "html_url": repo.get("html_url") or "",
        "topics": ";".join(repo.get("topics") or []),
        "stars": int(repo.get("stargazers_count") or 0),
        "forks": int(repo.get("forks_count") or 0),
        "open_issues": int(repo.get("open_issues_count") or 0),
        "language": repo.get("language") or "",
        "archived": bool(repo.get("archived")),
        "lifecycle_status": lifecycle_status(repo),
        "primary_family": score_card["primary_family"],
        "role": score_card["role"],
        "role_label": score_card["role_label"],
        "role_confidence": score_card["role_confidence"],
        "role_evidence": score_card["role_evidence"],
        "updated_at": repo.get("updated_at") or "",
        "pushed_at": repo.get("pushed_at") or "",
        "last_commit_sha": repo.get("last_commit_sha") or "",
        "_base_identity": score_card["base_identity"],
        "_base_detector_status": score_card["base_detector_status"],
        "_base_detector_reason": score_card["base_detector_reason"],
        "_base_detector_score": score_card["base_detector_score"],
        "_main_repository": score_card["main_repository"],
        "_main_repository_score": score_card["main_repository_score"],
        "_main_repository_reason": score_card["main_repository_reason"],
    }
    return row


def seeds_from_config(config: Config) -> list[str]:
    seeds = [] if config.replace_default_seeds else list(DEFAULT_SEEDS)
    for seed in config.extra_seeds:
        cleaned = seed.strip()
        if cleaned and cleaned not in seeds:
            seeds.append(cleaned)
    return seeds


def collect_candidates(
    config: Config,
    headers: dict[str, str],
    search_pause_seconds: float,
) -> dict[str, SearchCandidate]:
    candidates: dict[str, SearchCandidate] = {}
    seeds = seeds_from_config(config)
    print(f"Using {len(seeds)} search seeds.", file=sys.stderr)

    for seed in seeds:
        query = build_search_query(seed, config)
        print(f"Searching GitHub for seed: {seed}", file=sys.stderr)
        for page in range(1, config.max_pages_per_query + 1):
            payload = github_get_json(
                SEARCH_REPOSITORIES_URL,
                headers=headers,
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": config.per_page,
                    "page": page,
                },
                pause_seconds=search_pause_seconds,
            )
            items = payload.get("items") or []
            if not items:
                break
            for item in items:
                full_name = item.get("full_name")
                if not full_name:
                    continue
                if full_name not in candidates:
                    candidates[full_name] = SearchCandidate(item=item, matched_queries={seed})
                else:
                    candidates[full_name].matched_queries.add(seed)
                    if (item.get("stargazers_count") or 0) > (
                        candidates[full_name].item.get("stargazers_count") or 0
                    ):
                        candidates[full_name].item = item
            if len(items) < config.per_page:
                break

    return candidates


def row_quality_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        PUBLIC_ROLE_QUALITY.get(str(row.get("role") or ""), 0),
        int(row.get("_main_repository") is True or row.get("_main_repository") == "True"),
        int(row.get("_main_repository_score") or 0),
        int(row.get("_base_detector_status") == "base"),
        -repo_nonmain_variant_penalty(row),
        int(row.get("_base_detector_score") or 0),
        int(row.get("stars") or 0),
        int(row.get("_score") or 0),
    )


def row_owner_and_repo_name(row: dict[str, Any]) -> tuple[str, str]:
    full_name = (row.get("full_name") or "").lower()
    if "/" in full_name:
        return tuple(full_name.split("/", 1))  # type: ignore[return-value]
    return "", full_name


def row_deduplication_identity(row: dict[str, Any]) -> str:
    owner, repo_name = row_owner_and_repo_name(row)
    base_identity = row.get("_base_identity") or normalized_tool_identity(repo_name)
    name_identity = normalized_tool_identity(repo_name)
    family_identity = repo_family_identity(repo_name)

    if owner and family_identity and (
        family_identity != name_identity
        or base_identity == name_identity
        or base_identity == family_identity
    ):
        return f"{owner}/{family_identity}"
    return str(base_identity)


def describe_deduplication_group(identity: str, chosen: dict[str, Any], group: list[dict[str, Any]]) -> None:
    dropped = [row.get("full_name") or "" for row in group if row is not chosen]
    if not dropped:
        return
    print(
        "Deduplicated same-tool repository group "
        f"{identity}: kept {chosen.get('full_name')} "
        f"(quality={row_quality_key(chosen)}), dropped {', '.join(dropped)}",
        file=sys.stderr,
    )


def deduplicate_base_rows(rows: list[dict[str, Any]], log_groups: bool = True) -> list[dict[str, Any]]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_identity[row_deduplication_identity(row)].append(row)

    deduplicated: list[dict[str, Any]] = []
    for identity, group in by_identity.items():
        chosen = max(group, key=row_quality_key)
        if log_groups:
            describe_deduplication_group(identity, chosen, group)
        deduplicated.append(chosen)
    return deduplicated


def finalized_rows(rows: list[dict[str, Any]], max_repos: int, log_deduplication: bool = True) -> list[dict[str, Any]]:
    finalized = [dict(row) for row in rows]
    finalized = deduplicate_base_rows(finalized, log_groups=log_deduplication)
    finalized.sort(key=lambda row: (int(row["stars"]), int(row.get("_score", 0))), reverse=True)
    finalized = finalized[:max_repos]
    for rank, row in enumerate(finalized, start=1):
        row["rank"] = rank
    return finalized


def default_checkpoint_path(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    return f"{root}.partial{ext or '.csv'}"


def default_evaluated_cache_path(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    return f"{root}.evaluated{ext or '.csv'}"


def write_checkpoint(rows: list[dict[str, Any]], config: Config, evaluated: int, total: int) -> None:
    checkpoint_path = config.checkpoint_out or default_checkpoint_path(config.out)
    checkpoint_rows = finalized_rows(rows, config.max_repos, log_deduplication=False)
    try:
        write_csv(checkpoint_rows, checkpoint_path)
    except OSError as exc:
        print(f"Checkpoint write skipped for {checkpoint_path}: {exc}", file=sys.stderr)
        return
    print(
        f"Checkpoint saved after {evaluated}/{total} evaluated repos: "
        f"{len(checkpoint_rows)} rows -> {checkpoint_path}",
        file=sys.stderr,
    )


def repo_cache_key(full_name: str) -> str:
    return full_name.strip().lower()


def cache_row_repo(row: dict[str, str]) -> dict[str, Any]:
    repo = json_load(row.get("repo_json", ""), {})
    if repo:
        if row.get("last_commit_sha") and not repo.get("last_commit_sha"):
            repo["last_commit_sha"] = row.get("last_commit_sha")
        return repo
    topics = json_load(row.get("topics", ""), [])
    if isinstance(topics, str):
        topics = [topic for topic in topics.split(";") if topic]
    return {
        "full_name": row.get("full_name") or "",
        "name": (row.get("full_name") or "").split("/")[-1],
        "html_url": row.get("html_url") or "",
        "description": row.get("description") or "",
        "topics": topics if isinstance(topics, list) else [],
        "stargazers_count": int(row.get("stars") or 0),
        "forks_count": int(row.get("forks") or 0),
        "open_issues_count": int(row.get("open_issues") or 0),
        "language": row.get("language") or "",
        "archived": str(row.get("archived") or "").lower() == "true",
        "updated_at": row.get("updated_at") or "",
        "pushed_at": row.get("pushed_at") or "",
        "last_commit_sha": row.get("last_commit_sha") or "",
    }


def load_evaluated_cache(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Evaluated cache not found, starting without it: {path}", file=sys.stderr)
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            full_name = row.get("full_name") or ""
            if not full_name:
                continue
            repo = cache_row_repo(row)
            cache[repo_cache_key(full_name)] = {
                "row": row,
                "repo": repo,
                "readme_text": row.get("readme_text") or "",
                "docs_text": row.get("docs_text") or "",
                "docs_urls": json_load(row.get("docs_urls", ""), []),
                "matched_queries": set(json_load(row.get("matched_queries", ""), [])),
                "error": row.get("error") or "",
            }
    print(f"Loaded {len(cache)} evaluated repositories from cache: {path}", file=sys.stderr)
    return cache


def write_evaluated_cache(cache_rows: dict[str, dict[str, Any]], out_path: str) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    rows = sorted(
        cache_rows.values(),
        key=lambda row: (
            str(row.get("full_name") or "").lower(),
            str(row.get("evaluated_at") or ""),
        ),
    )
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATED_CACHE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in EVALUATED_CACHE_COLUMNS})


def write_evaluated_cache_checkpoint(
    cache_rows: dict[str, dict[str, Any]],
    config: Config,
    evaluated: int,
    total: int,
) -> None:
    if not config.evaluated_cache_out:
        return
    try:
        write_evaluated_cache(cache_rows, config.evaluated_cache_out)
    except OSError as exc:
        print(f"Evaluated cache write skipped for {config.evaluated_cache_out}: {exc}", file=sys.stderr)
        return
    print(
        f"Evaluated cache saved after {evaluated}/{total}: "
        f"{len(cache_rows)} repos -> {config.evaluated_cache_out}",
        file=sys.stderr,
    )


def evaluation_cache_row(
    *,
    full_name: str,
    matched_queries: set[str],
    repo: dict[str, Any],
    readme_text: str,
    docs_text: str,
    docs_urls: list[str],
    final_score: dict[str, Any] | None,
    accepted: bool,
    error: str = "",
) -> dict[str, Any]:
    topics = repo.get("topics") or []
    return {
        "cache_version": EVALUATED_CACHE_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "full_name": full_name,
        "accepted": "." if accepted else "",
        "error": error,
        "matched_queries": json_dump(sorted(matched_queries)),
        "repo_json": json_dump(repo),
        "readme_text": readme_text,
        "docs_text": docs_text,
        "docs_urls": json_dump(docs_urls),
        "score_card_json": json_dump(final_score or {}),
        "score": (final_score or {}).get("score", ""),
        "primary_family": (final_score or {}).get("primary_family", ""),
        "type_description": (final_score or {}).get("type_description", ""),
        "repo_role": (final_score or {}).get("repo_role", ""),
        "repo_kind": (final_score or {}).get("repo_kind", ""),
        "base_identity": (final_score or {}).get("base_identity", ""),
        "base_detector_status": (final_score or {}).get("base_detector_status", ""),
        "base_detector_reason": (final_score or {}).get("base_detector_reason", ""),
        "base_detector_score": (final_score or {}).get("base_detector_score", ""),
        "canonical_product_repo": (final_score or {}).get("canonical_product_repo", ""),
        "html_url": repo.get("html_url") or "",
        "description": repo.get("description") or "",
        "topics": json_dump(topics),
        "stars": int(repo.get("stargazers_count") or 0),
        "forks": int(repo.get("forks_count") or 0),
        "open_issues": int(repo.get("open_issues_count") or 0),
        "language": repo.get("language") or "",
        "archived": bool(repo.get("archived")),
        "updated_at": repo.get("updated_at") or "",
        "pushed_at": repo.get("pushed_at") or "",
        "last_commit_sha": repo.get("last_commit_sha") or "",
    }


def candidates_from_evaluated_cache(cache: dict[str, dict[str, Any]]) -> dict[str, SearchCandidate]:
    candidates: dict[str, SearchCandidate] = {}
    for entry in cache.values():
        if entry.get("error"):
            continue
        repo = entry.get("repo") or {}
        full_name = repo.get("full_name") or entry.get("row", {}).get("full_name")
        if not full_name:
            continue
        matched_queries = set(entry.get("matched_queries") or [])
        candidates[full_name] = SearchCandidate(item=repo, matched_queries=matched_queries)
    return candidates


def crawl(config: Config) -> list[dict[str, Any]]:
    headers = github_headers(config.token)
    authenticated = has_github_token(headers)
    search_pause_seconds = effective_search_pause(config, authenticated)
    core_pause_seconds = effective_core_pause(config, authenticated)
    product_signals = load_product_signals(config.product_signals_path)
    evaluated_cache = load_evaluated_cache(config.evaluated_cache_in)
    evaluated_cache_rows: dict[str, dict[str, Any]] = {
        key: dict(entry["row"])
        for key, entry in evaluated_cache.items()
        if entry.get("row")
    }

    if not authenticated:
        print(
            "No GitHub token found. Running in anonymous-friendly mode with slower pacing "
            f"(search pause {search_pause_seconds:.1f}s, detail pause {core_pause_seconds:.1f}s).",
            file=sys.stderr,
        )
    else:
        print(
            "GitHub token detected. Using faster pacing "
            f"(search pause {search_pause_seconds:.1f}s, detail pause {core_pause_seconds:.1f}s).",
            file=sys.stderr,
        )

    if config.cache_only:
        if not evaluated_cache:
            raise RuntimeError("--cache-only requires a non-empty --evaluated-cache-in CSV.")
        candidates = candidates_from_evaluated_cache(evaluated_cache)
        print(f"Using {len(candidates)} repositories from evaluated cache only.", file=sys.stderr)
    else:
        candidates = collect_candidates(config, headers, search_pause_seconds)

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            int(candidate.item.get("stargazers_count") or 0),
            len(candidate.matched_queries),
            str(candidate.item.get("full_name") or "").lower(),
        ),
        reverse=True,
    )
    candidate_pool_size = len(ranked_candidates)
    print(
        f"Collected {len(candidates)} unique repositories. "
        f"Enriching all {candidate_pool_size} before classification.",
        file=sys.stderr,
    )
    print(
        "Batch mode: final relevance decisions and compact CSV rows are delayed "
        "until all candidates have been enriched. Interrupting before the "
        "classification phase will not produce a final accepted-row CSV.",
        file=sys.stderr,
    )

    enriched_records: list[dict[str, Any]] = []

    for index, candidate in enumerate(ranked_candidates, start=1):
        full_name = candidate.item["full_name"]
        print(
            f"[{index}/{candidate_pool_size}] Evaluating {full_name}",
            file=sys.stderr,
        )

        matched_queries = set(candidate.matched_queries)
        try:
            cached_entry = evaluated_cache.get(repo_cache_key(full_name))
            if cached_entry and not cached_entry.get("error"):
                repo = cached_entry["repo"]
                if config.refresh_commit_sha and not repo.get("last_commit_sha"):
                    repo = ensure_last_commit_sha(repo, headers=headers, pause_seconds=core_pause_seconds)
                matched_queries.update(cached_entry.get("matched_queries") or set())
                cache_readme_text = cached_entry.get("readme_text") or ""
                cache_docs_text = cached_entry.get("docs_text") or ""
                readme_text = "" if config.readme_mode == "never" else cache_readme_text
                docs_text = "" if config.docs_mode == "never" else cache_docs_text
                docs_urls = list(cached_entry.get("docs_urls") or [])
                print(f"  Using evaluated cache for {full_name}", file=sys.stderr)
            else:
                repo = fetch_repository(full_name, headers=headers, pause_seconds=core_pause_seconds)
                repo = ensure_last_commit_sha(repo, headers=headers, pause_seconds=core_pause_seconds)
                cache_readme_text = ""
                cache_docs_text = ""
                provisional_score = score_repository(
                    repo,
                    matched_queries,
                    readme_text="",
                    product_signals=product_signals,
                )
                readme_text = ""
                if should_fetch_readme(repo, provisional_score, config, authenticated):
                    readme_text = fetch_readme_text(full_name, headers=headers, pause_seconds=core_pause_seconds)
                cache_readme_text = readme_text
                readme_score = score_repository(
                    repo,
                    matched_queries,
                    readme_text=readme_text,
                    product_signals=product_signals,
                )
                docs_text = ""
                docs_urls = []
                if config.docs_mode != "never":
                    docs_urls = discover_docs_urls(repo, readme_text, config.max_docs_pages)
                    if docs_urls and (
                        config.docs_mode == "always"
                        or readme_score["primary_family"] == "UNKNOWN"
                        or readme_score["detection_types"] == ["UNKNOWN"]
                    ):
                        fetched_docs = []
                        for docs_url in docs_urls:
                            text = fetch_docs_text(docs_url, pause_seconds=min(core_pause_seconds, 2.0))
                            if text:
                                fetched_docs.append(text)
                        docs_text = "\n".join(fetched_docs)
                        cache_docs_text = docs_text
            final_score = score_repository(
                repo,
                matched_queries,
                readme_text=readme_text,
                docs_text=docs_text,
                product_signals=product_signals,
            )
            record = {
                "full_name": full_name,
                "matched_queries": matched_queries,
                "repo": repo,
                "readme_text": cache_readme_text,
                "docs_text": cache_docs_text,
                "docs_urls": docs_urls,
                "final_score": final_score,
                "error": "",
            }
            enriched_records.append(record)
            evaluated_cache_rows[repo_cache_key(full_name)] = evaluation_cache_row(
                full_name=full_name,
                matched_queries=matched_queries,
                repo=repo,
                readme_text=cache_readme_text,
                docs_text=cache_docs_text,
                docs_urls=docs_urls,
                final_score=final_score,
                accepted=False,
            )
        except RuntimeError as exc:
            print(f"Skipping {full_name}: {exc}", file=sys.stderr)
            enriched_records.append(
                {
                    "full_name": full_name,
                    "matched_queries": matched_queries,
                    "repo": candidate.item,
                    "readme_text": "",
                    "docs_text": "",
                    "docs_urls": [],
                    "final_score": None,
                    "error": str(exc),
                }
            )
            evaluated_cache_rows[repo_cache_key(full_name)] = evaluation_cache_row(
                full_name=full_name,
                matched_queries=matched_queries,
                repo=candidate.item,
                readme_text="",
                docs_text="",
                docs_urls=[],
                final_score=None,
                accepted=False,
                error=str(exc),
            )
        except Exception as exc:
            print(f"Skipping {full_name}: unexpected error: {exc}", file=sys.stderr)
            error = f"unexpected error: {exc}"
            enriched_records.append(
                {
                    "full_name": full_name,
                    "matched_queries": matched_queries,
                    "repo": candidate.item,
                    "readme_text": "",
                    "docs_text": "",
                    "docs_urls": [],
                    "final_score": None,
                    "error": error,
                }
            )
            evaluated_cache_rows[repo_cache_key(full_name)] = evaluation_cache_row(
                full_name=full_name,
                matched_queries=matched_queries,
                repo=candidate.item,
                readme_text="",
                docs_text="",
                docs_urls=[],
                final_score=None,
                accepted=False,
                error=error,
            )
        finally:
            if config.checkpoint_every > 0 and index % config.checkpoint_every == 0:
                write_evaluated_cache_checkpoint(evaluated_cache_rows, config, index, candidate_pool_size)

    print(
        f"Classifying {len(enriched_records)} enriched repositories.",
        file=sys.stderr,
    )
    rows: list[dict[str, Any]] = []
    for record in enriched_records:
        final_score = record.get("final_score")
        accepted = False
        if final_score:
            accepted = accepted_repository(record["repo"], final_score, config)
            if accepted:
                rows.append(csv_row(record["repo"], final_score, record["matched_queries"]))
        evaluated_cache_rows[repo_cache_key(record["full_name"])] = evaluation_cache_row(
            full_name=record["full_name"],
            matched_queries=record["matched_queries"],
            repo=record["repo"],
            readme_text=record["readme_text"],
            docs_text=record["docs_text"],
            docs_urls=record["docs_urls"],
            final_score=final_score,
            accepted=accepted,
            error=record.get("error", ""),
        )
    print(
        f"Batch classification accepted {len(rows)} repositories before deduplication/ranking.",
        file=sys.stderr,
    )

    if config.checkpoint_every > 0:
        write_checkpoint(rows, config, candidate_pool_size, candidate_pool_size)
    write_evaluated_cache_checkpoint(evaluated_cache_rows, config, candidate_pool_size, candidate_pool_size)
    return finalized_rows(rows, config.max_repos)


def write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    columns = [
        "rank",
        "full_name",
        "type",
        "primary_family",
        "role",
        "role_label",
        "role_confidence",
        "role_evidence",
        "html_url",
        "topics",
        "stars",
        "forks",
        "open_issues",
        "language",
        "archived",
        "lifecycle_status",
        "updated_at",
        "pushed_at",
        "last_commit_sha",
    ]
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def print_summary(rows: list[dict[str, Any]], out_path: str) -> None:
    print(f"Saved {len(rows)} repositories to {out_path}")
    if not rows:
        return
    print()
    print("Top matches:")
    for row in rows[:10]:
        role = row.get("role_label") or row.get("role") or "unknown"
        print(
            f"{row['rank']:>2}. {row['full_name']} | stars={row['stars']} | "
            f"category={row['primary_family']} | type={row['type']} | role={role}"
        )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Crawl GitHub for public IDS, IPS, and related intrusion-detection repositories.",
    )
    parser.add_argument("--max-repos", type=int, default=200, help="Maximum number of CSV rows to keep.")
    parser.add_argument("--min-stars", type=int, default=100, help="Minimum GitHub stars for candidate repositories.")
    parser.add_argument("--min-score", type=int, default=10, help="Minimum heuristic relevance score.")
    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=10,
        help="Maximum GitHub search result pages to inspect per search seed.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="GitHub search page size. GitHub allows up to 100.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=5,
        help="Deprecated compatibility option. All collected candidates are enriched before classification.",
    )
    parser.add_argument(
        "--readme-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="Whether README text should be used for scoring.",
    )
    parser.add_argument(
        "--docs-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="Whether discovered documentation/homepage text should be used for classification.",
    )
    parser.add_argument(
        "--max-docs-pages",
        type=int,
        default=2,
        help="Maximum discovered documentation pages to fetch per enriched repository.",
    )
    parser.add_argument(
        "--pushed-within-days",
        type=int,
        default=None,
        help="If set, only keep repositories pushed within the last N days.",
    )
    parser.add_argument("--include-forks", action="store_true", help="Keep fork repositories.")
    parser.add_argument("--include-archived", action="store_true", help="Keep archived repositories.")
    parser.add_argument(
        "--include-research",
        action="store_true",
        help="Keep datasets, paper repos, tutorials, and resource lists.",
    )
    parser.add_argument(
        "--include-support-tools",
        action="store_true",
        help="Keep derivative/support repos such as rulesets, dashboards, deployment repos, and suites.",
    )
    parser.add_argument(
        "--extra-seed",
        action="append",
        default=[],
        help="Additional GitHub search seed. Can be provided multiple times.",
    )
    parser.add_argument(
        "--replace-default-seeds",
        action="store_true",
        help="Use only --extra-seed values instead of the built-in seed list.",
    )
    parser.add_argument(
        "--anonymous-friendly",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use very slow public-API pacing when no GitHub token is available.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help="Legacy setting: use the same pause for all GitHub API requests.",
    )
    parser.add_argument(
        "--search-pause-seconds",
        type=float,
        default=None,
        help="Pause between GitHub search API requests.",
    )
    parser.add_argument(
        "--core-pause-seconds",
        type=float,
        default=None,
        help="Pause between repository detail and README requests.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(script_dir(), "ids_repos.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write a ranked partial CSV after every N evaluated repositories. Use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default=None,
        help="Partial CSV path. Defaults to the output path with .partial before the extension.",
    )
    parser.add_argument(
        "--evaluated-cache-in",
        default=None,
        help="Wide evaluated-repository cache CSV to reuse instead of refetching matching repositories.",
    )
    parser.add_argument(
        "--evaluated-cache-out",
        default=None,
        help="Wide evaluated-repository cache CSV to write. Defaults to the output path with .evaluated before the extension.",
    )
    parser.add_argument(
        "--no-evaluated-cache",
        action="store_true",
        help="Disable writing the wide evaluated-repository cache CSV.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Skip GitHub search and reprocess only repositories from --evaluated-cache-in.",
    )
    parser.add_argument(
        "--refresh-commit-sha",
        action="store_true",
        help="Fetch missing last_commit_sha values even when a repository is reused from the evaluated cache.",
    )
    parser.add_argument(
        "--product-signals",
        default=DEFAULT_PRODUCT_SIGNALS_PATH,
        help="Deprecated compatibility option. Product-signal priors are not used for crawler decisions.",
    )
    parser.add_argument(
        "--use-product-signals",
        action="store_true",
        default=False,
        help="Deprecated compatibility option. Curated product priors are ignored.",
    )
    parser.add_argument(
        "--no-product-signals",
        action="store_true",
        default=False,
        help="Disable product-signal priors and rely only on generic textual evidence.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub token. If omitted, GITHUB_TOKEN or GH_TOKEN is used.",
    )

    args = parser.parse_args()

    if args.replace_default_seeds and not args.extra_seed:
        parser.error("--replace-default-seeds requires at least one --extra-seed.")
    if args.per_page < 1 or args.per_page > 100:
        parser.error("--per-page must be between 1 and 100.")
    if args.max_repos < 1:
        parser.error("--max-repos must be at least 1.")
    if args.min_stars < 0:
        parser.error("--min-stars cannot be negative.")
    if args.min_score < 1:
        parser.error("--min-score must be at least 1.")
    if args.max_docs_pages < 0:
        parser.error("--max-docs-pages cannot be negative.")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative.")
    if args.cache_only and not args.evaluated_cache_in:
        parser.error("--cache-only requires --evaluated-cache-in.")

    evaluated_cache_out = None
    if not args.no_evaluated_cache:
        evaluated_cache_out = os.path.abspath(args.evaluated_cache_out or default_evaluated_cache_path(args.out))

    return Config(
        max_repos=args.max_repos,
        min_stars=args.min_stars,
        min_score=args.min_score,
        max_pages_per_query=args.max_pages_per_query,
        per_page=args.per_page,
        candidate_multiplier=args.candidate_multiplier,
        include_forks=args.include_forks,
        include_archived=args.include_archived,
        include_research=args.include_research,
        readme_mode=args.readme_mode,
        docs_mode=args.docs_mode,
        max_docs_pages=args.max_docs_pages,
        include_support_tools=args.include_support_tools,
        pushed_within_days=args.pushed_within_days,
        out=os.path.abspath(args.out),
        token=args.token,
        extra_seeds=list(args.extra_seed),
        replace_default_seeds=args.replace_default_seeds,
        pause_seconds=args.pause_seconds,
        search_pause_seconds=args.search_pause_seconds,
        core_pause_seconds=args.core_pause_seconds,
        anonymous_friendly=args.anonymous_friendly,
        checkpoint_every=args.checkpoint_every,
        checkpoint_out=os.path.abspath(args.checkpoint_out) if args.checkpoint_out else None,
        evaluated_cache_in=os.path.abspath(args.evaluated_cache_in) if args.evaluated_cache_in else None,
        evaluated_cache_out=evaluated_cache_out,
        cache_only=args.cache_only,
        product_signals_path=None,
        refresh_commit_sha=args.refresh_commit_sha,
    )


def main() -> int:
    config = parse_args()
    try:
        rows = crawl(config)
    except RuntimeError as exc:
        print(f"Crawl failed: {exc}", file=sys.stderr)
        return 1
    write_csv(rows, config.out)
    print_summary(rows, config.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
