#!/usr/bin/env python3
"""OpenAlex-only IDS dataset discovery.

This crawler is survey-first and dataset-entity-first:

1. Use trusted survey/base papers supplied by the user, or discover trusted
   IDS-dataset survey papers from OpenAlex when none are supplied.
2. Extract dataset names from those survey papers and optional open-access PDF
   text, especially table-like contexts.
3. Resolve each dataset entity to a likely canonical/base publication with
   strict downstream-paper rejection.
4. Categorize datasets for later plots and write accepted, candidate, and
   rejected CSVs.

No Semantic Scholar API is used.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 4
MAX_RETRY_AFTER_SECONDS = 90

DEFAULT_OUT = Path("ids_datasets_openalex.csv")
CHECKPOINT_VERSION = 1
UNKNOWN = "Unknown"


class OpenAlexRateLimitError(RuntimeError):
    """Raised when OpenAlex asks us to wait longer than this run should block."""

    def __init__(self, wait_seconds: int, source: str = "OpenAlex") -> None:
        super().__init__(f"{source} rate limited for {wait_seconds}s")
        self.wait_seconds = wait_seconds
        self.source = source

DEFAULT_SURVEY_QUERIES = (
    "intrusion detection datasets survey",
    "network intrusion detection datasets review",
    "NIDS datasets benchmark survey",
    "IoT intrusion detection datasets survey",
    "industrial control system SCADA intrusion detection datasets survey",
    "cybersecurity datasets intrusion detection survey",
    "public datasets network intrusion detection survey",
    "survey intrusion detection systems techniques datasets challenges",
    "A Review of the Advancement in Intrusion Detection Datasets",
    "Advanced IDS: a comparative study of datasets and machine learning algorithms for network flow-based intrusion detection systems",
    "From Bytes to Insights: A Systematic Literature Review on Unraveling IDS Datasets for Enhanced Cybersecurity Understanding",
)

DEFAULT_DATASET_PUBLICATION_QUERIES = (
    "new intrusion detection dataset",
    "network intrusion detection dataset benchmark",
    "cybersecurity dataset intrusion detection",
    "network traffic dataset attacks",
    "network flow dataset intrusion detection",
    "botnet dataset intrusion detection",
    "ddos dataset network traffic",
    "iot intrusion detection dataset",
    "iiot intrusion detection dataset",
    "industrial control system intrusion detection dataset",
    "scada attack dataset",
    "dns exfiltration dataset",
    "malware traffic dataset intrusion detection",
    "wireless intrusion detection dataset",
    "host based intrusion detection dataset",
    "system call intrusion detection dataset",
    "can bus intrusion detection dataset",
    "automotive intrusion detection dataset",
    "malicious traffic injection benign background traffic",
    "attack injection network traffic traces",
    "anomaly injection network traffic",
    "network traffic generation intrusion detection dataset",
    "traffic generator intrusion detection dataset",
    "synthetic attack injection intrusion detection",
    "flow level anomaly modeling engine network traffic",
    "intrusion detection dataset toolkit synthetic attacks",
)

DEFAULT_BROAD_DATASET_QUERIES = (
    "intrusion detection evaluation data set",
    "network intrusion detection benchmark dataset",
    "IDS benchmark data",
    "traffic traces intrusion detection",
    "botnet traffic dataset",
    "malware traffic capture dataset",
    "network security laboratory dataset",
    "network traffic traces attacks dataset",
    "public network traffic traces dataset",
    "internet measurement traffic archive security dataset",
    "wide area network traffic traces dataset",
    "public cyber security data repository traffic traces",
    "network security data repository traces",
    "traffic trace data repository internet security",
    "research data repository intrusion detection traffic traces",
    "web attack traffic dataset",
    "http flooding attack dataset",
    "web server intrusion detection dataset",
    "application layer dos traffic dataset",
    "web intrusion detection traffic traces",
    "intrusion response system call dataset",
    "host intrusion response command dataset",
    "intrusion detection system call traces dataset",
    "incident response security command dataset",
    "anomaly detection network traffic benchmark",
    "cybersecurity benchmark data network attacks",
)

CANDIDATE_QUERY_TEMPLATES_COMPACT = (
    "{alias}",
    "{alias} dataset",
    "{alias} benchmark",
    "{alias} original paper",
    "{alias} intrusion detection dataset",
    "{alias} network traffic",
)

CANDIDATE_QUERY_TEMPLATES_FULL = (
    "{alias} dataset",
    "{alias} benchmark",
    "{alias} intrusion detection dataset",
    "{alias} traffic generator",
    "{alias} traffic generation",
    "{alias} generator framework",
    "{alias} attack injection",
    "{alias} anomaly modeling engine",
    "{alias} flow level anomaly modeling engine",
    "{alias} dataset creation toolkit",
    "{alias} original dataset paper",
    "{alias} original paper",
    "{alias} dataset generation",
    "{alias} generated dataset",
    "{alias} dataset collection",
    "{alias} intrusion traffic characterization",
    "{alias} reliable intrusion detection benchmark dataset",
    "{alias} Canadian Institute for Cybersecurity",
    "{alias} network traffic",
    "{alias}",
)

SURVEY_WORDS = (
    "survey",
    "review",
    "systematic literature",
    "taxonomy",
    "datasets and challenges",
    "data sets and challenges",
)

DATASET_WORDS = (
    "dataset",
    "data set",
    "datasets",
    "data sets",
    "benchmark",
    "benchmarks",
    "corpus",
    "testbed",
    "traffic traces",
    "trace dataset",
    "traffic archive",
    "packet traces",
    "flow data",
    "log data",
    "traffic data",
    "internet traffic",
    "network traces",
    "traffic archive",
)

GENERATOR_WORDS = (
    "traffic generator",
    "traffic generation",
    "network traffic generation",
    "trace generation",
    "traffic synthesis",
    "synthetic traffic",
    "synthetic attacks",
    "synthetic attack",
    "attack injection",
    "anomaly injection",
    "inject attacks",
    "inject attack",
    "injecting attacks",
    "injecting synthetic attacks",
    "inject malicious traffic",
    "injecting malicious traffic",
    "malicious traffic injection",
    "benign background traffic",
    "background traffic",
    "flow-level anomaly modeling",
    "flow level anomaly modeling",
    "anomaly modeling engine",
    "intrusion detection dataset toolkit",
    "dataset toolkit",
    "toolkit",
    "framework",
    "engine",
)

STRONG_GENERATOR_WORDS = (
    "traffic generator",
    "traffic generation",
    "network traffic generation",
    "trace generation",
    "traffic synthesis",
    "synthetic traffic",
    "synthetic attacks",
    "synthetic attack",
    "attack injection",
    "anomaly injection",
    "inject attacks",
    "inject attack",
    "injecting attacks",
    "injecting synthetic attacks",
    "inject malicious traffic",
    "injecting malicious traffic",
    "malicious traffic injection",
    "benign background traffic",
    "flow-level anomaly modeling",
    "flow level anomaly modeling",
    "anomaly modeling engine",
    "intrusion detection dataset toolkit",
    "dataset creation toolkit",
)

GENERIC_FRAMEWORK_WORDS = (
    "framework",
    "toolkit",
    "engine",
)

SECURITY_WORDS = (
    "intrusion",
    "attack",
    "cyber",
    "cybersecurity",
    "network security",
    "malware",
    "botnet",
    "ddos",
    "dos",
    "exfiltration",
    "tunnel",
    "anomaly",
    "forensic",
    "scada",
    "ics",
    "industrial control",
    "iot",
    "iiot",
    "mqtt",
    "dns",
    "can bus",
    "vehicular",
    "wireless",
    "wifi",
    "network traffic",
    "internet traffic",
    "traffic archive",
    "traffic trace",
    "traffic traces",
    "packet trace",
    "packet traces",
    "flow trace",
    "flow traces",
    "backbone traffic",
    "darknet",
    "system call",
    "event log",
    "cyber range",
)

IDS_SURVEY_SCOPE_WORDS = (
    "intrusion detection",
    "intrusion detection system",
    "intrusion detection systems",
    "network intrusion",
    "host intrusion",
    "ids dataset",
    "nids dataset",
    "cybersecurity dataset",
    "cyber security dataset",
    "network security dataset",
    "attack detection",
    "anomaly detection",
    "malicious traffic",
    "network attacks",
    "cyber attacks",
    "cyberattack",
    "cyber-physical",
    "industrial control system",
    "industrial control systems",
    "scada",
    "botnet detection",
    "ddos detection",
)

BROAD_SURVEY_TITLE_WORDS = (
    "civil applications",
    "key research challenges",
    "routing",
    "blockchain",
    "cloud computing",
    "privacy",
    "smart city",
    "wireless sensor networks",
    "edge computing",
    "internet of things: a survey",
)

INTRODUCTION_WORDS = (
    "introduce",
    "introduces",
    "introduced",
    "present",
    "presents",
    "presented",
    "propose",
    "proposes",
    "release",
    "released",
    "create",
    "created",
    "generate",
    "generated",
    "generation of",
    "a new",
    "a novel",
    "realistic",
    "real-world",
    "comprehensive",
    "annotated",
    "labeled",
    "labelled",
    "publicly available",
    "available",
    "collected",
    "collection",
    "capture",
    "captured",
    "captures",
)

CREATION_VERBS = (
    "introduce",
    "introduces",
    "introduced",
    "present",
    "presents",
    "presented",
    "release",
    "released",
    "create",
    "created",
    "generate",
    "generated",
    "collect",
    "collected",
    "capture",
    "captured",
    "construct",
    "constructed",
    "produce",
    "produced",
    "provide",
    "provides",
    "provided",
    "develop",
    "developed",
)

DOWNSTREAM_WORDS = (
    "using",
    "use of",
    "based on",
    "trained on",
    "trained using",
    "tested on",
    "evaluated on",
    "applied to",
    "classification",
    "classifier",
    "feature selection",
    "feature extraction",
    "deep learning",
    "machine learning",
    "neural network",
    "model",
    "algorithm",
    "approach",
    "method",
    "performance",
    "evaluation",
    "evaluate",
    "evaluating",
    "analysis",
    "analyzing",
    "assessment",
    "reduction",
    "optimization",
    "optimized",
    "resampling",
    "oversampling",
    "visualization",
    "comparison",
    "comparative",
    "prediction",
)

NON_CANONICAL_DATASET_TITLE_WORDS = (
    "case study",
    "troubleshooting",
    "research of",
    "detailed analysis",
    "feature analysis",
    "performance analysis",
    "benchmarking datasets",
    "multi-dataset analysis",
    "experimental insights",
    "comparative study",
    "alternatives",
    "labeling errors",
    "labelling errors",
    "quality assessment",
    "performance improvements",
    "revisiting",
    "benchmarking of",
    "benchmarking key features",
    "significant features",
    "statistical analysis",
    "evaluation of",
    "empirical evaluation",
    "visualization of",
    "subset feature",
    "feature elimination",
    "resampling imbalanced",
    "oversampling technique",
)

WEAK_NAME_TOKENS = {
    "dataset",
    "datasets",
    "data",
    "set",
    "sets",
    "benchmark",
    "benchmarks",
    "corpus",
    "corpora",
    "testbed",
    "testbeds",
    "trace",
    "traces",
    "traffic",
    "archive",
    "collection",
    "cup",
}

SYNONYM_SIMILARITY_THRESHOLD = 0.88
SYNONYM_NEAR_MISS_THRESHOLD = 0.72
MIN_YEAR_GUARD = 1980
MAX_REASONABLE_RELEASE_YEAR = 2035

DATASET_TABLE_HEADINGS = (
    "dataset",
    "data set",
    "year",
    "attacks",
    "attack",
    "normal",
    "records",
    "features",
    "traffic",
    "public",
    "availability",
    "environment",
    "label",
    "tool",
    "generator",
    "generation",
    "framework",
    "engine",
    "testbed",
)

LOW_PRECISION_NAMES = {
    # "impact",
    # "flame",
    "ton",
    "flow",
    "trace",
    "attack",
    "dataset",
    "data",
    "benchmark",
    "testbed",
}

COMPONENT_FOLLOWER_WORDS = (
    "telemetry",
    "command",
    "commands",
    "script",
    "scripts",
    "reproduction",
    "platform",
    "simulator",
    "simulation",
    "emulator",
    "emulation",
    "tool",
    "tools",
    "toolkit",
    "framework",
    "software",
    "operating system",
    "protocol",
    "sensor",
    "sensors",
    "device",
    "devices",
)

BAD_DATASET_NAMES = {
    "IDS",
    "IPS",
    "AIDS",
    "HADS",
    "IDPS",
    "NIDS",
    "HIDS",
    "IoT",
    "IIoT",
    "SCADA",
    "ICS",
    "ML",
    "AI",
    "AI/ML",
    "DL",
    "DNN",
    "CNN",
    "LSTM",
    "ANN",
    "SVM",
    "RF",
    "DIY",
    "TON",
    "CICIDS",
    "IoT/IIoT",
    "CAN",
    "The Dataset",
    "Dataset",
    "Benchmark",
    "Techniques",
    "This",
    "Windows",
    "Table",
    "Figure",
    "GitHub",
    "IEEE",
    "Elsevier",
    "AdaBoost",
    "OCSVM",
    "RIPPER",
    "MiTM",
    "MITM",
    "Multi-Layer",
    "Hyper-Parameter",
    "OpenFlow",
    "Wi-Fi",
    "Wifi",
    "Host-Based",
    "Flow-Based",
    "ML-Based",
    "Anomaly-Based",
    "SCADA-Based",
    "SDN-Based",
    "Inter-car",
}

MODEL_METHOD_NAME_TOKENS = {
    "1d",
    "2d",
    "3d",
    "ai",
    "ml",
    "dl",
    "cnn",
    "dcnn",
    "rnn",
    "gru",
    "bigru",
    "lstm",
    "bilstm",
    "ann",
    "dnn",
    "gan",
    "svm",
    "knn",
    "rf",
    "dt",
    "gbm",
    "gbdt",
    "xgboost",
    "adaboost",
    "bert",
    "transformer",
    "autoencoder",
    "autoencoders",
    "pca",
    "smote",
    "fed",
    "federated",
    "powered",
    "auc",
    "auroc",
    "roc",
    "apt",
}

GENERIC_NON_DATASET_NAME_WORDS = (
    "about",
    "based",
    "specific",
    "fold",
    "consisting",
    "approach",
    "model",
    "method",
    "algorithm",
    "classifier",
    "classification",
    "multi class",
    "multi-class",
    "learning",
    "adaboost",
    "ocsvm",
    "one class svm",
    "one-class svm",
    "ripper",
    "mitm",
    "man in the middle",
    "multi layer",
    "hyper parameter",
    "hyperparameter",
    "wi fi",
    "wifi",
    "openflow",
    "host based",
    "flow based",
    "ml based",
    "anomaly based",
    "scada based",
    "sdn based",
    "inter car",
    "powered",
    "smote",
    "tensorflow",
)

DOMAIN_RULES = (
    (
        "DNS",
        (
            "dns",
            "doh",
            "dns over https",
            "dns exfiltration",
            "dns tunnel",
            "dns tunneling",
        ),
    ),
    (
        "Automotive/CAN",
        (
            "can bus",
            "controller area network",
            "automotive",
            "vehicle",
            "vehicular",
            "vanet",
            "car hacking",
            "in-vehicle",
        ),
    ),
    (
        "ICS/SCADA",
        (
            "scada",
            "ics",
            "industrial control",
            "cyber physical",
            "cyber-physical",
            "power system",
            "water treatment",
            "swat",
            "wadi",
            "modbus",
            "plc",
        ),
    ),
    (
        "Host",
        (
            "system call",
            "host based",
            "host-based",
            "audit log",
            "event log",
            "windows",
            "linux",
        ),
    ),
    (
        "Wireless",
        (
            "wireless",
            "wifi",
            "wi fi",
            "802.11",
            "802 11",
            "wlan",
            "sensor network",
            "wsn",
            "uav",
            "drone",
        ),
    ),
    (
        "IIoT",
        (
            "iiot",
            "industrial iot",
            "industrial internet of things",
            "edge iiot",
        ),
    ),
    (
        "IoT",
        (
            "iot",
            "internet of things",
            "smart home",
            "smart city",
            "smart devices",
            "mqtt",
        ),
    ),
    (
        "Encrypted/VPN/Tor",
        (
            "tor",
            "nontor",
            "non tor",
            "vpn",
            "nonvpn",
            "non vpn",
            "encrypted traffic",
        ),
    ),
    (
        "Web/App",
        (
            "web application",
            "sql injection",
            "xss",
            "web attack",
        ),
    ),
    (
        "Malware/Botnet",
        (
            "malware dataset",
            "malware traffic",
            "botnet",
            "botnet dataset",
            "botnet traffic",
            "ransomware dataset",
            "ransomware traffic",
        ),
    ),
    (
        "General Network",
        (
            "network intrusion",
            "network traffic",
            "network flow",
            "traffic traces",
            "packet traces",
            "ddos",
            "dos attack",
        ),
    ),
)

DATA_FORM_RULES = (
    (
        "Host/System-call",
        (
            "system call",
            "system-call",
            "audit log",
            "event log",
            "host log",
        ),
    ),
    ("CAN", ("can bus", "controller area network", "automotive can", "in-vehicle")),
    ("Wireless", ("802.11", "802 11", "wifi", "wi fi", "wlan", "wireless")),
    ("Packet", ("pcap", "packet trace", "packet traces", "packet-level", "packet level")),
    ("Flow", ("flow", "flows", "netflow", "network flow", "flow-level", "flow level", "bidirectional flow")),
    ("Telemetry", ("telemetry", "sensor telemetry", "sensor data", "measurements")),
    ("Log", ("log data", "logs", "event log", "audit log")),
    ("Tabular features", ("feature", "features", "records", "csv", "tabular", "data frame", "41 features")),
    (
        "Network traffic",
        (
            "network traffic",
            "network data",
            "network data set",
            "network dataset",
            "traffic traces",
            "internet traffic",
            "traffic data",
            "benign traffic",
            "encrypted traffic",
            "vpn traffic",
            "tor traffic",
            "vpn",
            "nonvpn",
            "non vpn",
            "tor",
            "nontor",
            "non tor",
        ),
    ),
)

REAL_WORLD_PHRASES = (
    "real-world",
    "real world",
    "real network",
    "real traffic",
    "production",
    "operational",
    "backbone",
    "campus",
    "live network",
    "collected from",
    "captured from",
    "traffic traces from",
    "internet traffic",
)

SYNTHETIC_LAB_PHRASES = (
    "synthetic",
    "generated",
    "generate",
    "simulation",
    "simulated",
    "emulated",
    "emulation",
    "testbed",
    "test bed",
    "lab",
    "laboratory",
    "virtual",
    "injected",
    "injecting",
    "attack injection",
    "synthetic attack",
    "synthetic attacks",
)

LABELED_PHRASES = (
    "labeled",
    "labelled",
    "annotated",
    "ground truth",
    "normal and attack",
    "normal traffic",
    "benign traffic",
    "malicious traffic",
    "benign and attack",
    "benign traffic and",
    "attack classes",
    "attack labels",
    "class labels",
    "classified as",
)


@dataclass
class Work:
    openalex_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    citations: int = 0
    doi: str = ""
    url: str = ""
    authors: str = ""
    work_type: str = ""
    referenced_works: list[str] = field(default_factory=list)
    pdf_urls: list[str] = field(default_factory=list)
    landing_urls: list[str] = field(default_factory=list)
    survey_reference_links: list["SurveyReferenceLink"] = field(default_factory=list)
    relation: str = ""

    @property
    def key(self) -> str:
        return self.doi.casefold() or self.openalex_id.casefold() or compact_key(self.title)


@dataclass
class DatasetMention:
    raw_name: str
    source_work_id: str
    source_title: str
    source_kind: str
    context: str


@dataclass
class SurveyReferenceLink:
    dataset_name: str
    reference_title: str
    source_work_id: str
    source_title: str
    label: str
    context: str


@dataclass
class CandidateScore:
    work: Work
    score: float
    status: str
    reason: str


@dataclass
class TaxonomyDecision:
    label: str
    evidence: str
    score: float = 0.0


@dataclass
class SynonymDecision:
    stage: str
    left_key: str
    right_key: str
    left_name: str
    right_name: str
    decision: str
    score: float
    reason: str
    left_years: str
    right_years: str


@dataclass
class DatasetEntity:
    key: str
    aliases: set[str] = field(default_factory=set)
    mentions: list[DatasetMention] = field(default_factory=list)
    candidates: dict[str, Work] = field(default_factory=dict)
    rejected_candidates: list[CandidateScore] = field(default_factory=list)
    canonical: CandidateScore | None = None
    synonym_decisions: list[SynonymDecision] = field(default_factory=list)

    def add_alias(self, name: str) -> None:
        name = canonical_dataset_name(name)
        if name:
            self.aliases.add(name)

    @property
    def display_name(self) -> str:
        if not self.aliases:
            return self.key
        context_text = " ".join(f"{mention.raw_name} {mention.context} {mention.source_title}" for mention in self.mentions)

        def alias_score(alias: str) -> tuple[int, int, str]:
            alias_norm = normalize_text(alias)
            hits = len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(alias_norm)}(?![A-Za-z0-9])", context_text, flags=re.I))
            return (-hits, len(alias), alias.casefold())

        return sorted(self.aliases, key=alias_score)[0]

    @property
    def survey_count(self) -> int:
        return len({mention.source_work_id for mention in self.mentions})

    def merge_from(self, other: "DatasetEntity") -> None:
        self.aliases.update(other.aliases)
        self.mentions.extend(other.mentions)
        self.candidates.update(other.candidates)
        self.rejected_candidates.extend(other.rejected_candidates)
        self.synonym_decisions.extend(other.synonym_decisions)
        if other.canonical and (self.canonical is None or other.canonical.score > self.canonical.score):
            self.canonical = other.canonical


TRUSTED_SURVEY_MENTION_KINDS = {
    "survey_metadata",
    "survey_pdf_table",
    "survey_pdf_text",
    "survey_reference_link",
}


def trusted_survey_count(entity: DatasetEntity) -> int:
    return len(
        {
            mention.source_work_id
            for mention in entity.mentions
            if mention.source_kind in TRUSTED_SURVEY_MENTION_KINDS
        }
    )


def work_to_json(work: Work) -> dict:
    return {
        "openalex_id": work.openalex_id,
        "title": work.title,
        "abstract": work.abstract,
        "year": work.year,
        "citations": work.citations,
        "doi": work.doi,
        "url": work.url,
        "authors": work.authors,
        "work_type": work.work_type,
        "referenced_works": list(work.referenced_works),
        "pdf_urls": list(work.pdf_urls),
        "landing_urls": list(work.landing_urls),
        "survey_reference_links": [survey_reference_link_to_json(link) for link in work.survey_reference_links],
        "relation": work.relation,
    }


def work_from_json(data: dict) -> Work:
    return Work(
        openalex_id=str(data.get("openalex_id") or ""),
        title=str(data.get("title") or ""),
        abstract=str(data.get("abstract") or ""),
        year=data.get("year"),
        citations=int(data.get("citations") or 0),
        doi=str(data.get("doi") or ""),
        url=str(data.get("url") or ""),
        authors=str(data.get("authors") or ""),
        work_type=str(data.get("work_type") or ""),
        referenced_works=list(data.get("referenced_works") or []),
        pdf_urls=list(data.get("pdf_urls") or []),
        landing_urls=list(data.get("landing_urls") or []),
        survey_reference_links=[
            survey_reference_link_from_json(item) for item in data.get("survey_reference_links") or []
        ],
        relation=str(data.get("relation") or ""),
    )


def mention_to_json(mention: DatasetMention) -> dict:
    return {
        "raw_name": mention.raw_name,
        "source_work_id": mention.source_work_id,
        "source_title": mention.source_title,
        "source_kind": mention.source_kind,
        "context": mention.context,
    }


def mention_from_json(data: dict) -> DatasetMention:
    return DatasetMention(
        raw_name=str(data.get("raw_name") or ""),
        source_work_id=str(data.get("source_work_id") or ""),
        source_title=str(data.get("source_title") or ""),
        source_kind=str(data.get("source_kind") or ""),
        context=str(data.get("context") or ""),
    )


def survey_reference_link_to_json(link: SurveyReferenceLink) -> dict:
    return {
        "dataset_name": link.dataset_name,
        "reference_title": link.reference_title,
        "source_work_id": link.source_work_id,
        "source_title": link.source_title,
        "label": link.label,
        "context": link.context,
    }


def survey_reference_link_from_json(data: dict) -> SurveyReferenceLink:
    return SurveyReferenceLink(
        dataset_name=str(data.get("dataset_name") or ""),
        reference_title=str(data.get("reference_title") or ""),
        source_work_id=str(data.get("source_work_id") or ""),
        source_title=str(data.get("source_title") or ""),
        label=str(data.get("label") or ""),
        context=str(data.get("context") or ""),
    )


def candidate_score_to_json(score: CandidateScore | None) -> dict | None:
    if score is None:
        return None
    return {
        "work": work_to_json(score.work),
        "score": score.score,
        "status": score.status,
        "reason": score.reason,
    }


def candidate_score_from_json(data: dict | None) -> CandidateScore | None:
    if not data:
        return None
    return CandidateScore(
        work=work_from_json(data.get("work") or {}),
        score=float(data.get("score") or 0.0),
        status=str(data.get("status") or ""),
        reason=str(data.get("reason") or ""),
    )


def synonym_decision_to_json(decision: SynonymDecision) -> dict:
    return {
        "stage": decision.stage,
        "left_key": decision.left_key,
        "right_key": decision.right_key,
        "left_name": decision.left_name,
        "right_name": decision.right_name,
        "decision": decision.decision,
        "score": decision.score,
        "reason": decision.reason,
        "left_years": decision.left_years,
        "right_years": decision.right_years,
    }


def synonym_decision_from_json(data: dict) -> SynonymDecision:
    return SynonymDecision(
        stage=str(data.get("stage") or ""),
        left_key=str(data.get("left_key") or ""),
        right_key=str(data.get("right_key") or ""),
        left_name=str(data.get("left_name") or ""),
        right_name=str(data.get("right_name") or ""),
        decision=str(data.get("decision") or ""),
        score=float(data.get("score") or 0.0),
        reason=str(data.get("reason") or ""),
        left_years=str(data.get("left_years") or ""),
        right_years=str(data.get("right_years") or ""),
    )


def entity_to_json(entity: DatasetEntity) -> dict:
    return {
        "key": entity.key,
        "aliases": sorted(entity.aliases),
        "mentions": [mention_to_json(mention) for mention in entity.mentions],
        "candidates": {key: work_to_json(work) for key, work in entity.candidates.items()},
        "rejected_candidates": [candidate_score_to_json(score) for score in entity.rejected_candidates],
        "canonical": candidate_score_to_json(entity.canonical),
        "synonym_decisions": [synonym_decision_to_json(decision) for decision in entity.synonym_decisions],
    }


def entity_from_json(data: dict) -> DatasetEntity:
    entity = DatasetEntity(key=str(data.get("key") or ""))
    entity.aliases = set(data.get("aliases") or [])
    entity.mentions = [mention_from_json(item) for item in data.get("mentions") or []]
    entity.candidates = {
        str(key): work_from_json(work_data)
        for key, work_data in (data.get("candidates") or {}).items()
    }
    entity.rejected_candidates = [
        score
        for score in (candidate_score_from_json(item) for item in data.get("rejected_candidates") or [])
        if score is not None
    ]
    entity.canonical = candidate_score_from_json(data.get("canonical"))
    entity.synonym_decisions = [
        synonym_decision_from_json(item) for item in data.get("synonym_decisions") or []
    ]
    return entity


def normalize_text(text: str | None) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def comparable(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold()).strip()


def compact_key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text).casefold())


def alnum_tokens(text: str | None) -> list[str]:
    return re.findall(r"[a-z]+|\d+", comparable(text))


def year_from_token(token: str) -> int | None:
    if re.fullmatch(r"(?:19|20)\d{2}", token):
        year = int(token)
    elif re.fullmatch(r"\d{2}", token):
        value = int(token)
        if value < 80:
            return None
        year = 1900 + value
    else:
        return None
    if MIN_YEAR_GUARD <= year <= MAX_REASONABLE_RELEASE_YEAR:
        return year
    return None


def explicit_years_from_name(name: str) -> set[int]:
    years: set[int] = set()
    for token in alnum_tokens(name):
        year = year_from_token(token)
        if year is not None:
            years.add(year)
    return years


def significant_name_tokens(name: str, *, keep_years: bool = False) -> list[str]:
    tokens: list[str] = []
    for token in alnum_tokens(name):
        year = year_from_token(token)
        if year is not None and not keep_years:
            continue
        if token in WEAK_NAME_TOKENS:
            continue
        tokens.append(token)
    return tokens


def compact_name_base(name: str) -> str:
    return "".join(significant_name_tokens(name))


def entity_years_from_aliases(entity: "DatasetEntity") -> set[int]:
    years: set[int] = set()
    for alias in entity.aliases:
        years.update(explicit_years_from_name(alias))
    return years


def compatible_year_sets(left: set[int], right: set[int]) -> bool:
    return not left or not right or bool(left & right)


def name_similarity(left: str, right: str) -> float:
    left_base = compact_name_base(left)
    right_base = compact_name_base(right)
    if not left_base or not right_base:
        return 0.0
    return difflib.SequenceMatcher(None, left_base, right_base).ratio()


def edit_distance(left: str, right: str, limit: int = 3) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
            row_min = min(row_min, current[-1])
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def has_phrase(text: str | None, phrases: Iterable[str]) -> bool:
    normalized = f" {comparable(text)} "
    return any(f" {comparable(phrase)} " in normalized for phrase in phrases if comparable(phrase))


def first_phrase_position(text: str | None, phrases: Iterable[str]) -> int | None:
    normalized = comparable(text)
    positions = [normalized.find(comparable(phrase)) for phrase in phrases if comparable(phrase)]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else None


def detector_wording_before_dataset_wording(title: str) -> bool:
    detector_position = first_phrase_position(
        title,
        (
            "intrusion detection system",
            "intrusion detection systems",
            "intrusion detection",
            "network intrusion detection",
            "attack detection",
            "anomaly detection",
            "detection system",
            "detection method",
            "classification",
            "classifier",
            "machine learning",
            "deep learning",
            "model",
            "algorithm",
            "framework",
        ),
    )
    dataset_position = first_phrase_position(title, DATASET_WORDS)
    return detector_position is not None and (dataset_position is None or detector_position < dataset_position)


def request_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    source: str = "OpenAlex",
) -> dict:
    backoff = 1.0
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 429:
                try:
                    wait = int(float(response.headers.get("Retry-After") or backoff))
                except ValueError:
                    wait = int(backoff)
                if wait > MAX_RETRY_AFTER_SECONDS:
                    raise OpenAlexRateLimitError(wait, source)
                print(f"  {source} rate limited, sleeping {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_RETRY_AFTER_SECONDS)
                continue
            if 500 <= response.status_code < 600 and attempt < REQUEST_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_RETRY_AFTER_SECONDS)
                continue
            response.raise_for_status()
            return response.json()
        except OpenAlexRateLimitError:
            raise
        except requests.RequestException as exc:
            if attempt >= REQUEST_RETRIES:
                raise RuntimeError(f"{source} request failed for {url}: {exc}") from exc
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_RETRY_AFTER_SECONDS)
    raise RuntimeError(f"{source} request failed for {url}")


def openalex_select_fields() -> str:
    return ",".join(
        (
            "id",
            "doi",
            "title",
            "display_name",
            "publication_year",
            "cited_by_count",
            "abstract_inverted_index",
            "primary_location",
            "best_oa_location",
            "locations",
            "authorships",
            "referenced_works",
            "type",
            "type_crossref",
        )
    )


def reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    return normalize_text(" ".join(word for _index, word in sorted(positions)))


def extract_urls_from_location(location: dict | None) -> tuple[str, str]:
    if not location:
        return "", ""
    return normalize_text(location.get("landing_page_url") or ""), normalize_text(location.get("pdf_url") or "")


def work_from_openalex(item: dict, relation: str = "") -> Work:
    title = normalize_text(item.get("title") or item.get("display_name") or "")
    landing_urls: list[str] = []
    pdf_urls: list[str] = []
    for location in [item.get("primary_location"), item.get("best_oa_location"), *(item.get("locations") or [])]:
        landing_url, pdf_url = extract_urls_from_location(location)
        if landing_url and landing_url not in landing_urls:
            landing_urls.append(landing_url)
        if pdf_url and pdf_url not in pdf_urls:
            pdf_urls.append(pdf_url)

    authors: list[str] = []
    for authorship in (item.get("authorships") or [])[:8]:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            authors.append(author["display_name"])

    return Work(
        openalex_id=item.get("id") or "",
        title=title,
        abstract=reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
        year=item.get("publication_year"),
        citations=int(item.get("cited_by_count") or 0),
        doi=normalize_text(item.get("doi") or ""),
        url=landing_urls[0] if landing_urls else normalize_text(item.get("doi") or item.get("id") or ""),
        authors=", ".join(authors),
        work_type=item.get("type") or item.get("type_crossref") or "",
        referenced_works=list(item.get("referenced_works") or []),
        pdf_urls=pdf_urls,
        landing_urls=landing_urls,
        relation=relation,
    )


def openalex_params(args: argparse.Namespace, extra: dict | None = None) -> dict:
    params = dict(extra or {})
    if args.mailto:
        params["mailto"] = args.mailto
    if args.api_key:
        params["api_key"] = args.api_key
    return params


def openalex_search(
    session: requests.Session,
    args: argparse.Namespace,
    query: str,
    *,
    per_page: int,
    pages: int,
    relation: str,
    sort: str = "cited_by_count:desc",
    filters: str = "",
) -> list[Work]:
    works: list[Work] = []
    for page in range(1, pages + 1):
        try:
            payload = request_json(
                session,
                OPENALEX_WORKS_URL,
                openalex_params(
                    args,
                    {
                        "search": query,
                        "per-page": per_page,
                        "page": page,
                        "sort": sort,
                        "select": openalex_select_fields(),
                        **({"filter": filters} if filters else {}),
                    },
                ),
            )
        except RuntimeError as exc:
            if isinstance(exc, OpenAlexRateLimitError) and getattr(args, "stop_on_rate_limit", True):
                raise
            print(f"  OpenAlex search skipped for {query!r} page {page}: {exc}", file=sys.stderr)
            break
        for item in payload.get("results") or []:
            works.append(work_from_openalex(item, relation=relation))
        time.sleep(args.pause)
    return works


def fetch_openalex_work(
    session: requests.Session,
    args: argparse.Namespace,
    locator: str,
    *,
    relation: str,
    cache: dict[str, Work] | None = None,
) -> Work | None:
    normalized = normalize_work_locator(locator)
    cache_key = normalized.casefold()
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        return Work(**{**cached.__dict__, "relation": relation})
    if not normalized.startswith(OPENALEX_WORKS_URL):
        return None
    try:
        payload = request_json(
            session,
            normalized,
            openalex_params(args, {"select": openalex_select_fields()}),
        )
    except RuntimeError as exc:
        if isinstance(exc, OpenAlexRateLimitError) and getattr(args, "stop_on_rate_limit", True):
            raise
        print(f"  OpenAlex work skipped for {normalized}: {exc}", file=sys.stderr)
        return None
    work = work_from_openalex(payload, relation=relation)
    if cache is not None:
        cache[cache_key] = work
    time.sleep(args.pause)
    return work


def normalize_work_locator(locator: str) -> str:
    locator = locator.strip()
    if locator.startswith("https://openalex.org/"):
        return locator.replace("https://openalex.org/", "https://api.openalex.org/works/")
    if re.fullmatch(r"W\d+", locator):
        return f"{OPENALEX_WORKS_URL}/{locator}"
    if locator.startswith("doi:") or locator.startswith("https://doi.org/"):
        return f"{OPENALEX_WORKS_URL}/{locator}"
    if re.fullmatch(r"10\.\S+/\S+", locator):
        return f"{OPENALEX_WORKS_URL}/doi:{locator}"
    return locator


def fetch_work_by_locator(session: requests.Session, args: argparse.Namespace, locator: str) -> Work | None:
    locator = locator.strip()
    if not locator:
        return None
    normalized = normalize_work_locator(locator)
    if normalized.startswith(OPENALEX_WORKS_URL):
        return fetch_openalex_work(session, args, normalized, relation="trusted_input")

    # Treat any remaining locator as a title query and pick the highest-scoring survey-like work.
    candidates = openalex_search(
        session,
        args,
        locator,
        per_page=10,
        pages=1,
        relation="trusted_title_search",
        sort="relevance_score:desc",
    )
    if not candidates:
        return None
    candidates = [work for work in candidates if meets_citation_floor(work, args.min_survey_citations)]
    if not candidates:
        return None
    candidates.sort(key=lambda work: (-score_survey_work(work), -work.citations, work.year or 9999))
    return candidates[0]


def read_trusted_paper_inputs(args: argparse.Namespace) -> list[str]:
    locators = list(args.trusted_paper or [])
    if args.trusted_papers_file:
        for raw_line in args.trusted_papers_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                prefix, value = line.split(":", 1)
                if prefix.casefold() in {"doi", "openalex", "title", "paper"}:
                    line = value.strip()
            locators.append(line)
    return list(dict.fromkeys(locators))


def score_survey_work(work: Work) -> int:
    title = work.title
    score = 0
    survey_in_title = has_phrase(title, SURVEY_WORDS)
    dataset_in_title = has_phrase(title, DATASET_WORDS)
    scope_in_title = has_phrase(title, IDS_SURVEY_SCOPE_WORDS)

    if survey_in_title:
        score += 5
    if dataset_in_title:
        score += 4
    if has_phrase(title, SECURITY_WORDS):
        score += 4
    if scope_in_title:
        score += 6

    if not survey_in_title:
        score -= 8
    if not dataset_in_title:
        score -= 6
    if not scope_in_title:
        score -= 8
    if not (dataset_in_title or scope_in_title):
        score -= 2
    if has_phrase(title, BROAD_SURVEY_TITLE_WORDS) and not (scope_in_title and dataset_in_title):
        score -= 10

    if has_phrase(title, ("machine learning", "deep learning")) and not has_phrase(title, DATASET_WORDS):
        score -= 3
    if has_phrase(title, ("using", "classification", "classifier", "model", "algorithm")):
        score -= 4
    if work.work_type in {"article", "review"}:
        score += 1
    if work.citations >= 100:
        score += 1
    if work.citations >= 500:
        score += 1
    return score


def citation_floor_reason(work: Work, min_citations: int, paper_kind: str) -> str:
    return f"{paper_kind} citations {work.citations} below minimum {min_citations}"


def meets_citation_floor(work: Work, min_citations: int) -> bool:
    return work.citations >= min_citations


def discover_trusted_surveys(session: requests.Session, args: argparse.Namespace) -> list[Work]:
    trusted: dict[str, Work] = {}
    for query in args.survey_query or DEFAULT_SURVEY_QUERIES:
        print(f"Survey search: {query}", file=sys.stderr)
        works = openalex_search(
            session,
            args,
            query,
            per_page=args.survey_per_page,
            pages=args.survey_pages,
            relation="survey_search",
            sort="relevance_score:desc",
        )
        for work in works:
            survey_score = score_survey_work(work)
            if not meets_citation_floor(work, args.min_survey_citations):
                if args.verbose_skips:
                    print(
                        f"  Survey skipped: {citation_floor_reason(work, args.min_survey_citations, 'survey')} - {work.title}",
                        file=sys.stderr,
                    )
                continue
            if survey_score >= int(args.min_survey_score):
                trusted[work.key] = work
    ordered = sorted(trusted.values(), key=lambda item: (-score_survey_work(item), -item.citations, item.year or 9999))
    result = ordered[: args.max_surveys]
    for r in result:
        print(f"{r.title} - citations {r.citations} - heuristic: {score_survey_work(r)}")
    return result


def load_trusted_surveys(session: requests.Session, args: argparse.Namespace) -> list[Work]:
    locators = read_trusted_paper_inputs(args)
    trusted_by_key: dict[str, Work] = {}
    if locators:
        for locator in locators:
            try:
                work = fetch_work_by_locator(session, args, locator)
            except Exception as exc:
                print(f"Trusted paper skipped ({locator}): {exc}", file=sys.stderr)
                continue
            if work:
                if not meets_citation_floor(work, args.min_survey_citations):
                    print(
                        f"Trusted paper skipped ({locator}): "
                        f"{citation_floor_reason(work, args.min_survey_citations, 'survey')}",
                        file=sys.stderr,
                    )
                    continue
                trusted_by_key[work.key] = work
    if args.auto_surveys:
        for work in discover_trusted_surveys(session, args):
            trusted_by_key.setdefault(work.key, work)
    return list(trusted_by_key.values())


def canonical_dataset_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(
        r"\b(datasets?|data sets?|benchmarks?|corpus|corpora|testbeds?|traffic traces|traffic archive)\b$",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)
    name = name.strip(" ,.;:()[]{}\"'")
    name = re.sub(r"\s*-\s*", "-", name)
    name = re.sub(r"\s*/\s*", "/", name)
    name = re.sub(r"\s*_\s*", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name


def dataset_entity_key(name: str) -> str:
    key = compact_key(canonical_dataset_name(name))
    key = re.sub(r"(dataset|benchmark|corpus|testbed)$", "", key)
    return key


def looks_like_model_or_method_name(name: str) -> bool:
    name_norm = comparable(name)
    tokens = alnum_tokens(name)
    if not tokens:
        return False
    model_hits = [token for token in tokens if token in MODEL_METHOD_NAME_TOKENS]
    if not model_hits:
        return False
    if has_phrase(name, DATASET_WORDS):
        return False
    if explicit_years_from_name(name):
        return False
    if len(tokens) <= 4:
        return True
    return bool(re.search(r"\b(?:powered|based|model|classifier|classification|learning|network)\b", name_norm))


def looks_like_dataset_name(
    name: str,
    *,
    allow_low_precision: bool = False,
    allow_plain_name: bool = False,
) -> bool:
    name = canonical_dataset_name(name)
    if len(name) < 3 or len(name) > 80:
        return False
    if name in BAD_DATASET_NAMES or name.casefold() in {item.casefold() for item in BAD_DATASET_NAMES}:
        return False
    if looks_like_model_or_method_name(name):
        return False
    key = compact_key(name)
    if key in LOW_PRECISION_NAMES and not allow_low_precision:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)+", name):
        return False
    if re.search(r"(?:^|[-\s])(?:based|specific|fold|consisting)$", name, flags=re.I):
        return False
    if re.search(r"^(?:and|although|for|of|on|using|with)\b", name, flags=re.I):
        return False
    if re.search(r"'s$", name, flags=re.I):
        return False
    if has_phrase(name, GENERIC_NON_DATASET_NAME_WORDS):
        return False
    if re.search(r"\b(?:IEEE\s*)?802(?:\.\d+)?\b", name, flags=re.I):
        return False
    base = compact_name_base(name)
    if len(base) < 3:
        return False
    if re.fullmatch(r"\d{2,4}", name):
        return False
    if re.search(r"\b(model|method|algorithm|approach|framework|accuracy|precision|recall|table|figure|analysis|selected)\b", name, flags=re.I):
        return False
    has_digit = bool(re.search(r"\d", name))
    has_structural_separator = bool(re.search(r"[-_/+.']", name))
    has_acronym_token = " " in name and bool(re.search(r"\b[A-Z0-9]{2,}\b", name))
    has_internal_camel = bool(re.search(r"[a-z][A-Z]", name))
    all_caps = bool(re.fullmatch(r"[A-Z0-9][A-Z0-9_./'+-]{2,}", name))
    ends_with_set = name.casefold().endswith("set")
    plain_named_entity = bool(re.fullmatch(r"[A-Z][A-Za-z]{4,}(?:\s+\d{2,4}\+?)?", name))
    single_titlecase_word = bool(re.fullmatch(r"[A-Z][a-z]{4,}", name))
    titlecase_word_with_year = bool(re.fullmatch(r"[A-Z][a-z]{4,}\s+\d{2,4}\+?", name))
    if titlecase_word_with_year and not allow_plain_name:
        return False
    if single_titlecase_word:
        return False
    if all_caps and not has_digit and not has_structural_separator and len(name) > 6:
        return False
    return (
        has_digit
        or has_structural_separator
        or has_acronym_token
        or has_internal_camel
        or all_caps
        or ends_with_set
        or (allow_plain_name and plain_named_entity)
    )


def has_strong_dataset_name_shape(name: str) -> bool:
    name = canonical_dataset_name(name)
    return bool(
        re.search(r"\d", name)
        or re.search(r"[-_/+.']", name)
        or re.fullmatch(r"[A-Z0-9][A-Z0-9_./'+-]{2,}", name)
    )


NAME_EXPR = (
    r"[A-Za-z0-9]*\d[A-Za-z0-9]*(?:[-_./'+][A-Za-z0-9]+)+"
    r"|[A-Z][A-Za-z]{3,}\s+\d{2,4}\+?"
    r"|[A-Z]{3,}\s+\d{2,4}"
    r"|[A-Z]{2,}(?:\s+[A-Z]{2,})?\s+\d{2,4}[A-Za-z0-9+]*"
    r"|[A-Z]{2,}(?:[-_./'+][A-Za-z0-9]+)+"
    r"|[A-Z]{2,}(?:\s+[A-Z][A-Za-z0-9]*\d[A-Za-z0-9]*)+"
    r"|[A-Z]{2,}(?:\s+[A-Z][A-Za-z0-9]+)*(?:\s+Cup\s*'?99)"
    r"|[A-Z][A-Za-z0-9]*(?:[-_./'+][A-Za-z0-9]+)+"
    r"|[A-Z]{2,}[A-Z0-9_./'+-]*\d[A-Z0-9_./'+-]*"
    r"|[A-Z]{3,}(?:[-_./'+][A-Z0-9]+)*"
    r"|[A-Z][A-Za-z]+(?:[A-Z][A-Za-z0-9]+)+(?:\d{2,4})?"
)
DATASET_EXPR = (
    r"(?i:dataset|data set|benchmark|corpus|testbed|traffic traces|trace dataset|traffic archive|"
    r"network traces|packet traces|flow traces|internet traffic|network traffic data|traffic data)"
)

LIST_SEPARATOR_EXPR = r"(?:,|;|\band\b|\bor\b|/)"


def normalize_dataset_name_variant(name: str) -> str:
    name = canonical_dataset_name(name)
    name = re.sub(
        r"\b([A-Z]{2,})\s+([A-Z]{2,}\d{1,4}[A-Za-z0-9+]*)\b",
        r"\1-\2",
        name,
    )
    name = re.sub(r"\b([A-Z]{2,})\s+(\d{2,4}[A-Za-z0-9+]*)\b", r"\1-\2", name)
    name = re.sub(r"\b(Cup)\s*'?(\d{2})\b", r"\1'\2", name, flags=re.I)
    return canonical_dataset_name(name)


def high_recall_context_windows(text: str, *, max_windows: int = 250) -> list[str]:
    windows: list[str] = []
    for chunk in re.split(r"(?:\n+|(?<=[.;:])\s+)", normalize_text(text)):
        if len(windows) >= max_windows:
            break
        chunk = chunk.strip()
        if len(chunk) < 12:
            continue
        if not has_phrase(chunk, DATASET_WORDS):
            continue
        list_like = bool(re.search(r"\b(?:include|including|such as|namely|e\.g\.|for example)\b", chunk, flags=re.I))
        trace_like = has_phrase(
            chunk,
            (
                "publicly available network traces",
                "public network traces",
                "network traces",
                "traffic traces",
                "internet traffic",
            ),
        )
        if not list_like and not trace_like and not has_phrase(chunk, SECURITY_WORDS + GENERATOR_WORDS):
            continue
        windows.append(chunk[:900])
    return windows


def high_recall_dataset_name_patterns(text: str) -> set[str]:
    names: set[str] = set()
    for window in high_recall_context_windows(text):
        for anchor in re.finditer(
            r"\b(?:datasets?|data sets?|benchmarks?|corpora|traffic traces)\s+"
            r"(?:include|including|such as|namely|e\.g\.|for example)\s+(.{0,500})",
            window,
            flags=re.I,
        ):
            sequence = re.split(r"\b(?:for|that|which|where|while|although)\b", anchor.group(1), maxsplit=1, flags=re.I)[0]
            for item in re.findall(NAME_EXPR, sequence):
                names.add(normalize_dataset_name_variant(item if isinstance(item, str) else item[0]))

        for anchor in re.finditer(
            r"(.{0,500})\s+(?:datasets?|data sets?|benchmarks?|corpora|traffic traces)\b",
            window,
            flags=re.I,
        ):
            sequence = re.split(r"\b(?:using|evaluated|tested|trained|applied|compared)\b", anchor.group(1), maxsplit=1, flags=re.I)[-1]
            if not re.search(LIST_SEPARATOR_EXPR, sequence, flags=re.I):
                continue
            for item in re.findall(NAME_EXPR, sequence):
                names.add(normalize_dataset_name_variant(item if isinstance(item, str) else item[0]))

        adjacent_patterns = (
            rf"\b({NAME_EXPR})\b\s+(?:intrusion\s+detection\s+|network\s+security\s+|traffic\s+)?(?:{DATASET_EXPR})\b",
            rf"\b(?:{DATASET_EXPR})\s*(?::|-)?\s*(?:called|named|known as|referred to as)?\s*(?:the\s+)?({NAME_EXPR})\b",
        )
        for pattern in adjacent_patterns:
            for match in re.finditer(pattern, window, flags=re.I):
                candidate = match.group(1)
                end = match.end(1)
                if end < len(window) and window[end : end + 1] == "+":
                    candidate = f"{candidate}+"
                names.add(normalize_dataset_name_variant(candidate))
    plus_bases = {compact_key(name) for name in names if name.endswith("+")}
    return {
        name
        for name in names
        if name.endswith("+") or compact_key(name) not in plus_bases
        if looks_like_dataset_name(name, allow_low_precision=True, allow_plain_name=True)
    }


def extract_dataset_names(
    text: str,
    *,
    allow_low_precision: bool = False,
    allow_plain_names: bool = False,
) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    candidates: set[str] = set()
    patterns = [
        rf"^\s*({NAME_EXPR})\s*(?::|\s+-\s+).{{0,180}}\b{DATASET_EXPR}\b",
        rf"\b({NAME_EXPR})\b\s+(?:{DATASET_EXPR})\b",
        rf"\b(?:{DATASET_EXPR})\s*[-:]\s*({NAME_EXPR})\b",
        rf"\b(?:{DATASET_EXPR})\s+(?i:called|named|known as|referred to as)\s+(?i:the\s+)?({NAME_EXPR})\b",
        rf"\b(?i:called|named|known as|referred to as)\s+(?i:the\s+)?({NAME_EXPR})\b.{{0,100}}\b{DATASET_EXPR}\b",
        rf"[:;]\s*({NAME_EXPR})\s+(?:{DATASET_EXPR})\b",
        rf"\b({NAME_EXPR})\b.{{0,120}}\b(?i:public|labeled|labelled|annotated|realistic|real-world|synthetic)\s+{DATASET_EXPR}\b",
    ]
    if allow_plain_names:
        plain_name_expr = r"[A-Z][A-Za-z]{4,}(?:\s+\d{2,4}\+?)?"
        patterns.extend(
            [
                rf"^\s*({plain_name_expr})\s*(?::|\s+-\s+).{{0,180}}\b(?:{DATASET_EXPR}|traffic\s+generator|anomaly\s+modeling\s+engine|dataset\s+creation\s+toolkit)\b",
                rf"\b({plain_name_expr})\b.{{0,80}}\b(?:{DATASET_EXPR}|traffic\s+generator|anomaly\s+modeling\s+engine|dataset\s+creation\s+toolkit)\b",
                rf"\b(?:{DATASET_EXPR}|traffic\s+generator|anomaly\s+modeling\s+engine|dataset\s+creation\s+toolkit)\b.{{0,80}}\b({plain_name_expr})\b",
            ]
        )
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.M):
            candidate = normalize_dataset_name_variant(match if isinstance(match, str) else match[0])
            if looks_like_dataset_name(
                candidate,
                allow_low_precision=allow_low_precision,
                allow_plain_name=allow_plain_names,
            ):
                candidates.add(candidate)
    candidates.update(high_recall_dataset_name_patterns(text))
    return candidates


def short_context(text: str, name: str, width: int = 220) -> str:
    text = normalize_text(text)
    plain_index = text.casefold().find(name.casefold())
    if plain_index < 0:
        key_index = compact_key(text).find(compact_key(name))
        if key_index < 0:
            return text[:width]
        return text[:width]
    start = max(0, plain_index - width // 2)
    end = min(len(text), plain_index + len(name) + width // 2)
    return text[start:end].strip()


def table_like_context(text: str, name: str) -> bool:
    context = comparable(short_context(text, name, width=700))
    heading_hits = sum(1 for heading in DATASET_TABLE_HEADINGS if f" {comparable(heading)} " in f" {context} ")
    return heading_hits >= 2 or has_phrase(context, ("dataset year", "dataset attacks", "dataset traffic"))


def strong_survey_name_context(text: str, name: str) -> bool:
    context = comparable(short_context(text, name, width=700))
    if table_like_context(text, name):
        return True
    return has_phrase(
        context,
        (
            "dataset",
            "data set",
            "traffic generator",
            "traffic generation",
            "dataset creation toolkit",
            "anomaly modeling engine",
            "flow level anomaly modeling",
            "flow-level anomaly modeling",
            "testbed",
        ),
    ) and has_phrase(context, SECURITY_WORDS + GENERATOR_WORDS)


def fetch_pdf_text(session: requests.Session, url: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  PDF fetch skipped for {url}: {exc}", file=sys.stderr)
        return ""
    body = response.content
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "paper.pdf"
        txt_path = Path(tmpdir) / "paper.txt"
        pdf_path.write_bytes(body)
        try:
            subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), str(txt_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            print("  pdftotext unavailable or failed; PDF text skipped", file=sys.stderr)
            return ""
        return normalize_text(txt_path.read_text(encoding="utf-8", errors="ignore"))[:max_chars]


def split_body_and_references(fulltext: str) -> tuple[str, str]:
    lines = fulltext.splitlines()
    reference_start = None
    for index, line in enumerate(lines):
        heading = comparable(line)
        if heading in {"references", "bibliography"}:
            reference_start = index
    if reference_start is None:
        matches = list(re.finditer(r"\b(?:References|Bibliography)\b", fulltext, flags=re.I))
        if not matches:
            return fulltext, ""
        match = matches[-1]
        return fulltext[: match.start()], fulltext[match.end() :]
    return "\n".join(lines[:reference_start]), "\n".join(lines[reference_start + 1 :])


def numbered_reference_entries(reference_text: str) -> dict[str, str]:
    reference_text = normalize_text(reference_text)
    inline_matches = list(re.finditer(r"(?:^|\s)(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+", reference_text))
    if inline_matches:
        entries: dict[str, str] = {}
        for index, match in enumerate(inline_matches):
            label = match.group(1) or match.group(2) or ""
            end = inline_matches[index + 1].start() if index + 1 < len(inline_matches) else len(reference_text)
            entry = normalize_text(reference_text[match.end() : end])
            if label and entry:
                entries[label] = entry
        return entries

    entries: dict[str, str] = {}
    current_label = ""
    current_parts: list[str] = []
    for raw_line in reference_text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        match = re.match(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+(.+)$", line)
        if match:
            if current_label and current_parts:
                entries[current_label] = normalize_text(" ".join(current_parts))
            current_label = match.group(1) or match.group(2) or ""
            current_parts = [match.group(3)]
            continue
        if current_label:
            current_parts.append(line)
    if current_label and current_parts:
        entries[current_label] = normalize_text(" ".join(current_parts))
    return entries


def trim_reference_title(candidate: str) -> str:
    candidate = normalize_text(candidate)
    candidate = re.split(
        r"\.\s+(?:In:|In\s+|Proceedings of|Proc\.|IEEE|ACM|Springer|Elsevier|Future|Int\.?\s+J|Journal of)\b",
        candidate,
        maxsplit=1,
        flags=re.I,
    )[0]
    candidate = re.split(
        r"\s+(?:In:|Proceedings of|Proc\.|IEEE|ACM|Springer|Elsevier|Journal of|Int\.|International Conference)",
        candidate,
        maxsplit=1,
        flags=re.I,
    )[0]
    candidate = re.sub(r"\s+(?:pp\.?|pages?)\s+\d+.*$", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+https?://\S+.*$", "", candidate, flags=re.I)
    candidate = candidate.strip(" .,:;\"'")
    return candidate


def dataset_source_titles_from_reference_entry(entry: str) -> list[str]:
    text = normalize_text(entry)
    titles: list[str] = []
    start_re = (
        r"\b(?:towards?|generat(?:ing|ion)|construct(?:ing|ion)|develop(?:ing|ment)|"
        r"creat(?:ing|ion)|introduc(?:ing|es|e)|new|novel|realistic|labeled|labelled|annotated)\b"
    )
    target_re = r"\b(?:dataset|data set|benchmark|testbed|traffic characterization|traffic generation)\b"
    for match in re.finditer(rf"{start_re}.{{0,260}}{target_re}.{{0,180}}", text, flags=re.I):
        candidate = trim_reference_title(match.group(0))
        if len(candidate) < 25:
            continue
        if not has_phrase(candidate, SECURITY_WORDS + GENERATOR_WORDS):
            continue
        if title_has_dataset_creation_wording(candidate) or title_has_generator_creation_wording(candidate):
            titles.append(candidate)
    return list(dict.fromkeys(titles))


def citation_labels_near_name(text: str, name: str, *, radius: int = 320) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    for _alias, window in alias_context_windows(DatasetEntity(dataset_entity_key(name), aliases={name}), text, radius=radius):
        for citation in re.findall(r"\[([0-9,\-\s]+)\]", window):
            for part in re.split(r"\s*,\s*", citation):
                if "-" in part:
                    start_text, end_text = [piece.strip() for piece in part.split("-", 1)]
                    if start_text.isdigit() and end_text.isdigit():
                        start, end = int(start_text), int(end_text)
                        for value in range(start, min(end, start + 10) + 1):
                            labels.append((str(value), window))
                    continue
                if part.strip().isdigit():
                    labels.append((part.strip(), window))
    return labels


def extract_survey_reference_links(fulltext: str, survey: Work, dataset_names: set[str]) -> list[SurveyReferenceLink]:
    body_text, reference_text = split_body_and_references(fulltext)
    if not reference_text:
        return []
    titles_by_label: dict[str, list[str]] = {}
    for label, entry in numbered_reference_entries(reference_text).items():
        titles = dataset_source_titles_from_reference_entry(entry)
        if titles:
            titles_by_label[label] = titles
    if not titles_by_label:
        return []

    links: list[SurveyReferenceLink] = []
    seen: set[tuple[str, str, str]] = set()
    for name in sorted(dataset_names, key=lambda item: item.casefold()):
        for label, context in citation_labels_near_name(body_text, name):
            for title in titles_by_label.get(label, []):
                key = (dataset_entity_key(name), label, comparable(title))
                if key in seen:
                    continue
                seen.add(key)
                links.append(
                    SurveyReferenceLink(
                        dataset_name=name,
                        reference_title=title,
                        source_work_id=survey.openalex_id,
                        source_title=survey.title,
                        label=label,
                        context=f"{short_context(context, name, width=500)} [ref {label}: {title}]",
                    )
                )
    return links


def add_mention(
    entities: dict[str, DatasetEntity],
    name: str,
    source: Work,
    source_kind: str,
    context: str,
    *,
    allow_low_precision: bool = False,
    allow_plain_name: bool = False,
) -> DatasetEntity | None:
    name = canonical_dataset_name(name)
    if not looks_like_dataset_name(
        name,
        allow_low_precision=allow_low_precision,
        allow_plain_name=allow_plain_name,
    ):
        return None
    key = dataset_entity_key(name)
    if not key:
        return None
    entity = entities.setdefault(key, DatasetEntity(key=key))
    entity.add_alias(name)
    entity.mentions.append(
        DatasetMention(
            raw_name=name,
            source_work_id=source.openalex_id,
            source_title=source.title,
            source_kind=source_kind,
            context=context,
        )
    )
    return entity


def entity_has_plausible_dataset_alias(entity: DatasetEntity) -> bool:
    return any(
        looks_like_dataset_name(alias, allow_low_precision=True, allow_plain_name=True)
        for alias in entity.aliases
    )


def prune_implausible_entities(entities: dict[str, DatasetEntity], *, stage: str) -> int:
    removed: list[DatasetEntity] = []
    for key, entity in list(entities.items()):
        if entity_has_plausible_dataset_alias(entity):
            continue
        removed.append(entity)
        del entities[key]
    if removed:
        examples = ", ".join(entity.display_name for entity in removed[:12])
        suffix = "" if len(removed) <= 12 else f", ... +{len(removed) - 12} more"
        print(
            f"Pruned {len(removed)} implausible dataset entities before {stage}: {examples}{suffix}",
            file=sys.stderr,
        )
    return len(removed)


def entity_alias_bases(entity: DatasetEntity) -> set[str]:
    bases = {compact_name_base(alias) for alias in entity.aliases}
    return {base for base in bases if base}


def best_alias_similarity(left: DatasetEntity, right: DatasetEntity) -> float:
    best = 0.0
    for left_alias in left.aliases or {left.key}:
        for right_alias in right.aliases or {right.key}:
            best = max(best, name_similarity(left_alias, right_alias))
    return best


def synonym_candidate_decision(left: DatasetEntity, right: DatasetEntity, stage: str) -> SynonymDecision | None:
    left_years = entity_years_from_aliases(left)
    right_years = entity_years_from_aliases(right)
    left_bases = entity_alias_bases(left)
    right_bases = entity_alias_bases(right)
    shared_bases = left_bases & right_bases
    best_score = best_alias_similarity(left, right)
    left_name = left.display_name
    right_name = right.display_name
    left_year_text = ";".join(str(year) for year in sorted(left_years))
    right_year_text = ";".join(str(year) for year in sorted(right_years))

    def decision(kind: str, score: float, reason: str) -> SynonymDecision:
        return SynonymDecision(
            stage=stage,
            left_key=left.key,
            right_key=right.key,
            left_name=left_name,
            right_name=right_name,
            decision=kind,
            score=round(score, 4),
            reason=reason,
            left_years=left_year_text,
            right_years=right_year_text,
        )

    comparable_pair = bool(shared_bases) or best_score >= SYNONYM_NEAR_MISS_THRESHOLD
    if not comparable_pair:
        return None

    if not compatible_year_sets(left_years, right_years):
        return decision(
            "rejected_year_conflict",
            best_score,
            "lexically similar, but explicit year-like tokens differ",
        )

    if shared_bases:
        return decision("merged", 1.0, f"same weak-token-stripped base: {', '.join(sorted(shared_bases))}")

    if best_score >= SYNONYM_SIMILARITY_THRESHOLD:
        return decision("merged", best_score, "high character similarity after weak-token/year normalization")

    containment = any(
        (len(left_base) >= 5 and len(right_base) >= 5 and (left_base in right_base or right_base in left_base))
        for left_base in left_bases
        for right_base in right_bases
    )
    if containment and best_score >= 0.8:
        return decision("merged", best_score, "one normalized base contains the other")

    best_jaccard = 0.0
    for left_alias in left.aliases or {left.key}:
        left_tokens = set(significant_name_tokens(left_alias))
        for right_alias in right.aliases or {right.key}:
            right_tokens = set(significant_name_tokens(right_alias))
            best_jaccard = max(best_jaccard, token_jaccard(left_tokens, right_tokens))
    if best_jaccard >= 0.66 and best_score >= 0.72:
        return decision("merged", max(best_score, best_jaccard), "high token overlap after weak-token/year normalization")

    close_edit = any(
        len(left_base) >= 6
        and len(left_base) == len(right_base)
        and edit_distance(left_base, right_base, limit=2) <= 2
        for left_base in left_bases
        for right_base in right_bases
    )
    if close_edit:
        return decision("merged", best_score, "small edit distance between normalized bases")

    short_acronym_containment = False
    bad_short_tokens = {compact_key(name) for name in BAD_DATASET_NAMES} | LOW_PRECISION_NAMES
    for left_base in left_bases:
        for right_base in right_bases:
            short, long = sorted((left_base, right_base), key=len)
            if (
                3 <= len(short) <= 4
                and short not in bad_short_tokens
                and len(long) <= len(short) + 4
                and (long.startswith(short) or long.endswith(short))
            ):
                short_acronym_containment = True
    if short_acronym_containment and best_score >= 0.62:
        return decision("merged", best_score, "short acronym appears inside longer normalized base")

    return decision(
        "rejected_low_similarity",
        best_score,
        "looked related enough to audit, but below synonym threshold",
    )


def merge_synonym_entities(
    entities: dict[str, DatasetEntity],
    *,
    stage: str,
    verbose: bool,
) -> list[SynonymDecision]:
    decisions: list[SynonymDecision] = []
    changed = True
    while changed:
        changed = False
        keys = sorted(entities)
        for left_index, left_key in enumerate(keys):
            if left_key not in entities:
                continue
            left = entities[left_key]
            for right_key in keys[left_index + 1 :]:
                if right_key not in entities:
                    continue
                right = entities[right_key]
                decision = synonym_candidate_decision(left, right, stage)
                if decision is None:
                    continue
                decisions.append(decision)
                if verbose:
                    print(
                        f"  synonym {decision.decision}: {decision.left_name!r} <-> {decision.right_name!r} "
                        f"(score={decision.score}, years={decision.left_years or '-'}|{decision.right_years or '-'}; "
                        f"{decision.reason})",
                        file=sys.stderr,
                    )
                if decision.decision != "merged":
                    continue

                target, source = sorted(
                    (left, right),
                    key=lambda entity: (
                        -entity.survey_count,
                        -len(entity.candidates),
                        len(entity.display_name),
                        entity.key,
                    ),
                )
                target.synonym_decisions.append(decision)
                target.merge_from(source)
                entities[target.key] = target
                del entities[source.key]
                changed = True
                break
            if changed:
                break
    return decisions


def extract_entities_from_survey(session: requests.Session, args: argparse.Namespace, survey: Work, entities: dict[str, DatasetEntity]) -> None:
    metadata_text = survey.title
    for name in extract_dataset_names(metadata_text, allow_plain_names=False):
        context = short_context(metadata_text, name)
        add_mention(
            entities,
            name,
            survey,
            "survey_metadata",
            context,
            allow_plain_name=strong_survey_name_context(metadata_text, name),
        )

    if not args.fetch_pdf:
        return

    for pdf_url in survey.pdf_urls[: args.max_pdf_urls_per_survey]:
        fulltext = fetch_pdf_text(session, pdf_url, args.max_pdf_chars)
        if not fulltext:
            continue
        body_text, _reference_text = split_body_and_references(fulltext)
        dataset_names = extract_dataset_names(body_text, allow_low_precision=True, allow_plain_names=True)
        survey.survey_reference_links.extend(extract_survey_reference_links(fulltext, survey, dataset_names))
        for name in dataset_names:
            context = short_context(body_text, name, width=500)
            if not table_like_context(body_text, name):
                continue
            kind = "survey_pdf_table"
            strong_context = strong_survey_name_context(body_text, name)
            add_mention(
                entities,
                name,
                survey,
                kind,
                context,
                allow_low_precision=strong_context,
                allow_plain_name=strong_context,
            )


def aliases_in_text(entity: DatasetEntity, text: str, *, allow_low_precision: bool = False) -> list[str]:
    text_norm = normalize_text(text)
    text_key = compact_key(text_norm)
    matches: list[str] = []
    for alias in sorted(entity.aliases, key=len, reverse=True):
        alias_norm = normalize_text(alias)
        alias_key = compact_key(alias_norm)
        if not alias_key or (alias_key in LOW_PRECISION_NAMES and not allow_low_precision):
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias_norm)}(?![A-Za-z0-9])", text_norm, flags=re.I):
            matches.append(alias)
        elif len(alias_key) >= 6 and alias_key in text_key:
            matches.append(alias)
    return list(dict.fromkeys(matches))


def alias_pattern(alias: str) -> str:
    return rf"(?<![A-Za-z0-9]){re.escape(normalize_text(alias))}(?![A-Za-z0-9])"


def alias_is_parenthetical_dataset_abbrev(alias: str, text: str) -> bool:
    text_norm = normalize_text(text)
    alias_norm = normalize_text(alias)
    if not text_norm or not alias_norm:
        return False
    for match in re.finditer(rf"\(\s*{re.escape(alias_norm)}\s*\)", text_norm, flags=re.I):
        before = text_norm[max(0, match.start() - 180) : match.start()]
        if has_phrase(before, DATASET_WORDS):
            return True
    return False


def alias_is_named_dataset(alias: str, text: str) -> bool:
    text_norm = normalize_text(text)
    alias_norm = normalize_text(alias)
    if not text_norm or not alias_norm:
        return False
    alias_re = re.escape(alias_norm)
    dataset_re = DATASET_EXPR
    patterns = (
        rf"\b(?:{dataset_re})\s+(?i:called|named|known as|referred to as)\s+(?i:the\s+)?{alias_re}\b",
        rf"\b(?i:called|named|known as|referred to as)\s+(?i:the\s+)?{alias_re}\b.{{0,100}}\b{dataset_re}\b",
        rf"\b(?i:named|called)\s+(?:the\s+)?{alias_re}\b",
    )
    return any(re.search(pattern, text_norm, flags=re.I) for pattern in patterns)


def alias_is_adjacent_dataset_name(alias: str, text: str) -> bool:
    text_norm = normalize_text(text)
    alias_norm = normalize_text(alias)
    if not text_norm or not alias_norm:
        return False
    immediate_dataset = re.search(
        rf"{alias_pattern(alias)}\s+(?:dataset|data set|benchmark|corpus|testbed|traffic traces|traffic archive)\b",
        text_norm,
        flags=re.I,
    )
    if immediate_dataset:
        return True
    if has_strong_dataset_name_shape(alias):
        return bool(
            re.search(
                rf"{alias_pattern(alias)}.{{0,35}}\b(?:dataset|data set|benchmark|corpus|testbed)\b",
                text_norm,
                flags=re.I,
            )
        )
    return False


def alias_is_primary_dataset_name(alias: str, text: str) -> bool:
    return (
        alias_is_parenthetical_dataset_abbrev(alias, text)
        or alias_is_named_dataset(alias, text)
        or alias_is_adjacent_dataset_name(alias, text)
    )


def alias_is_primary_generator_name(alias: str, text: str) -> bool:
    text_norm = normalize_text(text)
    if not text_norm or not alias:
        return False
    generator_re = (
        r"(?:traffic generator|traffic generation|network traffic generation|trace generation|"
        r"attack generation|attack injection|anomaly injection|anomaly modeling engine|"
        r"flow-level anomaly modeling|flow level anomaly modeling|dataset creation toolkit|"
        r"toolkit|generator|framework|engine)"
    )
    return bool(
        title_has_generator_creation_wording(text_norm)
        and (
            re.search(rf"{alias_pattern(alias)}\s*(?::|-|,|\().{{0,100}}\b{generator_re}\b", text_norm, flags=re.I)
            or re.search(rf"\b{generator_re}\b.{{0,100}}{alias_pattern(alias)}", text_norm, flags=re.I)
            or alias_is_named_dataset(alias, text_norm)
        )
    )


def alias_context_windows(entity: DatasetEntity, text: str, *, radius: int = 180) -> list[tuple[str, str]]:
    text_norm = normalize_text(text)
    if not text_norm:
        return []
    windows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias in sorted(entity.aliases, key=len, reverse=True):
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(alias_norm)}(?![A-Za-z0-9])", text_norm, flags=re.I):
            window = text_norm[max(0, match.start() - radius) : match.end() + radius]
            key = (alias, window)
            if key in seen:
                continue
            seen.add(key)
            windows.append(key)
    return windows


def abstract_alias_evidence(entity: DatasetEntity, work: Work) -> list[tuple[str, str]]:
    evidence: list[tuple[str, str]] = []
    for alias, window in alias_context_windows(entity, work.abstract):
        if not has_phrase(window, DATASET_WORDS):
            continue
        if not has_phrase(window, SECURITY_WORDS + GENERATOR_WORDS):
            continue
        if has_phrase(window, ("using", "used", "trained", "tested", "evaluated", "applied", "compared")) and not has_phrase(
            window,
            CREATION_VERBS + INTRODUCTION_WORDS,
        ):
            continue
        evidence.append((alias, short_context(window, alias, width=260)))
    return evidence


def abstract_evidence_summary(entity: DatasetEntity, work: Work) -> str:
    evidence = abstract_alias_evidence(entity, work)
    if not evidence:
        if work.abstract and aliases_in_text(entity, work.abstract, allow_low_precision=True):
            return "alias in abstract without dataset-creation support"
        return ""
    return " | ".join(f"{alias}: {context}" for alias, context in evidence[:3])


def title_evidence_summary(entity: DatasetEntity, title: str) -> str:
    parts: list[str] = []
    generator_scope = has_generator_creation_evidence(title)
    if aliases_in_text(entity, title, allow_low_precision=generator_scope):
        parts.append("alias in title")
    if title_starts_with_alias(entity, title):
        parts.append("title starts with alias")
    if aliases_with_primary_dataset_evidence(entity, title):
        parts.append("title identifies alias as artifact")
    if title_has_dataset_creation_wording(title):
        parts.append("dataset-creation title")
    if title_has_generator_creation_wording(title):
        parts.append("generator/framework title")
    if is_downstream_or_analysis_title(title):
        parts.append("downstream/analysis title")
    return "; ".join(parts)


def aliases_with_primary_dataset_evidence(entity: DatasetEntity, text: str) -> list[str]:
    return [
        alias
        for alias in aliases_in_text(entity, text, allow_low_precision=True)
        if alias_is_primary_dataset_name(alias, text) or alias_is_primary_generator_name(alias, text)
    ]


def name_looks_like_component_in_text(name: str, text: str) -> bool:
    text_norm = normalize_text(text)
    name_norm = normalize_text(name)
    if not text_norm or not name_norm:
        return False
    follower_re = "|".join(re.escape(comparable(word)).replace(r"\ ", r"\s+") for word in COMPONENT_FOLLOWER_WORDS)
    text_cmp = comparable(text_norm)
    name_cmp = comparable(name_norm)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(name_cmp)}\s+(?:{follower_re})(?![a-z0-9])", text_cmp))


def title_starts_with_alias(entity: DatasetEntity, title: str) -> bool:
    title_norm = normalize_text(title)
    title_cmp = comparable(title)
    for alias in sorted(entity.aliases, key=len, reverse=True):
        alias_norm = normalize_text(alias)
        if re.search(rf"^\s*{re.escape(alias_norm)}\s*(?::|-|,|\()", title_norm, flags=re.I):
            return True
        alias_cmp = comparable(alias)
        if title_cmp == alias_cmp:
            return True
        if re.search(rf"^\s*{re.escape(alias_cmp)}\b\s*(?::|-|,|\()", title_cmp):
            return True
        if re.search(rf"^\s*{re.escape(alias_cmp)}\b\s+(?:dataset|benchmark|corpus|testbed|traffic archive)\b", title_cmp):
            return True
    return False


def title_has_dataset_creation_wording(title: str) -> bool:
    if not has_phrase(title, DATASET_WORDS):
        return False
    if has_phrase(title, NON_CANONICAL_DATASET_TITLE_WORDS):
        return False
    explicit_creation_title = has_phrase(title, CREATION_VERBS) or bool(
        re.search(
            r"\b(?:new|novel|realistic|labeled|labelled|annotated|comprehensive|public)\b"
            r".{0,90}\b(?:dataset|data set|benchmark|corpus|testbed)\b",
            comparable(title),
        )
    )
    if detector_wording_before_dataset_wording(title) and not explicit_creation_title:
        return False
    return (
        explicit_creation_title
        or bool(
            re.search(
                r"\b(?:dataset|data set|benchmark|corpus|testbed)\b.{0,90}"
                r"\b(?:for|of|to support|for evaluating|for evaluation)\b",
                comparable(title),
            )
        )
    )


def title_has_generator_creation_wording(title: str) -> bool:
    title_cmp = comparable(title)
    new_generation_dataset = bool(
        re.search(r"\b(?:new|next)\s+generation\s+(?:dataset|data\s+set|benchmark|corpus)\b", title_cmp)
    )
    strong_generation_evidence = has_phrase(
        title,
        (
            "traffic generation",
            "network traffic generation",
            "trace generation",
            "traffic generator",
            "attack generation",
            "attack injection",
            "anomaly injection",
            "dataset creation toolkit",
            "anomaly modeling engine",
        ),
    )
    if new_generation_dataset and not strong_generation_evidence:
        return False
    if title_is_generic_detector_framework(title) and not strong_generation_evidence:
        return False
    return has_generator_creation_evidence(title) and (
        has_phrase(title, INTRODUCTION_WORDS)
        or bool(
            re.search(
                r"\b(?:framework|toolkit|engine|generator|traffic\s+generation|trace\s+generation|attack\s+generation|injection|modeling)\b",
                title_cmp,
            )
        )
    )


def has_generator_creation_evidence(text: str) -> bool:
    if has_phrase(text, STRONG_GENERATOR_WORDS):
        return True
    text_cmp = comparable(text)
    return bool(
        re.search(
            r"\b(?:dataset|traffic|trace|attack|anomaly|flow|flow level|malicious|benign)\b"
            r".{0,80}\b(?:framework|toolkit|engine|generator)\b",
            text_cmp,
        )
        or re.search(
            r"\b(?:framework|toolkit|engine|generator)\b"
            r".{0,80}\b(?:dataset|traffic|trace|attack|anomaly|flow|malicious|benign)\b",
            text_cmp,
        )
    )


def title_is_generic_detector_framework(title: str) -> bool:
    if has_phrase(
        title,
        (
            "intrusion detection framework",
            "detection framework",
            "anomaly detection framework",
            "deep learning framework",
            "machine learning framework",
            "federated learning",
            "classification framework",
            "detection model",
            "intrusion detection model",
        ),
    ):
        return True
    title_cmp = comparable(title)
    return bool(
        re.search(
            r"\b(?:framework|model|approach|method|algorithm)\b.{0,90}"
            r"\b(?:intrusion|anomaly|attack|malicious|detection|learning|classification|classifier)\b",
            title_cmp,
        )
        or re.search(
            r"\b(?:intrusion|anomaly|attack|malicious|detection|learning|classification|classifier)\b.{0,90}"
            r"\b(?:framework|model|approach|method|algorithm)\b",
            title_cmp,
        )
    )


def work_has_generator_scope(work: Work) -> bool:
    return has_generator_creation_evidence(work.title) and not title_is_generic_detector_framework(work.title)


def is_downstream_or_analysis_title(title: str) -> bool:
    title_cmp = comparable(title)
    if any(re.search(rf"\b{re.escape(term)}\b", title_cmp) for term in map(comparable, DOWNSTREAM_WORDS)):
        return True
    return has_phrase(title, NON_CANONICAL_DATASET_TITLE_WORDS)


def release_year_from_aliases(entity: DatasetEntity) -> int | None:
    years = entity_years_from_aliases(entity)
    if not years:
        return None
    return min(years)


def years_near_aliases(entity: DatasetEntity) -> list[tuple[int, str]]:
    hints: list[tuple[int, str]] = []
    aliases = sorted(entity.aliases, key=len, reverse=True)
    for mention in entity.mentions:
        context = normalize_text(mention.context)
        if not context:
            continue
        for alias in aliases:
            alias_norm = normalize_text(alias)
            matches = list(re.finditer(rf"(?<![A-Za-z0-9]){re.escape(alias_norm)}(?![A-Za-z0-9])", context, flags=re.I))
            if not matches:
                continue
            for alias_match in matches:
                window = context[max(0, alias_match.start() - 160) : alias_match.end() + 160]
                for match in re.finditer(r"\b((?:19|20)\d{2})\b", window):
                    year = int(match.group(1))
                    if MIN_YEAR_GUARD <= year <= MAX_REASONABLE_RELEASE_YEAR:
                        hints.append((year, f"{mention.source_kind}: year near alias/context"))
    return hints


def infer_release_year(entity: DatasetEntity, work: Work | None) -> tuple[int | str, str]:
    alias_years = sorted(entity_years_from_aliases(entity))
    if alias_years:
        return alias_years[0], "explicit year-like token in dataset name/alias"

    if work and work.year and aliases_in_text(entity, work.title) and (
        title_has_dataset_creation_wording(work.title) or title_has_generator_creation_wording(work.title)
    ):
        return work.year, "canonical title names and introduces the dataset"

    context_years = years_near_aliases(entity)
    if context_years:
        counts: dict[int, int] = {}
        for year, _source in context_years:
            counts[year] = counts.get(year, 0) + 1
        best_year = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return best_year, "year found near dataset alias in trusted mention context"

    if work:
        title_years = explicit_years_from_name(work.title)
        if title_years:
            return min(title_years), "explicit year-like token in canonical title"
        if work.year:
            return work.year, "fallback to canonical publication year"

    return "", ""


def mention_taxonomy_windows(entity: DatasetEntity, mention: DatasetMention, *, radius: int = 220) -> list[str]:
    context = comparable(mention.context)
    if not context:
        return []
    aliases = [mention.raw_name, *sorted(entity.aliases, key=len, reverse=True)]
    windows: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias_cmp = comparable(alias)
        if len(alias_cmp) < 3:
            continue
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias_cmp)}(?![a-z0-9])", context):
            window = context[max(0, match.start() - radius) : match.end() + radius]
            if window in seen:
                continue
            seen.add(window)
            windows.append(window)
    return windows


def mention_taxonomy_text(entity: DatasetEntity, mention: DatasetMention) -> str:
    pieces: list[str] = []
    if mention.source_title and (
        aliases_in_text(entity, mention.source_title, allow_low_precision=True)
        or title_has_dataset_creation_wording(mention.source_title)
        or title_has_generator_creation_wording(mention.source_title)
    ):
        pieces.append(mention.source_title)
    windows = mention_taxonomy_windows(entity, mention)
    if windows:
        pieces.extend(windows[:3])
    else:
        pieces.append(mention.context[:700])
    return " ".join(piece for piece in pieces if piece)


def taxonomy_documents(entity: DatasetEntity | None, work: Work | None) -> list[tuple[float, str, str]]:
    documents: list[tuple[float, str, str]] = []
    if entity:
        documents.append((5.0, "dataset name/aliases", f"{entity.display_name} {' '.join(sorted(entity.aliases))}"))
    if work and work.title:
        documents.append((6.0, "canonical title", work.title))
    if work and work.abstract:
        documents.append((2.0, "canonical abstract", work.abstract[:1800]))
    if entity:
        for mention in entity.mentions[:8]:
            documents.append((1.0, mention.source_kind, f"{mention.raw_name} {mention_taxonomy_text(entity, mention)}"))
    return [(weight, source, normalize_text(text)) for weight, source, text in documents if normalize_text(text)]


def weighted_phrase_score(
    documents: list[tuple[float, str, str]],
    phrases: Iterable[str],
) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    for weight, source, text in documents:
        matched = [phrase for phrase in phrases if has_phrase(text, (phrase,))]
        if not matched:
            continue
        score += weight
        evidence.append(f"{source}: {', '.join(sorted(set(matched))[:3])}")
    return score, evidence


def taxonomy_decision_from_rules(
    documents: list[tuple[float, str, str]],
    rules: Iterable[tuple[str, tuple[str, ...]]],
    *,
    default: str = "Unknown",
    min_score: float = 1.0,
) -> TaxonomyDecision:
    best = TaxonomyDecision(default, "", 0.0)
    for label, phrases in rules:
        score, evidence = weighted_phrase_score(documents, phrases)
        if score > best.score:
            best = TaxonomyDecision(label, " | ".join(evidence[:4]), score)
    if best.score < min_score:
        return TaxonomyDecision(default, "", 0.0)
    return best


def taxonomy_text(entity: DatasetEntity | None, work: Work | None) -> str:
    return " ".join(text for _weight, _source, text in taxonomy_documents(entity, work))


def artifact_type_decision(entity: DatasetEntity | None, work: Work | None) -> TaxonomyDecision:
    documents = taxonomy_documents(entity, work)
    title = work.title if work else ""
    title_text = comparable(title)
    all_text = taxonomy_text(entity, work)
    title_has_dataset = has_phrase(title, DATASET_WORDS)
    title_has_generator = title_has_generator_creation_wording(title)
    strong_generator_score, generator_evidence = weighted_phrase_score(documents, STRONG_GENERATOR_WORDS)
    dataset_score, dataset_evidence = weighted_phrase_score(documents, DATASET_WORDS)

    generator_is_primary = (
        title_has_generator
        or bool(
            re.search(
                r"\b(?:dataset|traffic|attack|anomaly|intrusion detection)\b.{0,70}\b(?:toolkit|generator|engine|traffic\s+generation|trace\s+generation|attack\s+generation)\b",
                title_text,
            )
        )
        or bool(
            re.search(
                r"\b(?:toolkit|generator|engine|traffic\s+generation|trace\s+generation|attack\s+generation)\b.{0,70}\b(?:dataset|traffic|attack|anomaly|intrusion detection)\b",
                title_text,
            )
        )
    )
    if generator_is_primary:
        return TaxonomyDecision(
            "Generator/framework",
            " | ".join(
                (["canonical title: generator/framework wording"] if title_has_generator else [])
                + generator_evidence[:3]
            ),
            max(6.0, strong_generator_score),
        )
    if title_has_dataset or has_phrase(all_text, ("dataset", "data set", "corpus", "benchmark")):
        return TaxonomyDecision("Dataset", " | ".join(dataset_evidence[:4]), max(1.0, dataset_score))
    if strong_generator_score >= 3:
        return TaxonomyDecision("Generator/framework", " | ".join(generator_evidence[:4]), strong_generator_score)
    return TaxonomyDecision("Dataset", "", 0.0)


def artifact_type_for_work(work: Work | None, entity: DatasetEntity | None = None) -> str:
    return artifact_type_decision(entity, work).label


def audit_artifact_type_for_work(work: Work) -> str:
    title = work.title
    has_dataset = has_phrase(title, DATASET_WORDS)
    has_generator = has_phrase(title, STRONG_GENERATOR_WORDS) and not title_is_generic_detector_framework(title)
    if has_dataset and has_generator:
        return "Dataset + generator/framework"
    if has_generator:
        return "Generator/framework"
    return "Dataset"


def score_canonical_candidate(entity: DatasetEntity, work: Work) -> CandidateScore:
    title = work.title
    generator_scope = work_has_generator_scope(work)
    title_alias = bool(aliases_in_text(entity, title, allow_low_precision=generator_scope))
    primary_title_alias = bool(aliases_with_primary_dataset_evidence(entity, title) or title_starts_with_alias(entity, title))
    creation_title = title_has_dataset_creation_wording(title) or title_has_generator_creation_wording(title)
    abstract_aliases = abstract_alias_evidence(entity, work)
    abstract_supports_alias = bool(abstract_aliases) and creation_title and has_phrase(title, SECURITY_WORDS + DATASET_WORDS + GENERATOR_WORDS)
    release_year_hint = release_year_from_aliases(entity)
    reference_link_supports_alias = (
        work.relation == "survey_reference_link"
        and creation_title
        and has_phrase(title, SECURITY_WORDS + DATASET_WORDS + GENERATOR_WORDS)
        and (not release_year_hint or not work.year or abs(work.year - release_year_hint) <= 1)
    )
    alias_present = title_alias or abstract_supports_alias or reference_link_supports_alias
    reasons: list[str] = []

    if not title:
        return CandidateScore(work, -10, "rejected", "missing title")
    if has_phrase(title, SURVEY_WORDS):
        return CandidateScore(work, -8, "rejected", "survey/review title")
    if detector_wording_before_dataset_wording(title) and not generator_scope and not creation_title:
        return CandidateScore(work, -3, "rejected", "detector/model title before dataset")
    if not has_phrase(title, SECURITY_WORDS):
        return CandidateScore(work, -5, "rejected", "out of IDS/security scope")
    if not alias_present:
        return CandidateScore(work, -4, "rejected", "dataset alias absent from title and supported abstract")
    if title_alias and creation_title and not primary_title_alias and not title_has_generator_creation_wording(title):
        return CandidateScore(work, -4, "rejected", "alias appears as component/context, not dataset name")

    if release_year_hint and work.year and work.year > release_year_hint + 3 and not (
        title_starts_with_alias(entity, title) and creation_title
    ):
        return CandidateScore(
            work,
            -4,
            "rejected",
            f"candidate year {work.year} too late for release-year hint {release_year_hint}",
        )

    if title_alias and has_phrase(title, NON_CANONICAL_DATASET_TITLE_WORDS) and not creation_title:
        return CandidateScore(work, -3, "rejected", "dataset analysis/quality paper title")

    if is_downstream_or_analysis_title(title):
        if not title_starts_with_alias(entity, title):
            return CandidateScore(work, -3, "rejected", "downstream/model/evaluation title does not start with dataset alias")
        if not (title_starts_with_alias(entity, title) and creation_title):
            return CandidateScore(work, -3, "rejected", "downstream/model/evaluation wording in title")

    score = 0.0
    if title_alias:
        if primary_title_alias:
            score += 5.0
            reasons.append("alias in title as dataset name")
        else:
            score += 1.0
            reasons.append("alias in title context")
    elif abstract_supports_alias:
        score += 3.0
        reasons.append("alias in abstract with dataset/base-title support")
    elif reference_link_supports_alias:
        score += 4.0
        reasons.append("trusted survey citation links alias to base publication")
    if title_starts_with_alias(entity, title):
        score += 3.0
        reasons.append("title starts with alias")
    if title_has_dataset_creation_wording(title):
        score += 4.0
        reasons.append("dataset-creation title")
    if title_has_generator_creation_wording(title):
        score += 4.0
        reasons.append("generator/framework title")
    if work.doi:
        score += 0.25
        reasons.append("has DOI")
    if work.citations:
        score += min(1.5, math.log(work.citations + 1, 10) * 0.75)
        reasons.append("citation signal")
    if trusted_survey_count(entity) >= 2:
        score += 1.0
        reasons.append("multiple surveys mention dataset")
    if work.year:
        score += 0.1
    if release_year_hint and work.year:
        delta = abs(work.year - release_year_hint)
        if delta == 0:
            score += 2.0
            reasons.append("canonical year matches release-year hint")
        elif delta <= 1:
            score += 1.0
            reasons.append("canonical year near release-year hint")
        elif work.year > release_year_hint + 1:
            penalty = min(3.0, (work.year - release_year_hint) / 5)
            score -= penalty
            reasons.append(f"late relative to release-year hint (-{penalty:.1f})")

    if score >= 6:
        status = "canonical_resolved"
    elif trusted_survey_count(entity) >= 2:
        status = "survey_verified"
    else:
        status = "low_confidence"
    return CandidateScore(work=work, score=round(score, 4), status=status, reason=", ".join(reasons))


def dataset_name_evidence_score(work: Work, name: str) -> float:
    title = normalize_text(work.title)
    score = 0.0
    if title_starts_with_alias(DatasetEntity(dataset_entity_key(name), aliases={name}), title):
        score += 12.0
    if alias_is_primary_dataset_name(name, title):
        score += 10.0
    if name in extract_dataset_names(title, allow_low_precision=True):
        score += 4.0
    context = short_context(title, name, width=260)
    if has_phrase(context, DATASET_WORDS) and has_phrase(context, SECURITY_WORDS + GENERATOR_WORDS):
        score += 2.0
    if explicit_years_from_name(name):
        score += 1.5
    elif has_strong_dataset_name_shape(name):
        score += 0.75
    if name_looks_like_component_in_text(name, title) and not alias_is_primary_dataset_name(name, title):
        score -= 5.0
    return score


def dataset_names_from_work(work: Work) -> set[str]:
    title = normalize_text(work.title)
    names = set(extract_dataset_names(title, allow_low_precision=True))
    for match in re.findall(NAME_EXPR, title):
        candidate = canonical_dataset_name(match if isinstance(match, str) else match[0])
        if not looks_like_dataset_name(candidate, allow_low_precision=True):
            continue
        context = short_context(title, candidate, width=260)
        if explicit_years_from_name(candidate) or (
            has_phrase(context, DATASET_WORDS)
            and has_phrase(context, SECURITY_WORDS + GENERATOR_WORDS)
        ):
            names.add(candidate)

    if has_phrase(title, DATASET_WORDS) or title_has_dataset_creation_wording(title) or title_has_generator_creation_wording(title):
        title_patterns = [
            r"^\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,})\s*(?::|,|\()",
            rf"^\s*({NAME_EXPR})(?:\s+[A-Z][A-Za-z0-9/-]+){{0,4}}\s+(?:{DATASET_EXPR})\b",
            rf"^\s*({NAME_EXPR})\s*(?::|,|\()",
            rf"\b({NAME_EXPR})\b\s*(?::|-)\s*(?i:a|an|the|new|novel|public|realistic|labeled|labelled|annotated)?\s*(?:{DATASET_EXPR})\b",
            rf"\b(?:{DATASET_EXPR})\s*(?::|-)\s*({NAME_EXPR})\b",
            rf"\b(?i:named|called)\s+(?:the\s+)?({NAME_EXPR})\b",
        ]
        for pattern in title_patterns:
            for match in re.findall(pattern, title, flags=re.M):
                candidate = canonical_dataset_name(match if isinstance(match, str) else match[0])
                if looks_like_dataset_name(candidate, allow_low_precision=True):
                    names.add(candidate)

    names = {name for name in names if looks_like_dataset_name(name, allow_low_precision=True)}
    if len(names) <= 1:
        return names

    scored = {name: dataset_name_evidence_score(work, name) for name in names}
    strongest = max(scored.values())
    if strongest >= 8.0:
        return {name for name, score in scored.items() if score >= strongest - 1.0 and score >= 8.0}
    return {name for name, score in scored.items() if score >= 2.0}


def score_dataset_source_work(work: Work) -> tuple[int, str]:
    title = work.title
    names = dataset_names_from_work(work)
    generator_scope = work_has_generator_scope(work)
    reasons: list[str] = []
    score = 0

    if not title:
        return -10, "missing title"
    if has_phrase(title, SURVEY_WORDS):
        return -8, "survey/review title"
    if has_phrase(title, NON_CANONICAL_DATASET_TITLE_WORDS):
        return -4, "dataset analysis/quality paper title"
    if detector_wording_before_dataset_wording(title) and not generator_scope and not title_has_dataset_creation_wording(title):
        return -4, "detector/model title before dataset"
    if is_downstream_or_analysis_title(title):
        if not title_has_dataset_creation_wording(title) and not generator_scope:
            return -4, "downstream/model/evaluation wording in title"

    if names:
        score += min(6, 2 + len(names))
        reasons.append("dataset-like name")
    else:
        score -= 4
    if has_phrase(title, DATASET_WORDS):
        score += 4
        reasons.append("dataset word in title")
    if title_has_dataset_creation_wording(title):
        score += 5
        reasons.append("dataset-creation title")
    if title_has_generator_creation_wording(title):
        score += 5
        reasons.append("generator/framework title")
    if has_phrase(title, SECURITY_WORDS):
        score += 4
        reasons.append("security/traffic scope")
    else:
        score -= 4
    if work.doi:
        score += 1
    if work.citations >= 25:
        score += 1
    if work.citations >= 100:
        score += 1

    return score, ", ".join(reasons) or "weak dataset-publication evidence"


def add_entities_from_source_work(
    entities: dict[str, DatasetEntity],
    work: Work,
    source_kind: str,
    *,
    min_score: int,
    min_citations: int,
) -> int:
    source_score, source_reason = score_dataset_source_work(work)
    if source_score < min_score:
        return 0

    added = 0
    context = work.title
    for name in sorted(dataset_names_from_work(work), key=lambda item: item.casefold()):
        entity = add_mention(
            entities,
            name,
            work,
            source_kind,
            f"{source_reason}: {short_context(context, name, width=500)}",
            allow_low_precision=True,
        )
        if not entity:
            continue
        before_candidates = len(entity.candidates)
        entity.candidates[work.key] = work
        score = score_canonical_candidate(entity, work)
        canonical_allowed = meets_citation_floor(work, min_citations) or trusted_survey_count(entity) >= 2
        if canonical_allowed and score.score > 0 and (entity.canonical is None or score.score > entity.canonical.score):
            entity.canonical = score
        elif score.score <= 0:
            entity.rejected_candidates.append(score)
        if len(entity.candidates) > before_candidates:
            added += 1
    return added


def discover_from_dataset_publication_search(
    session: requests.Session,
    args: argparse.Namespace,
    entities: dict[str, DatasetEntity],
    *,
    completed_queries: set[str] | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> int:
    added = 0
    seen: dict[str, Work] = {}
    queries = args.dataset_query or DEFAULT_DATASET_PUBLICATION_QUERIES
    remaining_queries = [query for query in queries if completed_queries is None or query not in completed_queries]
    print(
        "Direct dataset-publication search fanout: "
        f"{len(remaining_queries)} query/queries x {args.dataset_search_pages} page(s) "
        f"= up to {len(remaining_queries) * args.dataset_search_pages} OpenAlex search request(s).",
        file=sys.stderr,
    )
    for query in queries:
        if completed_queries is not None and query in completed_queries:
            continue
        print(f"Dataset publication search: {query}", file=sys.stderr)
        works = openalex_search(
            session,
            args,
            query,
            per_page=args.dataset_search_per_page,
            pages=args.dataset_search_pages,
            relation="dataset_publication_search",
            sort="relevance_score:desc",
        )
        for work in works:
            if work.key in seen:
                continue
            seen[work.key] = work
            added += add_entities_from_source_work(
                entities,
                work,
                "dataset_publication_search",
                min_score=args.min_dataset_source_score,
                min_citations=args.min_dataset_citations,
            )
        if completed_queries is not None:
            completed_queries.add(query)
        if checkpoint_callback:
            checkpoint_callback()
    return added


def discover_from_broad_dataset_search(
    session: requests.Session,
    args: argparse.Namespace,
    entities: dict[str, DatasetEntity],
    *,
    completed_queries: set[str] | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> int:
    queries = args.broad_dataset_query or DEFAULT_BROAD_DATASET_QUERIES
    remaining_queries = [query for query in queries if completed_queries is None or query not in completed_queries]
    print(
        "Broad dataset search fanout: "
        f"{len(remaining_queries)} query/queries x {args.dataset_search_pages} page(s) "
        f"= up to {len(remaining_queries) * args.dataset_search_pages} OpenAlex search request(s).",
        file=sys.stderr,
    )

    added = 0
    seen: dict[str, Work] = {}
    for query in queries:
        checkpoint_key = f"broad:{query}"
        if completed_queries is not None and checkpoint_key in completed_queries:
            continue
        print(f"Broad dataset search: {query}", file=sys.stderr)
        works = openalex_search(
            session,
            args,
            query,
            per_page=args.dataset_search_per_page,
            pages=args.dataset_search_pages,
            relation="broad_dataset_search",
            sort="relevance_score:desc",
        )
        for work in works:
            if work.key in seen:
                continue
            seen[work.key] = work
            added += add_entities_from_source_work(
                entities,
                work,
                "broad_dataset_search",
                min_score=args.min_broad_dataset_source_score,
                min_citations=args.min_dataset_citations,
            )
        if completed_queries is not None:
            completed_queries.add(checkpoint_key)
        if checkpoint_callback:
            checkpoint_callback()
    return added


def discover_from_trusted_references(
    session: requests.Session,
    args: argparse.Namespace,
    trusted: list[Work],
    entities: dict[str, DatasetEntity],
    *,
    completed_references: set[str] | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> int:
    added = 0
    cache: dict[str, Work] = {}
    reference_ids: list[str] = []
    for survey in trusted:
        for reference in survey.referenced_works:
            if reference not in reference_ids:
                reference_ids.append(reference)

    selected_references = reference_ids[: args.max_reference_works]
    remaining_references = [
        reference
        for reference in selected_references
        if completed_references is None or reference not in completed_references
    ]
    print(
        "Trusted-reference snowball fanout: "
        f"{len(remaining_references)} remaining work request(s) "
        f"out of {len(selected_references)} selected reference(s).",
        file=sys.stderr,
    )

    for index, reference in enumerate(selected_references, start=1):
        if completed_references is not None and reference in completed_references:
            continue
        try:
            work = fetch_openalex_work(
                session,
                args,
                reference,
                relation="trusted_reference",
                cache=cache,
            )
        except OpenAlexRateLimitError:
            raise
        except Exception as exc:
            print(f"  Reference skipped ({reference}): {exc}", file=sys.stderr)
            continue
        if not work:
            if completed_references is not None:
                completed_references.add(reference)
            if checkpoint_callback:
                checkpoint_callback()
            continue
        if index % 25 == 0:
            print(f"  Checked {index} trusted references", file=sys.stderr)
        added += add_entities_from_source_work(
            entities,
            work,
            "trusted_reference",
            min_score=args.min_dataset_source_score,
            min_citations=args.min_dataset_citations,
        )
        if completed_references is not None:
            completed_references.add(reference)
        if checkpoint_callback:
            checkpoint_callback()
    return added


def title_match_score(left: str, right: str) -> float:
    left_cmp = comparable(left)
    right_cmp = comparable(right)
    if not left_cmp or not right_cmp:
        return 0.0
    if left_cmp == right_cmp:
        return 1.0
    if left_cmp in right_cmp or right_cmp in left_cmp:
        return 0.95
    return difflib.SequenceMatcher(None, left_cmp, right_cmp).ratio()


def best_reference_title_match(title: str, works: list[Work], min_source_score: int) -> Work | None:
    matches: list[tuple[float, int, Work]] = []
    for work in works:
        similarity = title_match_score(title, work.title)
        if similarity < 0.88:
            continue
        source_score, _source_reason = score_dataset_source_work(work)
        if source_score < min_source_score:
            continue
        matches.append((similarity, work.citations, work))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], -item[1], item[2].year or 9999))
    return matches[0][2]


def discover_from_survey_reference_links(
    session: requests.Session,
    args: argparse.Namespace,
    trusted: list[Work],
    entities: dict[str, DatasetEntity],
    *,
    checkpoint_callback: Callable[[], None] | None = None,
) -> int:
    links: list[SurveyReferenceLink] = []
    for survey in trusted:
        links.extend(survey.survey_reference_links)
    unique_links = list({(dataset_entity_key(link.dataset_name), comparable(link.reference_title)): link for link in links}.values())
    print(
        "Survey reference-title fanout: "
        f"{len(unique_links)} citation-linked title(s) x 1 page = up to {len(unique_links)} OpenAlex search request(s).",
        file=sys.stderr,
    )

    added = 0
    cache: dict[str, Work] = {}
    for link in unique_links:
        entity = add_mention(
            entities,
            link.dataset_name,
            Work(
                openalex_id=link.source_work_id,
                title=link.source_title,
                relation="survey_reference_link",
            ),
            "survey_reference_link",
            link.context,
            allow_low_precision=True,
            allow_plain_name=True,
        )
        if not entity:
            continue
        works = openalex_search(
            session,
            args,
            link.reference_title,
            per_page=10,
            pages=1,
            relation="survey_reference_link",
            sort="relevance_score:desc",
        )
        for work in works:
            cache[work.key] = work
        work = best_reference_title_match(link.reference_title, works, args.min_dataset_source_score)
        if not work:
            continue
        before_candidates = len(entity.candidates)
        entity.candidates[work.key] = work
        score = score_canonical_candidate(entity, work)
        canonical_allowed = meets_citation_floor(work, args.min_dataset_citations) or trusted_survey_count(entity) >= 2
        if canonical_allowed and score.score > 0 and (entity.canonical is None or score.score > entity.canonical.score):
            entity.canonical = score
        elif score.score <= 0:
            entity.rejected_candidates.append(score)
        if len(entity.candidates) > before_candidates:
            added += 1
        if checkpoint_callback:
            checkpoint_callback()
    return added


def candidate_aliases(entity: DatasetEntity, args: argparse.Namespace) -> list[str]:
    aliases = [
        alias
        for alias in sorted(entity.aliases, key=lambda item: (len(item), item.casefold()))
        if looks_like_dataset_name(alias, allow_low_precision=True, allow_plain_name=True)
    ]
    return aliases[: max(1, args.max_aliases_per_entity)]


def candidate_queries(entity: DatasetEntity, args: argparse.Namespace) -> list[str]:
    templates = (
        CANDIDATE_QUERY_TEMPLATES_FULL
        if args.candidate_query_profile == "full"
        else CANDIDATE_QUERY_TEMPLATES_COMPACT
    )
    queries: list[str] = []
    for alias in candidate_aliases(entity, args):
        queries.extend(template.format(alias=alias) for template in templates)
    unique_queries = list(dict.fromkeys(queries))
    if args.max_candidate_queries_per_entity:
        unique_queries = unique_queries[: args.max_candidate_queries_per_entity]
    return unique_queries


def resolve_entities(
    session: requests.Session,
    args: argparse.Namespace,
    entities: dict[str, DatasetEntity],
    *,
    resolved_entity_keys: set[str] | None = None,
    checkpoint_callback: Callable[[], None] | None = None,
) -> None:
    entity_filters = [value for value in (args.resolve_entity or []) if value.strip()]
    pending_entities = [
        entity
        for entity in sorted(entities.values(), key=lambda item: item.display_name.casefold())
        if (not entity_filters or any(entity_matches_text(entity, value) for value in entity_filters))
        if entity.survey_count >= args.min_survey_mentions
        and entity_has_plausible_dataset_alias(entity)
        and not (resolved_entity_keys is not None and entity.key in resolved_entity_keys and not entity_filters)
        and not (
            entity.canonical
            and entity.canonical.status == "canonical_resolved"
            and not args.expand_resolved_entities
            and not entity_filters
        )
    ]
    estimated_requests = sum(len(candidate_queries(entity, args)) * args.candidate_pages for entity in pending_entities)
    print(
        "Candidate resolution fanout: "
        f"{len(pending_entities)} pending entities, profile={args.candidate_query_profile}, "
        f"max_aliases={args.max_aliases_per_entity}, "
        f"query_cap={args.max_candidate_queries_per_entity or 'none'}, "
        f"pages_per_query={args.candidate_pages}, "
        f"up to {estimated_requests} OpenAlex search request(s).",
        file=sys.stderr,
    )

    for index, entity in enumerate(sorted(entities.values(), key=lambda item: item.display_name.casefold()), start=1):
        if entity_filters and not any(entity_matches_text(entity, value) for value in entity_filters):
            continue
        if entity.survey_count < args.min_survey_mentions:
            continue
        if not entity_has_plausible_dataset_alias(entity):
            if resolved_entity_keys is not None:
                resolved_entity_keys.add(entity.key)
            if checkpoint_callback:
                checkpoint_callback()
            continue
        if resolved_entity_keys is not None and entity.key in resolved_entity_keys and not entity_filters:
            continue
        if entity.canonical and entity.canonical.status == "canonical_resolved" and not args.expand_resolved_entities and not entity_filters:
            if resolved_entity_keys is not None:
                resolved_entity_keys.add(entity.key)
            if checkpoint_callback:
                checkpoint_callback()
            continue
        queries = candidate_queries(entity, args)
        aliases = candidate_aliases(entity, args)
        print(
            f"[{index}/{len(entities)}] Resolving {entity.display_name}: "
            f"{len(queries)} candidate queries x {args.candidate_pages} page(s) "
            f"= up to {len(queries) * args.candidate_pages} OpenAlex request(s); "
            f"aliases={', '.join(aliases)}",
            file=sys.stderr,
        )
        for query in queries:
            works = openalex_search(
                session,
                args,
                query,
                per_page=args.candidate_per_page,
                pages=args.candidate_pages,
                relation="candidate_resolution",
                sort="relevance_score:desc",
            )
            for work in works:
                entity.candidates[work.key] = work

        scored: list[CandidateScore] = []
        for work in entity.candidates.values():
            score = score_canonical_candidate(entity, work)
            canonical_allowed = meets_citation_floor(work, args.min_dataset_citations) or trusted_survey_count(entity) >= 2
            if canonical_allowed and score.score > 0:
                scored.append(score)
            else:
                entity.rejected_candidates.append(score)
        scored.sort(key=lambda item: (-item.score, -item.work.citations, item.work.year or 9999))
        if scored:
            entity.canonical = scored[0]
        if resolved_entity_keys is not None:
            resolved_entity_keys.add(entity.key)
        if checkpoint_callback:
            checkpoint_callback()


def rescore_entity_canonicals(entities: dict[str, DatasetEntity], *, min_citations: int = 0) -> None:
    for entity in entities.values():
        scored: list[CandidateScore] = []
        rejected_by_key: dict[tuple[str, str], CandidateScore] = {
            (candidate.work.key, candidate.reason): candidate for candidate in entity.rejected_candidates
        }
        for work in entity.candidates.values():
            score = score_canonical_candidate(entity, work)
            canonical_allowed = meets_citation_floor(work, min_citations) or trusted_survey_count(entity) >= 2
            if canonical_allowed and score.score > 0:
                scored.append(score)
            else:
                rejected_by_key[(work.key, score.reason)] = score
        scored.sort(key=lambda item: (-item.score, -item.work.citations, item.work.year or 9999))
        entity.canonical = scored[0] if scored else None
        entity.rejected_candidates = list(rejected_by_key.values())


def normalize_single_source_canonical_entity_keys(entities: dict[str, DatasetEntity]) -> int:
    renamed = 0
    for key, entity in list(entities.items()):
        if not entity.canonical or entity.canonical.status != "canonical_resolved":
            continue
        names = dataset_names_from_work(entity.canonical.work)
        if len(names) != 1:
            continue
        canonical_name = next(iter(names))
        canonical_key = dataset_entity_key(canonical_name)
        if not canonical_key or canonical_key == key:
            entity.add_alias(canonical_name)
            continue

        entity.aliases = {canonical_dataset_name(canonical_name)}
        if canonical_key in entities:
            entities[canonical_key].merge_from(entity)
            del entities[key]
        else:
            del entities[key]
            entity.key = canonical_key
            entities[canonical_key] = entity
        renamed += 1
    return renamed


def known_candidate_works(entities: dict[str, DatasetEntity]) -> list[Work]:
    works: dict[str, Work] = {}
    for entity in entities.values():
        for work in entity.candidates.values():
            works.setdefault(work.key, work)
        for score in entity.rejected_candidates:
            works.setdefault(score.work.key, score.work)
    return list(works.values())


def reconstruct_high_recall_entities(
    entities: dict[str, DatasetEntity],
    *,
    min_citations: int,
) -> int:
    added = 0
    for work in known_candidate_works(entities):
        text = normalize_text(f"{work.title} {work.abstract}")
        if not has_phrase(text, DATASET_WORDS):
            continue
        if not has_phrase(text, SECURITY_WORDS + GENERATOR_WORDS):
            continue
        names = high_recall_dataset_name_patterns(text[:8000])
        for name in sorted(names, key=lambda item: item.casefold()):
            context = short_context(text, name, width=520)
            if not has_phrase(context, DATASET_WORDS) or not has_phrase(context, SECURITY_WORDS + GENERATOR_WORDS):
                continue
            existing = entities.get(dataset_entity_key(name))
            if existing and any(
                mention.raw_name == canonical_dataset_name(name)
                and mention.source_work_id == work.openalex_id
                and mention.source_kind == "high_recall_work_text"
                for mention in existing.mentions
            ):
                continue
            entity = add_mention(
                entities,
                name,
                work,
                "high_recall_work_text",
                f"generic dataset-name pattern in known work: {context}",
                allow_low_precision=True,
                allow_plain_name=True,
            )
            if not entity:
                continue
            before_candidates = len(entity.candidates)
            entity.candidates[work.key] = work
            score = score_canonical_candidate(entity, work)
            canonical_allowed = meets_citation_floor(work, min_citations) or trusted_survey_count(entity) >= 2
            if canonical_allowed and score.score > 0 and (entity.canonical is None or score.score > entity.canonical.score):
                entity.canonical = score
            elif score.score <= 0:
                entity.rejected_candidates.append(score)
            if len(entity.candidates) > before_candidates:
                added += 1
    return added


def entity_matches_text(entity: DatasetEntity, text: str) -> bool:
    needle = compact_key(text)
    if not needle:
        return False
    values = [entity.key, entity.display_name, *entity.aliases]
    return any(needle in compact_key(value) or compact_key(value) in needle for value in values if compact_key(value))


def refresh_missing_candidate_abstracts(
    session: requests.Session,
    args: argparse.Namespace,
    entities: dict[str, DatasetEntity],
) -> int:
    if not args.refresh_missing_abstracts:
        return 0
    filters = [value for value in (args.refresh_abstract_entity or []) if value.strip()]
    selected = [
        entity
        for entity in sorted(entities.values(), key=lambda item: item.display_name.casefold())
        if not filters or any(entity_matches_text(entity, value) for value in filters)
    ]
    refreshed = 0
    cache: dict[str, Work] = {}
    for entity in selected:
        for key, work in list(entity.candidates.items()):
            if args.refresh_abstract_limit and refreshed >= args.refresh_abstract_limit:
                return refreshed
            if work.abstract or not work.openalex_id:
                continue
            try:
                updated = fetch_openalex_work(session, args, work.openalex_id, relation=work.relation, cache=cache)
            except OpenAlexRateLimitError:
                raise
            except Exception as exc:
                print(f"  Abstract refresh skipped for {work.openalex_id}: {exc}", file=sys.stderr)
                continue
            if not updated:
                continue
            entity.candidates[key] = updated
            refreshed += 1
    return refreshed


def domain_decision(entity: DatasetEntity, work: Work | None) -> TaxonomyDecision:
    documents = taxonomy_documents(entity, work)
    scores: dict[str, tuple[float, list[str]]] = {
        label: weighted_phrase_score(documents, phrases) for label, phrases in DOMAIN_RULES
    }
    if scores.get("IoT", (0.0, []))[0] > 0 and scores.get("IIoT", (0.0, []))[0] > 0:
        iot_score, iot_evidence = scores["IoT"]
        iiot_score, iiot_evidence = scores["IIoT"]
        return TaxonomyDecision(
            "IoT/IIoT",
            " | ".join((iot_evidence + iiot_evidence)[:4]),
            iot_score + iiot_score,
        )

    positive = {label: value for label, value in scores.items() if value[0] >= 1.0}
    if not positive:
        return TaxonomyDecision("General Network", "", 0.0)

    priority = {
        "DNS": 8,
        "Automotive/CAN": 8,
        "ICS/SCADA": 8,
        "Host": 7,
        "Wireless": 7,
        "IIoT": 7,
        "IoT": 7,
        "Encrypted/VPN/Tor": 7,
        "Web/App": 6,
        "Malware/Botnet": 6,
        "General Network": 4,
    }
    ordered = sorted(
        positive.items(),
        key=lambda item: (item[1][0], priority.get(item[0], 0)),
        reverse=True,
    )
    best_label, (best_score, best_evidence) = ordered[0]
    general_score, general_evidence = scores.get("General Network", (0.0, []))
    if (
        best_label in {"Web/App", "Malware/Botnet"}
        and best_score <= 1.0
        and general_score >= best_score
    ):
        best_label, best_score, best_evidence = "General Network", general_score, general_evidence
    if (
        best_label == "Host"
        and general_score > 0
        and "canonical " not in " ".join(best_evidence).casefold()
    ):
        best_label, best_score, best_evidence = "General Network", general_score, general_evidence
    if best_label in {"IoT", "IIoT"}:
        combined_evidence = scores.get("IoT", (0.0, []))[1] + scores.get("IIoT", (0.0, []))[1]
        return TaxonomyDecision("IoT/IIoT", " | ".join(combined_evidence[:4]), best_score)
    return TaxonomyDecision(best_label, " | ".join(best_evidence[:4]), best_score)


def infer_domain(entity: DatasetEntity, work: Work | None) -> str:
    return domain_decision(entity, work).label


def data_form_decision(entity: DatasetEntity, work: Work | None) -> TaxonomyDecision:
    documents = taxonomy_documents(entity, work)
    scores: dict[str, tuple[float, list[str]]] = {
        label: weighted_phrase_score(documents, phrases) for label, phrases in DATA_FORM_RULES
    }
    positive = {label: value for label, value in scores.items() if value[0] > 0}
    if not positive:
        return TaxonomyDecision("Unknown", "", 0.0)

    # Packet evidence is more specific than generic "flow" evidence; host/CAN/wireless evidence is domain-specific.
    priority = {
        "Host/System-call": 7,
        "CAN": 7,
        "Wireless": 6,
        "Packet": 5,
        "Flow": 4,
        "Telemetry": 4,
        "Log": 4,
        "Tabular features": 3,
        "Network traffic": 2,
    }
    ordered = sorted(
        positive.items(),
        key=lambda item: (item[1][0], priority.get(item[0], 0)),
        reverse=True,
    )
    best_label, (best_score, best_evidence) = ordered[0]
    if best_label == "Network traffic" and "Packet" in positive and positive["Packet"][0] >= best_score * 0.65:
        best_label, (best_score, best_evidence) = "Packet", positive["Packet"]
    if best_label == "Network traffic" and "Flow" in positive and positive["Flow"][0] >= best_score * 0.65:
        best_label, (best_score, best_evidence) = "Flow", positive["Flow"]
    return TaxonomyDecision(best_label, " | ".join(best_evidence[:4]), best_score)


def infer_data_form(entity: DatasetEntity, work: Work | None) -> str:
    return data_form_decision(entity, work).label


def environment_decision(entity: DatasetEntity, work: Work | None) -> TaxonomyDecision:
    documents = taxonomy_documents(entity, work)
    real_score, real_evidence = weighted_phrase_score(documents, REAL_WORLD_PHRASES)
    synthetic_score, synthetic_evidence = weighted_phrase_score(documents, SYNTHETIC_LAB_PHRASES)
    text = taxonomy_text(entity, work)
    hybrid_phrase = has_phrase(
        text,
        (
            "real and synthetic",
            "real traffic and synthetic",
            "synthetic attacks and benign traffic",
            "synthetic attacks into background traffic",
            "injected synthetic attacks",
            "real encrypted synthetic attack traffic",
        ),
    )
    if hybrid_phrase or (real_score >= 2.0 and synthetic_score >= 2.0):
        return TaxonomyDecision("Hybrid", "real-world and synthetic/lab evidence", real_score + synthetic_score)
    if synthetic_score > real_score and synthetic_score >= 1.0:
        return TaxonomyDecision("Synthetic/Lab", " | ".join(synthetic_evidence[:4]), synthetic_score)
    if real_score > synthetic_score and real_score >= 2.0:
        return TaxonomyDecision("Real-world", " | ".join(real_evidence[:4]), real_score)
    return TaxonomyDecision("Unknown", "", 0.0)


def infer_environment(entity: DatasetEntity, work: Work | None) -> str:
    return environment_decision(entity, work).label


def temporal_type_decision(entity: DatasetEntity, work: Work | None, data_form: str | None = None) -> TaxonomyDecision:
    documents = taxonomy_documents(entity, work)
    dynamic_phrases = (
        "dynamic",
        "time series",
        "temporal",
        "stream",
        "online",
        "real time",
        "real-time",
        "traffic traces",
        "packet traces",
        "captured traffic",
        "network traffic",
        "sensor telemetry",
    )
    static_phrases = (
        "static",
        "tabular",
        "features",
        "csv",
        "records",
        "feature vector",
        "data records",
    )
    dynamic_score, dynamic_evidence = weighted_phrase_score(documents, dynamic_phrases)
    static_score, static_evidence = weighted_phrase_score(documents, static_phrases)
    form = data_form or data_form_decision(entity, work).label
    dynamic_form = form in {"Packet", "Flow", "Network traffic", "CAN", "Wireless", "Telemetry", "Host/System-call", "Log"}
    static_form = form in {"Tabular features"}
    if dynamic_form:
        dynamic_score += 1.5
        dynamic_evidence.append(f"data form: {form}")
    if form in {"Tabular features", "Log"}:
        static_score += 1.5
        static_evidence.append(f"data form: {form}")
    dynamic_threshold = 1.0 if dynamic_form else 2.0
    static_threshold = 1.0 if static_form else 2.0
    if dynamic_score >= dynamic_threshold and static_score >= static_threshold:
        return TaxonomyDecision("Mixed", "dynamic and static evidence", dynamic_score + static_score)
    if dynamic_score >= dynamic_threshold:
        return TaxonomyDecision("Dynamic", " | ".join(dynamic_evidence[:4]), dynamic_score)
    if static_score >= static_threshold:
        return TaxonomyDecision("Static", " | ".join(static_evidence[:4]), static_score)
    return TaxonomyDecision("Unknown", "", 0.0)


def infer_temporal_type(entity: DatasetEntity, work: Work | None) -> str:
    return temporal_type_decision(entity, work).label


def labeling_decision(entity: DatasetEntity, work: Work | None) -> TaxonomyDecision:
    documents = taxonomy_documents(entity, work)
    unlabeled_score, unlabeled_evidence = weighted_phrase_score(
        documents,
        ("unlabeled", "unlabelled", "without labels", "without label"),
    )
    partial_score, partial_evidence = weighted_phrase_score(
        documents,
        ("partially labeled", "partially labelled", "partially annotated"),
    )
    labeled_score, labeled_evidence = weighted_phrase_score(documents, LABELED_PHRASES)
    if unlabeled_score > 0 and unlabeled_score >= labeled_score:
        return TaxonomyDecision("Unlabeled", " | ".join(unlabeled_evidence[:4]), unlabeled_score)
    if partial_score > 0:
        return TaxonomyDecision("Partially labeled", " | ".join(partial_evidence[:4]), partial_score)
    if labeled_score > 0:
        return TaxonomyDecision("Labeled", " | ".join(labeled_evidence[:4]), labeled_score)
    return TaxonomyDecision("Unknown", "", 0.0)


def infer_labeling(entity: DatasetEntity, work: Work | None) -> str:
    return labeling_decision(entity, work).label


def dataset_type_decisions(entity: DatasetEntity, work: Work | None) -> dict[str, TaxonomyDecision]:
    data_form = data_form_decision(entity, work)
    return {
        "artifact_type": artifact_type_decision(entity, work),
        "domain": domain_decision(entity, work),
        "data_form": data_form,
        "temporal_type": temporal_type_decision(entity, work, data_form.label),
        "environment": environment_decision(entity, work),
        "labeling": labeling_decision(entity, work),
    }


def confidence_for_entity(entity: DatasetEntity) -> str:
    trusted_mentions = trusted_survey_count(entity)
    if entity.canonical and entity.canonical.status == "canonical_resolved" and trusted_mentions >= 2:
        return "High"
    if entity.canonical and entity.canonical.status == "canonical_resolved":
        return "Medium"
    if trusted_mentions >= 2:
        return "Medium"
    return "Low"


def should_output_entity(entity: DatasetEntity, args: argparse.Namespace) -> bool:
    if args.include_low_confidence:
        return True

    confidence = confidence_for_entity(entity)
    if confidence not in {"High", "Medium"}:
        return False
    if entity.canonical and entity.canonical.status == "canonical_resolved":
        return True
    return trusted_survey_count(entity) >= args.min_survey_mentions


def readable_discovery_method(method: str) -> str:
    method = str(method or "").strip().casefold().replace("-", "_")
    labels = {
        "survey_metadata": "Survey metadata",
        "survey_pdf_table": "Survey PDF table",
        "survey_pdf_text": "Survey PDF text",
        "trusted_reference": "Trusted references",
        "survey_reference_link": "Survey reference links",
        "dataset_publication_search": "Dataset query",
        "broad_dataset_search": "Broad dataset query",
        "high_recall_work_text": "High-recall work text",
        "candidate_resolution": "Alias query",
        "survey_search": "Survey query",
        "trusted_input": "Trusted input",
        "trusted_title_search": "Trusted title search",
        "candidate_reconstruction": "Candidate audit",
    }
    return labels.get(method, method.replace("_", " ").title() if method else UNKNOWN)


def entity_discovery_methods(entity: DatasetEntity) -> list[str]:
    methods = {readable_discovery_method(mention.source_kind) for mention in entity.mentions}
    methods.update(readable_discovery_method(work.relation) for work in entity.candidates.values() if work.relation)
    return sorted(method for method in methods if method and method != UNKNOWN)


def primary_discovery_method(entity: DatasetEntity, work: Work | None) -> str:
    if work and work.relation:
        return readable_discovery_method(work.relation)
    methods = entity_discovery_methods(entity)
    priority = {
        "Dataset query": 0,
        "Broad dataset query": 1,
        "Trusted references": 2,
        "Survey reference links": 3,
        "High-recall work text": 4,
        "Alias query": 5,
        "Survey PDF table": 6,
        "Survey PDF text": 7,
        "Survey metadata": 8,
        "Trusted input": 9,
        "Survey query": 10,
    }
    return sorted(methods, key=lambda method: (priority.get(method, 99), method))[0] if methods else UNKNOWN


def entity_description(entity: DatasetEntity, work: Work | None) -> str:
    contexts = [
        normalize_text(mention.context)
        for mention in entity.mentions
        if normalize_text(mention.context)
    ]
    if not contexts:
        return "Survey-derived entity without title-level context."
    contexts.sort(key=lambda text: (-int(has_phrase(text, DATASET_WORDS + GENERATOR_WORDS)), -len(text)))
    return contexts[0][:420]


def entity_to_row(entity: DatasetEntity) -> dict[str, str | int | float]:
    canonical = entity.canonical
    work = canonical.work if canonical else None
    status = canonical.status if canonical else ("survey_verified" if trusted_survey_count(entity) >= 2 else "low_confidence")
    evidence = canonical.reason if canonical else "mentioned in trusted survey source(s)"
    release_year, release_year_source = infer_release_year(entity, work)
    canonical_publication_year = work.year if work and work.year else ""
    type_decisions = dataset_type_decisions(entity, work)
    type_evidence = "; ".join(
        f"{field}={decision.label}"
        + (f" ({decision.evidence})" if decision.evidence else "")
        for field, decision in type_decisions.items()
    )
    return {
        "dataset_id": entity.key,
        "name": entity.display_name,
        "aliases": "; ".join(sorted(entity.aliases)),
        "artifact_type": type_decisions["artifact_type"].label,
        "status": status,
        "canonical_openalex_id": work.openalex_id if work else "",
        "canonical_title": work.title if work else "",
        "description": entity_description(entity, work),
        "release_year": release_year,
        "canonical_publication_year": canonical_publication_year,
        "year": release_year,
        "release_year_source": release_year_source,
        "doi": work.doi if work else "",
        "url": work.url if work else "",
        "citations": work.citations if work else "",
        "survey_count": entity.survey_count,
        "trusted_survey_count": trusted_survey_count(entity),
        "discovery_method": primary_discovery_method(entity, work),
        "discovery_methods": "; ".join(entity_discovery_methods(entity)),
        "survey_sources": "; ".join(sorted({mention.source_title for mention in entity.mentions})),
        "mention_sources": "; ".join(sorted({mention.source_kind for mention in entity.mentions})),
        "domain": type_decisions["domain"].label,
        "data_form": type_decisions["data_form"].label,
        "temporal_type": type_decisions["temporal_type"].label,
        "environment": type_decisions["environment"].label,
        "labeling": type_decisions["labeling"].label,
        "confidence": confidence_for_entity(entity),
        "canonical_score": canonical.score if canonical else "",
        "evidence": evidence,
        "type_evidence": type_evidence,
        "mention_contexts": " || ".join(mention.context for mention in entity.mentions[:4]),
        "candidate_count": len(entity.candidates),
    }


def rejected_rows(entities: dict[str, DatasetEntity]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for entity in entities.values():
        for rejected in entity.rejected_candidates[:20]:
            rows.append(
                {
                    "dataset_id": entity.key,
                    "name": entity.display_name,
                    "artifact_type": artifact_type_for_work(rejected.work, entity),
                    "candidate_openalex_id": rejected.work.openalex_id,
                    "candidate_title": rejected.work.title,
                    "candidate_year": rejected.work.year or "",
                    "release_year_hint": release_year_from_aliases(entity) or "",
                    "candidate_citations": rejected.work.citations,
                    "score": rejected.score,
                    "discovery_method": readable_discovery_method(rejected.work.relation),
                    "reason": rejected.reason,
                    "alias_in_abstract": "yes" if aliases_in_text(entity, rejected.work.abstract, allow_low_precision=True) else "no",
                    "abstract_evidence": abstract_evidence_summary(entity, rejected.work),
                    "title_evidence": title_evidence_summary(entity, rejected.work.title),
                }
            )
    return rows


def candidate_rows(entities: dict[str, DatasetEntity]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for entity in entities.values():
        scored_by_work: dict[str, CandidateScore] = {}
        if entity.canonical:
            scored_by_work[entity.canonical.work.key] = entity.canonical
        for rejected in entity.rejected_candidates:
            scored_by_work.setdefault(rejected.work.key, rejected)
        for work in entity.candidates.values():
            score = scored_by_work.get(work.key) or score_canonical_candidate(entity, work)
            rows.append(
                {
                    "dataset_id": entity.key,
                    "name": entity.display_name,
                    "aliases": "; ".join(sorted(entity.aliases)),
                    "artifact_type": audit_artifact_type_for_work(work),
                    "candidate_openalex_id": work.openalex_id,
                    "candidate_title": work.title,
                    "candidate_year": work.year or "",
                    "release_year_hint": release_year_from_aliases(entity) or "",
                    "candidate_doi": work.doi,
                    "candidate_citations": work.citations,
                    "score": score.score,
                    "discovery_method": readable_discovery_method(work.relation),
                    "status": score.status,
                    "reason": score.reason,
                    "alias_in_abstract": "yes" if aliases_in_text(entity, work.abstract, allow_low_precision=True) else "no",
                    "abstract_evidence": abstract_evidence_summary(entity, work),
                    "title_evidence": score.reason,
                }
            )
    rows.sort(key=lambda row: (str(row["dataset_id"]), -float(row["score"])))
    return rows


def possible_dataset_rows(entities: dict[str, DatasetEntity]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for entity in entities.values():
        confidence = confidence_for_entity(entity)
        if confidence != "Low" and entity.canonical:
            continue
        canonical = entity.canonical
        work = canonical.work if canonical else None
        rows.append(
            {
                "dataset_id": entity.key,
                "name": entity.display_name,
                "aliases": "; ".join(sorted(entity.aliases)),
                "status": canonical.status if canonical else ("survey_verified" if trusted_survey_count(entity) >= 2 else "low_confidence"),
                "confidence": confidence,
                "survey_count": entity.survey_count,
                "trusted_survey_count": trusted_survey_count(entity),
                "candidate_count": len(entity.candidates),
                "rejected_candidate_count": len(entity.rejected_candidates),
                "best_candidate_openalex_id": work.openalex_id if work else "",
                "best_candidate_title": work.title if work else "",
                "best_candidate_score": canonical.score if canonical else "",
                "discovery_methods": "; ".join(entity_discovery_methods(entity)),
                "mention_sources": "; ".join(sorted({mention.source_kind for mention in entity.mentions})),
                "evidence": canonical.reason if canonical else "candidate-only or low-confidence mention",
                "mention_contexts": " || ".join(mention.context for mention in entity.mentions[:4]),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["survey_count"]),
            -int(row["candidate_count"]),
            str(row["name"]).casefold(),
        )
    )
    return rows


def synonym_rows(decisions: list[SynonymDecision]) -> list[dict[str, str | int | float]]:
    return [
        {
            "stage": decision.stage,
            "left_key": decision.left_key,
            "right_key": decision.right_key,
            "left_name": decision.left_name,
            "right_name": decision.right_name,
            "decision": decision.decision,
            "score": decision.score,
            "reason": decision.reason,
            "left_years": decision.left_years,
            "right_years": decision.right_years,
        }
        for decision in decisions
    ]


def validation_rows(entities: dict[str, DatasetEntity], accepted: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    seen_identity: dict[tuple[str, str], str] = {}
    for row in accepted:
        release_year = str(row.get("release_year") or row.get("year") or "")
        identity = (compact_name_base(str(row.get("name") or "")), release_year)
        old = seen_identity.get(identity)
        if old:
            rows.append(
                {
                    "level": "warning",
                    "check": "duplicate_normalized_identity",
                    "dataset_id": row.get("dataset_id", ""),
                    "name": row.get("name", ""),
                    "details": f"shares normalized identity with {old}",
                }
            )
        else:
            seen_identity[identity] = str(row.get("dataset_id") or "")

    for entity in entities.values():
        release_year, release_source = infer_release_year(entity, entity.canonical.work if entity.canonical else None)
        alias_years = sorted(entity_years_from_aliases(entity))
        if alias_years and release_year and int(release_year) != alias_years[0]:
            rows.append(
                {
                    "level": "warning",
                    "check": "release_year_mismatch",
                    "dataset_id": entity.key,
                    "name": entity.display_name,
                    "details": f"release_year={release_year} ({release_source}) but alias years={alias_years}",
                }
            )
        if entity.canonical and release_year and entity.canonical.work.year:
            if entity.canonical.work.year > int(release_year) + 3:
                rows.append(
                    {
                        "level": "warning",
                        "check": "late_canonical_publication",
                        "dataset_id": entity.key,
                        "name": entity.display_name,
                        "details": (
                            f"canonical_publication_year={entity.canonical.work.year} is much later than "
                            f"release_year={release_year}: {entity.canonical.work.title}"
                        ),
                    }
                )
    return rows


def read_recall_checks_from_file(path: Path | None) -> list[str]:
    if not path:
        return []
    checks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        checks.append(line)
    return checks


def recall_checks_for_args(args: argparse.Namespace) -> list[str]:
    checks = list(args.recall_check or [])
    checks.extend(read_recall_checks_from_file(args.require_recall_file))
    return list(dict.fromkeys(checks))


def recall_match_kind(entity: DatasetEntity, check: str) -> str:
    check_key = dataset_entity_key(check)
    check_compact = compact_key(check)
    alias_compacts = {compact_key(alias) for alias in entity.aliases}
    if check_key and check_key == entity.key:
        return "normalized_key"
    if check_compact and check_compact in alias_compacts:
        return "alias_exact"
    if entity_matches_text(entity, check):
        return "normalized_containment"
    if any(name_similarity(check, alias) >= 0.82 for alias in entity.aliases):
        return "fuzzy_alias"
    return ""


def recall_check_rows(
    entities: dict[str, DatasetEntity],
    checks: list[str] | None,
    *,
    min_survey_mentions: int = 2,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for raw_check in checks or []:
        check = canonical_dataset_name(raw_check)
        matches = [(recall_match_kind(entity, check), entity) for entity in entities.values()]
        matches = [(kind, entity) for kind, entity in matches if kind]
        if not matches:
            rows.append(
                {
                    "check": raw_check,
                    "status": "missing",
                    "match_kind": "",
                    "matched_dataset_id": "",
                    "matched_name": "",
                    "confidence": "",
                    "entity_status": "",
                    "discovery_methods": "",
                    "canonical_title": "",
                    "details": "No matching entity found by generic discovery. Inspect survey extraction, broad search hits, and possible-candidate audit.",
                }
            )
            continue
        matches.sort(
            key=lambda item: (
                {"normalized_key": 0, "alias_exact": 1, "normalized_containment": 2, "fuzzy_alias": 3}.get(item[0], 9),
                {"High": 0, "Medium": 1, "Low": 2}.get(confidence_for_entity(item[1]), 9),
                -trusted_survey_count(item[1]),
                -item[1].survey_count,
                item[1].display_name.casefold(),
            )
        )
        match_kind, entity = matches[0]
        canonical = entity.canonical
        work = canonical.work if canonical else None
        confidence = confidence_for_entity(entity)
        included = should_output_entity(
            entity,
            argparse.Namespace(include_low_confidence=False, min_survey_mentions=min_survey_mentions),
        )
        status = "accepted" if included else ("alias_matched" if match_kind == "fuzzy_alias" else "candidate_only")
        rows.append(
            {
                "check": raw_check,
                "status": status,
                "match_kind": match_kind,
                "matched_dataset_id": entity.key,
                "matched_name": entity.display_name,
                "confidence": confidence,
                "entity_status": canonical.status if canonical else ("survey_verified" if trusted_survey_count(entity) >= 2 else "low_confidence"),
                "discovery_methods": "; ".join(entity_discovery_methods(entity)),
                "canonical_title": work.title if work else "",
                "details": "Found by generic discovery/audit; not used as a discovery seed or search term.",
            }
        )
    return rows


def entity_audit_rows(entities: dict[str, DatasetEntity]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for entity in sorted(entities.values(), key=lambda item: item.display_name.casefold()):
        canonical = entity.canonical
        work = canonical.work if canonical else None
        rows.append(
            {
                "dataset_id": entity.key,
                "name": entity.display_name,
                "aliases": "; ".join(sorted(entity.aliases)),
                "status": canonical.status if canonical else ("survey_verified" if trusted_survey_count(entity) >= 2 else "low_confidence"),
                "confidence": confidence_for_entity(entity),
                "survey_count": entity.survey_count,
                "trusted_survey_count": trusted_survey_count(entity),
                "candidate_count": len(entity.candidates),
                "rejected_candidate_count": len(entity.rejected_candidates),
                "canonical_openalex_id": work.openalex_id if work else "",
                "canonical_title": work.title if work else "",
                "canonical_score": canonical.score if canonical else "",
                "artifact_type": artifact_type_for_work(work, entity) if work else artifact_type_decision(entity, None).label,
                "mention_sources": "; ".join(sorted({mention.source_kind for mention in entity.mentions})),
                "survey_sources": "; ".join(sorted({mention.source_title for mention in entity.mentions})),
                "mention_contexts": " || ".join(mention.context for mention in entity.mentions[:6]),
            }
        )
    return rows


def survey_rows(trusted: list[Work], entities: dict[str, DatasetEntity]) -> list[dict[str, str | int | float]]:
    entity_mentions_by_survey: dict[str, set[str]] = {}
    mention_count_by_survey: dict[str, int] = {}
    for entity in entities.values():
        for mention in entity.mentions:
            entity_mentions_by_survey.setdefault(mention.source_work_id, set()).add(entity.key)
            mention_count_by_survey[mention.source_work_id] = mention_count_by_survey.get(mention.source_work_id, 0) + 1

    rows: list[dict[str, str | int | float]] = []
    for index, work in enumerate(sorted(trusted, key=lambda item: (-score_survey_work(item), -item.citations, item.year or 9999)), start=1):
        rows.append(
            {
                "rank": index,
                "openalex_id": work.openalex_id,
                "doi": work.doi,
                "title": work.title,
                "year": work.year or "",
                "citations": work.citations,
                "survey_score": score_survey_work(work),
                "relation": readable_discovery_method(work.relation),
                "work_type": work.work_type,
                "url": work.url,
                "authors": work.authors,
                "referenced_work_count": len(work.referenced_works),
                "pdf_url_count": len(work.pdf_urls),
                "extracted_mention_count": mention_count_by_survey.get(work.openalex_id, 0),
                "extracted_entity_count": len(entity_mentions_by_survey.get(work.openalex_id, set())),
                "extracted_entity_ids": "; ".join(sorted(entity_mentions_by_survey.get(work.openalex_id, set()))),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(out: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    stem = out.with_suffix("")
    suffix = out.suffix or ".csv"
    return (
        out,
        stem.with_name(f"{stem.name}_candidates").with_suffix(suffix),
        stem.with_name(f"{stem.name}_rejected").with_suffix(suffix),
        stem.with_name(f"{stem.name}_synonyms").with_suffix(suffix),
        stem.with_name(f"{stem.name}_validation").with_suffix(suffix),
        stem.with_name(f"{stem.name}_entity_audit").with_suffix(suffix),
        stem.with_name(f"{stem.name}_surveys").with_suffix(suffix),
        stem.with_name(f"{stem.name}_possible").with_suffix(suffix),
        stem.with_name(f"{stem.name}_recall_check").with_suffix(suffix),
    )


def default_checkpoint_path(out: Path) -> Path:
    stem = out.with_suffix("")
    return stem.with_name(f"{stem.name}.checkpoint.json")


def checkpoint_path_for_args(args: argparse.Namespace) -> Path:
    return args.checkpoint or default_checkpoint_path(args.out)


def checkpoint_config(args: argparse.Namespace) -> dict:
    keys = (
        "trusted_paper",
        "trusted_papers_file",
        "auto_surveys",
        "survey_query",
        "max_surveys",
        "survey_pages",
        "survey_per_page",
        "min_survey_score",
        "min_survey_citations",
        "min_survey_mentions",
        "reference_snowball",
        "max_reference_works",
        "survey_reference_title_search",
        "direct_dataset_search",
        "broad_dataset_search",
        "dataset_query",
        "broad_dataset_query",
        "dataset_search_pages",
        "dataset_search_per_page",
        "min_dataset_source_score",
        "min_broad_dataset_source_score",
        "min_dataset_citations",
        "expand_resolved_entities",
        "fetch_pdf",
        "max_pdf_urls_per_survey",
        "max_pdf_chars",
        "candidate_query_profile",
        "max_aliases_per_entity",
        "max_candidate_queries_per_entity",
        "candidate_pages",
        "candidate_per_page",
        "resolve_entity",
        "include_low_confidence",
        "refresh_missing_abstracts",
        "refresh_abstract_entity",
        "refresh_abstract_limit",
        "recall_check",
        "require_recall_file",
    )
    config: dict[str, object] = {}
    for key in keys:
        value = getattr(args, key, None)
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def load_checkpoint(path: Path, args: argparse.Namespace) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Checkpoint ignored because it could not be read: {path} ({exc})", file=sys.stderr)
        return None
    if data.get("version") != CHECKPOINT_VERSION:
        print(f"Checkpoint ignored because its version is unsupported: {path}", file=sys.stderr)
        return None
    saved_config = data.get("config") or {}
    current_config = checkpoint_config(args)
    if saved_config and saved_config != current_config:
        print(
            f"Checkpoint config differs from current arguments; resuming anyway from {path}. "
            "Use --no-resume for a fresh crawl.",
            file=sys.stderr,
        )
    return data


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    *,
    current_stage: str,
    trusted: list[Work],
    entities: dict[str, DatasetEntity],
    synonym_decisions: list[SynonymDecision],
    completed_stages: set[str],
    completed_surveys: set[str],
    completed_references: set[str],
    completed_dataset_queries: set[str],
    resolved_entity_keys: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": CHECKPOINT_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "current_stage": current_stage,
        "config": checkpoint_config(args),
        "completed_stages": sorted(completed_stages),
        "completed_surveys": sorted(completed_surveys),
        "completed_references": sorted(completed_references),
        "completed_dataset_queries": sorted(completed_dataset_queries),
        "resolved_entity_keys": sorted(resolved_entity_keys),
        "trusted": [work_to_json(work) for work in trusted],
        "entities": {key: entity_to_json(entity) for key, entity in entities.items()},
        "synonym_decisions": [synonym_decision_to_json(decision) for decision in synonym_decisions],
    }
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def checkpoint_sets(data: dict | None) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    if not data:
        return set(), set(), set(), set(), set()
    return (
        set(data.get("completed_stages") or []),
        set(data.get("completed_surveys") or []),
        set(data.get("completed_references") or []),
        set(data.get("completed_dataset_queries") or []),
        set(data.get("resolved_entity_keys") or []),
    )


def run(args: argparse.Namespace) -> int:
    session = requests.Session()
    user_agent = "IDS-Dataset-Discovery/2.0"
    if args.mailto:
        user_agent += f" (mailto:{args.mailto})"
    session.headers.update({"User-Agent": user_agent})
    print(
        "OpenAlex request pacing: "
        f"pause={args.pause}s, mailto={'set' if args.mailto else 'missing'}, "
        f"api_key={'set' if args.api_key else 'missing'}, "
        f"stop_on_rate_limit={args.stop_on_rate_limit}.",
        file=sys.stderr,
    )
    if not args.mailto and not args.api_key:
        print(
            "OpenAlex polite-pool identity is missing; pass --mailto or set OPENALEX_MAILTO "
            "to reduce shared-pool throttling.",
            file=sys.stderr,
        )

    checkpoint_path = checkpoint_path_for_args(args)
    checkpoint_data = load_checkpoint(checkpoint_path, args) if args.resume else None
    completed_stages, completed_surveys, completed_references, completed_dataset_queries, resolved_entity_keys = checkpoint_sets(
        checkpoint_data
    )
    trusted = [work_from_json(item) for item in (checkpoint_data or {}).get("trusted", [])]
    entities = {
        str(key): entity_from_json(value)
        for key, value in ((checkpoint_data or {}).get("entities") or {}).items()
    }
    synonym_decisions = [
        synonym_decision_from_json(item) for item in (checkpoint_data or {}).get("synonym_decisions", [])
    ]
    if checkpoint_data:
        print(
            f"Resuming from checkpoint {checkpoint_path} "
            f"({len(entities)} entities, {len(resolved_entity_keys)} resolved entities).",
            file=sys.stderr,
        )

    def checkpoint(stage: str) -> None:
        save_checkpoint(
            checkpoint_path,
            args,
            current_stage=stage,
            trusted=trusted,
            entities=entities,
            synonym_decisions=synonym_decisions,
            completed_stages=completed_stages,
            completed_surveys=completed_surveys,
            completed_references=completed_references,
            completed_dataset_queries=completed_dataset_queries,
            resolved_entity_keys=resolved_entity_keys,
        )

    try:
        if "trusted_surveys" not in completed_stages:
            trusted = load_trusted_surveys(session, args)
            if not trusted:
                print("No trusted survey/base papers found.", file=sys.stderr)
                return 1
            completed_stages.add("trusted_surveys")
            checkpoint("trusted_surveys")

        print(f"Using {len(trusted)} trusted survey/base papers.", file=sys.stderr)
        for work in trusted:
            print(f"  - {work.title} ({work.year or 'n.d.'}, {work.citations} cites)", file=sys.stderr)

        if "survey_extraction" not in completed_stages:
            for survey in trusted:
                survey_key = survey.key
                if survey_key in completed_surveys:
                    continue
                extract_entities_from_survey(session, args, survey, entities)
                completed_surveys.add(survey_key)
                checkpoint("survey_extraction")

            print(f"Extracted {len(entities)} dataset-like entities from trusted survey text.", file=sys.stderr)
            synonym_decisions.extend(
                merge_synonym_entities(
                    entities,
                    stage="after_survey_extraction",
                    verbose=args.verbose_synonyms,
                )
            )
            completed_stages.add("survey_extraction")
            checkpoint("survey_extraction")
            print(f"Dataset-like entities after synonym merge: {len(entities)}", file=sys.stderr)
        else:
            print(f"Survey extraction already completed in checkpoint ({len(entities)} entities).", file=sys.stderr)

        if args.survey_reference_title_search:
            if "survey_reference_title_search" not in completed_stages:
                reference_link_added = discover_from_survey_reference_links(
                    session,
                    args,
                    trusted,
                    entities,
                    checkpoint_callback=lambda: checkpoint("survey_reference_title_search"),
                )
                print(
                    f"Added {reference_link_added} entity-candidate links from survey reference titles.",
                    file=sys.stderr,
                )
                synonym_decisions.extend(
                    merge_synonym_entities(
                        entities,
                        stage="after_survey_reference_titles",
                        verbose=args.verbose_synonyms,
                    )
                )
                completed_stages.add("survey_reference_title_search")
                checkpoint("survey_reference_title_search")
                print(f"Dataset-like entities after survey-reference synonym merge: {len(entities)}", file=sys.stderr)
            else:
                print("Survey reference-title search already completed in checkpoint.", file=sys.stderr)

        if args.reference_snowball:
            if "reference_snowball" not in completed_stages:
                reference_added = discover_from_trusted_references(
                    session,
                    args,
                    trusted,
                    entities,
                    completed_references=completed_references,
                    checkpoint_callback=lambda: checkpoint("reference_snowball"),
                )
                print(f"Added {reference_added} entity-candidate links from trusted references.", file=sys.stderr)
                synonym_decisions.extend(
                    merge_synonym_entities(
                        entities,
                        stage="after_reference_snowball",
                        verbose=args.verbose_synonyms,
                    )
                )
                completed_stages.add("reference_snowball")
                checkpoint("reference_snowball")
                print(f"Dataset-like entities after reference synonym merge: {len(entities)}", file=sys.stderr)
            else:
                print("Reference snowball already completed in checkpoint.", file=sys.stderr)

        if args.direct_dataset_search:
            if "direct_dataset_search" not in completed_stages:
                direct_added = discover_from_dataset_publication_search(
                    session,
                    args,
                    entities,
                    completed_queries=completed_dataset_queries,
                    checkpoint_callback=lambda: checkpoint("direct_dataset_search"),
                )
                print(f"Added {direct_added} entity-candidate links from direct dataset-publication search.", file=sys.stderr)
                synonym_decisions.extend(
                    merge_synonym_entities(
                        entities,
                        stage="after_direct_search",
                        verbose=args.verbose_synonyms,
                    )
                )
                completed_stages.add("direct_dataset_search")
                checkpoint("direct_dataset_search")
                print(f"Dataset-like entities after direct-search synonym merge: {len(entities)}", file=sys.stderr)
            else:
                print("Direct dataset-publication search already completed in checkpoint.", file=sys.stderr)

        if args.broad_dataset_search:
            broad_added = discover_from_broad_dataset_search(
                session,
                args,
                entities,
                completed_queries=completed_dataset_queries,
                checkpoint_callback=lambda: checkpoint("broad_dataset_search"),
            )
            if broad_added:
                print(f"Added {broad_added} entity-candidate links from broad dataset search.", file=sys.stderr)
                synonym_decisions.extend(
                    merge_synonym_entities(
                        entities,
                        stage="after_broad_search",
                        verbose=args.verbose_synonyms,
                    )
                )
                completed_stages.add("broad_dataset_search")
                checkpoint("broad_dataset_search")
                print(f"Dataset-like entities after broad-search synonym merge: {len(entities)}", file=sys.stderr)
            else:
                completed_stages.add("broad_dataset_search")
                print("No new broad dataset search results to add.", file=sys.stderr)

        reconstructed = reconstruct_high_recall_entities(
            entities,
            min_citations=args.min_dataset_citations,
        )
        if reconstructed:
            print(f"Added {reconstructed} high-recall entity-candidate links from known work text.", file=sys.stderr)
            checkpoint("high_recall_reconstruction")

        pruned = prune_implausible_entities(entities, stage="candidate resolution")
        if pruned:
            checkpoint("entity_pruning")

        print(f"Total dataset-like entities before resolution: {len(entities)}", file=sys.stderr)
        force_resolution = bool(args.resolve_entity)
        if "resolution" not in completed_stages or force_resolution:
            resolve_entities(
                session,
                args,
                entities,
                resolved_entity_keys=resolved_entity_keys,
                checkpoint_callback=lambda: checkpoint("resolution"),
            )
            if not force_resolution:
                completed_stages.add("resolution")
            checkpoint("resolution")
        else:
            print("Candidate resolution already completed in checkpoint.", file=sys.stderr)

        refreshed_abstracts = refresh_missing_candidate_abstracts(session, args, entities)
        if refreshed_abstracts:
            print(f"Refreshed abstracts for {refreshed_abstracts} candidate work(s).", file=sys.stderr)
            checkpoint("abstract_refresh")

        if "final_synonym_merge" not in completed_stages:
            synonym_decisions.extend(
                merge_synonym_entities(
                    entities,
                    stage="after_resolution",
                    verbose=args.verbose_synonyms,
                )
            )
            rescore_entity_canonicals(entities, min_citations=args.min_dataset_citations)
            completed_stages.add("final_synonym_merge")
            checkpoint("final_synonym_merge")
        else:
            rescore_entity_canonicals(entities, min_citations=args.min_dataset_citations)
        normalized_entities = normalize_single_source_canonical_entity_keys(entities)
        if normalized_entities:
            rescore_entity_canonicals(entities, min_citations=args.min_dataset_citations)
            checkpoint("canonical_title_key_normalization")
            print(
                f"Normalized {normalized_entities} single-source canonical entity key(s) from canonical titles.",
                file=sys.stderr,
            )
        print(f"Total dataset-like entities after final synonym merge: {len(entities)}", file=sys.stderr)

        accepted = [entity_to_row(entity) for entity in entities.values() if should_output_entity(entity, args)]

        accepted.sort(
            key=lambda row: (
                {"High": 0, "Medium": 1, "Low": 2}.get(str(row["confidence"]), 9),
                -(int(row["trusted_survey_count"]) if row.get("trusted_survey_count") else 0),
                -int(row["survey_count"]),
                -(int(row["citations"]) if row["citations"] else 0),
                str(row["name"]).casefold(),
            )
        )

        validations = validation_rows(entities, accepted)
        for row in validations:
            print(
                f"  validation {row['level']}: {row['check']} for {row['name']} ({row['dataset_id']}): {row['details']}",
                file=sys.stderr,
            )

        (
            accepted_path,
            candidates_path,
            rejected_path,
            synonyms_path,
            validation_path,
            entity_audit_path,
            surveys_path,
            possible_path,
            recall_check_path,
        ) = output_paths(args.out)
        write_csv(accepted_path, accepted)
        write_csv(candidates_path, candidate_rows(entities))
        write_csv(rejected_path, rejected_rows(entities))
        write_csv(synonyms_path, synonym_rows(synonym_decisions))
        write_csv(validation_path, validations)
        write_csv(entity_audit_path, entity_audit_rows(entities))
        write_csv(surveys_path, survey_rows(trusted, entities))
        possible = possible_dataset_rows(entities)
        required_checks = recall_checks_for_args(args)
        recall_checks = recall_check_rows(
            entities,
            required_checks,
            min_survey_mentions=args.min_survey_mentions,
        )
        missing_required = [row for row in recall_checks if row.get("status") == "missing"]
        write_csv(possible_path, possible)
        write_csv(recall_check_path, recall_checks)
        completed_stages.add("outputs_written")
        checkpoint("outputs_written")

        print(f"Saved accepted datasets: {accepted_path} ({len(accepted)} rows)")
        print(f"Saved candidate audit:   {candidates_path}")
        print(f"Saved rejected audit:    {rejected_path}")
        print(f"Saved synonym audit:     {synonyms_path}")
        print(f"Saved validation audit:  {validation_path}")
        print(f"Saved entity audit:      {entity_audit_path}")
        print(f"Saved survey audit:      {surveys_path}")
        print(f"Saved possible audit:    {possible_path} ({len(possible)} rows)")
        if required_checks:
            print(f"Saved recall check:      {recall_check_path}")
        print(f"Saved resume checkpoint: {checkpoint_path}")
        if missing_required and args.require_recall_file:
            missing_text = ", ".join(str(row["check"]) for row in missing_required)
            print(
                f"Required recall missing {len(missing_required)} dataset(s): {missing_text}",
                file=sys.stderr,
            )
            return 2
        return 0
    except OpenAlexRateLimitError as exc:
        checkpoint("rate_limited")
        print(
            f"Stopped after {exc.source} rate limit requested a {exc.wait_seconds}s wait. "
            f"Progress was saved to {checkpoint_path}; rerun the same command to resume.",
            file=sys.stderr,
        )
        return 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex-only IDS dataset discovery from trusted surveys.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Accepted dataset CSV path.")
    parser.add_argument("--checkpoint", type=Path, help="Resume-state JSON path. Defaults to <out-stem>.checkpoint.json.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from an existing checkpoint when available.")
    parser.add_argument("--trusted-paper", action="append", help="Trusted paper DOI, OpenAlex ID/URL, or title. Repeatable.")
    parser.add_argument("--trusted-papers-file", type=Path, help="Line-based file of trusted papers. Supports doi:, openalex:, title: prefixes.")
    parser.add_argument("--auto-surveys", action=argparse.BooleanOptionalAction, default=True, help="Also auto-discover trusted survey/base papers and merge them with manually supplied trusted papers.")
    parser.add_argument("--survey-query", action="append", help="Survey discovery query. If omitted, built-in survey queries are used.")
    parser.add_argument("--max-surveys", type=int, default=30, help="Maximum discovered trusted surveys to use.")
    parser.add_argument("--survey-pages", type=int, default=3, help="OpenAlex pages per survey query.")
    parser.add_argument("--survey-per-page", type=int, default=75, help="OpenAlex results per survey page.")
    parser.add_argument("--min-survey-score", type=int, default=15, help="Minimum heuristic score for auto-discovered survey papers.")
    parser.add_argument("--min-survey-citations", type=int, default=20, help="Minimum OpenAlex citation count for trusted survey papers.")
    parser.add_argument(
        "--min-survey-mentions",
        type=int,
        default=2,
        help="Minimum trusted-survey mentions required for survey-only resolution/output.",
    )
    parser.add_argument("--reference-snowball", action=argparse.BooleanOptionalAction, default=True, help="Inspect trusted papers' referenced works for dataset-publication papers.")
    parser.add_argument("--max-reference-works", type=int, default=300, help="Maximum trusted-paper references to inspect.")
    parser.add_argument(
        "--survey-reference-title-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use citation-linked survey bibliography titles as candidate base publications.",
    )
    parser.add_argument("--direct-dataset-search", action=argparse.BooleanOptionalAction, default=True, help="Run narrow OpenAlex searches for dataset-publication papers.")
    parser.add_argument("--broad-dataset-search", action=argparse.BooleanOptionalAction, default=True, help="Run broad generic OpenAlex searches for dataset-publication candidates.")
    parser.add_argument("--dataset-query", action="append", help="Dataset-publication discovery query. If omitted, built-in IDS dataset queries are used.")
    parser.add_argument("--broad-dataset-query", action="append", help="Broad dataset discovery query. If omitted, built-in generic IDS dataset queries are used.")
    parser.add_argument("--dataset-search-pages", type=int, default=2, help="OpenAlex pages per dataset-publication query.")
    parser.add_argument("--dataset-search-per-page", type=int, default=75, help="OpenAlex results per dataset-publication page.")
    parser.add_argument("--min-dataset-source-score", type=int, default=5, help="Minimum heuristic score for a paper to seed a dataset entity.")
    parser.add_argument("--min-broad-dataset-source-score", type=int, default=3, help="Minimum heuristic score for broad-search papers to seed candidate entities.")
    parser.add_argument("--min-dataset-citations", type=int, default=10, help="Minimum OpenAlex citation count for dataset/base-paper candidates.")
    parser.add_argument("--expand-resolved-entities", action="store_true", help="Also run alias-resolution searches for entities already resolved from direct papers.")
    parser.add_argument("--fetch-pdf", action=argparse.BooleanOptionalAction, default=True, help="Fetch OA PDFs and extract text with pdftotext.")
    parser.add_argument("--max-pdf-urls-per-survey", type=int, default=2)
    parser.add_argument("--max-pdf-chars", type=int, default=180000)
    parser.add_argument(
        "--candidate-query-profile",
        choices=("compact", "full"),
        default="compact",
        help="Candidate-resolution query template breadth. Use full with higher caps to reproduce the old broad fanout.",
    )
    parser.add_argument("--max-aliases-per-entity", type=int, default=3, help="Maximum entity aliases used for candidate-resolution queries.")
    parser.add_argument(
        "--max-candidate-queries-per-entity",
        type=int,
        default=15,
        help="Maximum candidate-resolution queries per entity. Use 0 for no cap.",
    )
    parser.add_argument("--candidate-pages", type=int, default=2, help="OpenAlex pages per dataset candidate query.")
    parser.add_argument("--candidate-per-page", type=int, default=75, help="OpenAlex results per dataset candidate query.")
    parser.add_argument("--resolve-entity", action="append", help="Limit candidate resolution to matching entity id/name/alias and rerun it even if checkpoint marks it resolved.")
    parser.add_argument("--include-low-confidence", action="store_true", help="Include low-confidence entities in accepted CSV.")
    parser.add_argument("--refresh-missing-abstracts", action="store_true", help="Refetch candidate works that are missing cached OpenAlex abstracts before rescoring.")
    parser.add_argument("--refresh-abstract-entity", action="append", help="Limit abstract refresh to matching entity id/name/alias. Repeatable.")
    parser.add_argument("--refresh-abstract-limit", type=int, default=200, help="Maximum candidate works to refetch during abstract refresh. Use 0 for no cap.")
    parser.add_argument("--recall-check", action="append", help="Dataset name to report as accepted, candidate-only, or missing without using it as a discovery seed.")
    parser.add_argument("--require-recall-file", type=Path, help="Line-based required recall targets. Used only for post-discovery validation; missing targets exit nonzero.")
    parser.add_argument("--verbose-synonyms", action=argparse.BooleanOptionalAction, default=True, help="Log every audited synonym merge/rejection decision to stderr.")
    parser.add_argument("--verbose-skips", action=argparse.BooleanOptionalAction, default=False, help="Log papers skipped by hard citation floors.")
    parser.add_argument("--mailto", default=os.getenv("OPENALEX_MAILTO"), help="Email for OpenAlex polite pool.")
    parser.add_argument("--api-key", default=os.getenv("OPENALEX_API_KEY"), help="Optional OpenAlex API key.")
    parser.add_argument("--pause", type=float, default=0.15, help="Pause between OpenAlex list requests.")
    parser.add_argument("--stop-on-rate-limit", action=argparse.BooleanOptionalAction, default=True, help="Stop and checkpoint when OpenAlex asks for a long rate-limit wait.")
    args = parser.parse_args()

    if args.max_surveys < 1:
        parser.error("--max-surveys must be at least 1")
    if args.survey_pages < 1 or args.candidate_pages < 1 or args.dataset_search_pages < 1:
        parser.error("--survey-pages, --candidate-pages, and --dataset-search-pages must be at least 1")
    if args.survey_per_page < 1 or args.survey_per_page > 100:
        parser.error("--survey-per-page must be between 1 and 100")
    if args.dataset_search_per_page < 1 or args.dataset_search_per_page > 100:
        parser.error("--dataset-search-per-page must be between 1 and 100")
    if args.candidate_per_page < 1 or args.candidate_per_page > 100:
        parser.error("--candidate-per-page must be between 1 and 100")
    if args.max_aliases_per_entity < 1:
        parser.error("--max-aliases-per-entity must be at least 1")
    if args.max_candidate_queries_per_entity < 0:
        parser.error("--max-candidate-queries-per-entity must be non-negative")
    if args.max_reference_works < 0:
        parser.error("--max-reference-works must be non-negative")
    if args.refresh_abstract_limit < 0:
        parser.error("--refresh-abstract-limit must be non-negative")
    if args.require_recall_file and not args.require_recall_file.exists():
        parser.error(f"--require-recall-file does not exist: {args.require_recall_file}")
    if args.min_broad_dataset_source_score < 0:
        parser.error("--min-broad-dataset-source-score must be non-negative")
    if args.min_survey_citations < 0:
        parser.error("--min-survey-citations must be non-negative")
    if args.min_dataset_citations < 0:
        parser.error("--min-dataset-citations must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
