#!/usr/bin/env python3
"""Discover IDS and attack-detection dataset publications.

This crawler intentionally does not maintain a fixed catalogue of dataset
names. It queries scholarly APIs with broad intrusion/attack-detection search
phrases, extracts likely dataset names from primary dataset-publication titles,
and keeps the best publication per discovered dataset.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


OUT = Path("ids_dataset_candidates.csv")
OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 4
MAX_RETRY_AFTER_SECONDS = 60

DEFAULT_PER_PAGE = 75
DEFAULT_PAGES = 2
DEFAULT_MIN_CITATIONS = 0
DEFAULT_SOURCES = ("openalex", "crossref")

CATEGORY_ORDER = [
    "Classical",
    "Enterprise",
    "IoT",
    "IIoT",
    "MQTT",
    "ICS/SCADA",
    "DNS",
    "Wireless",
    "Host",
    "Web/App",
    "Cloud",
    "Automotive/CAN",
    "Malware/Botnet",
    "Unknown",
]

DISCOVERY_QUERIES = [
    "intrusion detection dataset benchmark",
    "network intrusion detection dataset",
    "intrusion detection systems dataset",
    "benchmark dataset intrusion detection systems",
    "towards generation intrusion detection dataset",
    "generation of intrusion detection dataset",
    "labeled network traffic dataset intrusion",
    "malicious benign network traffic dataset",
    "network flow dataset intrusion",
    "cybersecurity intrusion detection dataset",
    "cybersecurity dataset malicious benign traffic",
    "attack detection dataset cybersecurity",
    "cyber attack dataset benchmark",
    "network attack dataset",
    "network security dataset intrusion",
    "network traffic dataset attack detection",
    "flow dataset intrusion detection",
    "botnet traffic dataset intrusion detection",
    "malware traffic dataset",
    "DDoS attack dataset",
    "DDoS traffic dataset",
    "DoS DDoS intrusion detection dataset",
    "IoT intrusion detection dataset",
    "IoT botnet dataset",
    "IIoT intrusion detection dataset",
    "industrial IoT attack dataset",
    "industrial control system security dataset",
    "industrial control system intrusion dataset",
    "SCADA intrusion detection dataset",
    "cyber physical system attack dataset",
    "smart grid cyber attack dataset",
    "water treatment attack dataset",
    "MQTT intrusion detection dataset",
    "MQTT attack dataset",
    "DNS exfiltration dataset",
    "DNS data exfiltration dataset",
    "DNS exfiltration attack dataset",
    "DNS tunneling dataset",
    "DNS over HTTPS tunnel dataset",
    "wireless intrusion detection dataset",
    "WiFi attack dataset",
    "host intrusion detection dataset",
    "system call intrusion detection dataset",
    "Windows event log intrusion detection dataset",
    "provenance intrusion detection dataset",
    "database intrusion detection dataset",
    "web attack dataset",
    "SQL injection dataset",
    "cloud intrusion detection dataset",
    "container security dataset attack detection",
    "CAN bus intrusion detection dataset",
    "vehicular intrusion detection dataset",
    "automotive intrusion detection dataset",
    "malware detection dataset benchmark",
]

DATASET_WORDS = (
    "dataset",
    "data set",
    "benchmark",
    "corpus",
    "testbed",
    "traffic traces",
    "trace dataset",
    "collection",
)

PRIMARY_DATASET_TITLE_WORDS = (
    "a dataset for",
    "dataset for",
    "a benchmark for",
    "benchmark for",
    "towards the generation",
    "toward the generation",
    "generation of",
    "new intrusion detection dataset",
    "new dataset",
    "labeled dataset",
    "labelled dataset",
    "annotated dataset",
    "realistic dataset",
    "real-world dataset",
    "real world dataset",
)

INTRODUCTION_WORDS = (
    "a new",
    "new dataset",
    "realistic",
    "real-world",
    "real world",
    "comprehensive",
    "annotated",
    "labeled",
    "labelled",
    "toward generating",
    "towards generating",
    "developing",
    "development of",
    "generated",
    "generation of",
    "release",
    "releasing",
    "introduced",
    "introducing",
    "presents",
    "we present",
    "we release",
    "dataset to support",
)

CYBERSECURITY_WORDS = (
    "intrusion",
    "attack",
    "cyber",
    "security",
    "malware",
    "botnet",
    "ddos",
    "dos",
    "phishing",
    "ransomware",
    "cyberattack",
    "cyber-attack",
    "pentesting",
    "penetration testing",
    "exfiltration",
    "tunnel",
    "anomaly",
    "threat",
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
    "system call",
    "event log",
)

STRONG_CYBERSECURITY_TITLE_WORDS = (
    "ids",
    "intrusion",
    "attack",
    "cyber",
    "cybersecurity",
    "cyberattack",
    "cyber-attack",
    "malware",
    "botnet",
    "ddos",
    "dos",
    "phishing",
    "ransomware",
    "exfiltration",
    "tunnel",
    "threat detection",
    "scada",
    "industrial control",
    "pentesting",
    "penetration testing",
    "forensic",
)

DOWNSTREAM_TITLE_PATTERNS = (
    r"\busing\b",
    r"\bbased on\b",
    r"\bclassification\b",
    r"\bclassifier\b",
    r"\bmodel\b",
    r"\bapproach\b",
    r"\balgorithm\b",
    r"\bfeature selection\b",
    r"\bfeature extraction\b",
    r"\boptimization\b",
    r"\boptimizing\b",
    r"\btransfer learning\b",
    r"\bexplainable\b",
    r"\badversarial\b",
    r"\bcomparative\b",
    r"\bcomparison\b",
    r"\bevaluating\b",
    r"\bevaluation of\b",
    r"\bperformance\b",
)

REJECT_TITLE_WORDS = (
    "survey",
    "review",
    "systematic literature",
    "comparative study",
    "bibliometric",
    "taxonomy",
    "analysis of benchmark",
    "detailed analysis",
    "statistical analysis",
    "usefulness of",
    "performance improvements",
    "validation using",
    "validation of",
    "application of machine learning",
    "identifying relevant features",
    "studying the impact",
    "impact of dataset",
    "cross-dataset",
    "cross dataset",
    "open-set",
    "open set",
    "trained dataset",
    "dataset validation",
    "dataset-centric evaluation",
    "dataset centric evaluation",
    "implementation of",
    "machine learning approach",
    "machine learning based intrusion detection",
    "machine learning algorithms for",
    "enhancing intrusion detection",
    "visualizing realistic benchmarked",
    "person detection",
    "cyber threat intelligence",
    "poster:",
    "toolkit",
    "supplemental information",
    "supplementary information",
    "supplemental material",
    "supplementary material",
    "supplemental file",
    "supplementary file",
)

BAD_DATASET_NAMES = {
    "IDS",
    "IPS",
    "NIDS",
    "HIDS",
    "IoT",
    "IIoT",
    "ICS",
    "SCADA",
    "MQTT",
    "DNS",
    "DoH",
    "TCP",
    "UDP",
    "HTTP",
    "HTTPS",
    "URL",
    "URLs",
    "URI",
    "URIs",
    "IP",
    "CAN",
    "ML",
    "DL",
    "AI",
    "CNN",
    "RNN",
    "LSTM",
    "SVM",
    "DNN",
    "F1",
    "F1-score",
    "F1-scores",
    "F-Measure",
    "RF",
    "KNN",
    "ROC",
    "ROC-AUC",
    "AUC",
    "C4.5",
    "C-Means",
    "K-Nearest",
    "One-Class",
    "CNN-LSTM",
    "ML-based",
    "FL-based",
    "IoT-based",
    "IoT-specific",
    "SDN-based",
    "SDN-managed",
    "SDN-oriented",
    "Software-Defined",
    "IDS-based",
    "UDS-based",
    "Deep-Learning",
    "Cyber-Attack",
    "Z-score",
    "MQTT-based",
    "DGA-based",
    "RPL-based",
    "IDSs-enabled",
    "IoT/IIoT",
    "Wi-Fi",
    "GNS3",
    "NS3",
    "VLAN1",
    "VLAN2",
    "Modbus/TCP",
    "TCP-SYNC",
    "PLC-HMI",
    "Comma-Separated",
    "Design/methodology/approach",
    "Originality/value",
    "Dataset",
    "Data",
    "Benchmark",
    "Network",
    "Intrusion",
    "Detection",
    "Security",
    "Poster",
}

BAD_DATASET_NAME_KEYS = {item.casefold() for item in BAD_DATASET_NAMES}


@dataclass
class Work:
    source: str
    source_id: str
    title: str
    abstract: str
    year: int | None
    citations: int
    authors: str
    paper_url: str
    doi: str
    source_query: str


def normalize_text(text: str | None) -> str:
    text = str(text or "")
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def comparable(text: str | None) -> str:
    text = normalize_text(text).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def has_phrase(text: str | None, phrases: Iterable[str]) -> bool:
    normalized = f" {comparable(text)} "
    return any(f" {comparable(phrase)} " in normalized for phrase in phrases if comparable(phrase))


def compact_key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text).casefold())


def canonical_dataset_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"\b(dataset|data set|benchmark|corpus|testbed)\b$", "", name, flags=re.I).strip()
    name = name.strip(" ,.;:()[]{}")
    name = re.sub(r"\s*-\s*", "-", name)
    name = re.sub(r"\s+", " ", name)
    return name


def looks_like_dataset_name(candidate: str) -> bool:
    candidate = canonical_dataset_name(candidate)
    if len(candidate) < 3 or len(candidate) > 80:
        return False
    if candidate in BAD_DATASET_NAMES or candidate.casefold() in BAD_DATASET_NAME_KEYS:
        return False
    if re.fullmatch(r"\d{4}", candidate):
        return False

    has_digit = bool(re.search(r"\d", candidate))
    has_separator = bool(re.search(r"[-_/]", candidate))
    has_mixed_caps = bool(re.search(r"[A-Z]", candidate)) and bool(re.search(r"[a-z]", candidate))
    all_caps = bool(re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{2,}", candidate))
    ends_with_set = candidate.lower().endswith("set")
    return has_digit or has_separator or has_mixed_caps or all_caps or ends_with_set


def reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    return " ".join(word for _, word in sorted(positions))


def request_json(session: requests.Session, url: str, params: dict, source: str) -> dict:
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
                    raise RuntimeError(f"{source} rate limited for {wait}s")
                print(f"{source} rate limited, sleeping {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            if 500 <= response.status_code < 600 and attempt < REQUEST_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt >= REQUEST_RETRIES:
                raise RuntimeError(f"{source} request failed for {url}: {exc}") from exc
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    raise RuntimeError(f"{source} request failed for {url}")


def openalex_search(session: requests.Session, query: str, per_page: int, pages: int) -> list[Work]:
    works: list[Work] = []
    for page in range(1, pages + 1):
        payload = request_json(
            session,
            OPENALEX_URL,
            {
                "search": query,
                "per-page": per_page,
                "page": page,
                "select": ",".join(
                    [
                        "id",
                        "doi",
                        "title",
                        "display_name",
                        "publication_year",
                        "cited_by_count",
                        "abstract_inverted_index",
                        "primary_location",
                        "authorships",
                    ]
                ),
            },
            "OpenAlex",
        )
        for item in payload.get("results") or []:
            title = item.get("title") or item.get("display_name") or ""
            location = item.get("primary_location") or {}
            authors = []
            for authorship in (item.get("authorships") or [])[:6]:
                author = authorship.get("author") or {}
                if author.get("display_name"):
                    authors.append(author["display_name"])
            works.append(
                Work(
                    source="openalex",
                    source_id=item.get("id") or "",
                    title=normalize_text(title),
                    abstract=normalize_text(reconstruct_openalex_abstract(item.get("abstract_inverted_index"))),
                    year=item.get("publication_year"),
                    citations=int(item.get("cited_by_count") or 0),
                    authors=", ".join(authors),
                    paper_url=location.get("landing_page_url") or item.get("doi") or item.get("id") or "",
                    doi=item.get("doi") or "",
                    source_query=query,
                )
            )
        time.sleep(0.25)
    return works


def crossref_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [[]])[0]
        if parts and parts[0] is not None:
            return int(parts[0])
    return None


def crossref_authors(item: dict) -> str:
    names = []
    for author in (item.get("author") or [])[:6]:
        given = author.get("given") or ""
        family = author.get("family") or ""
        name = " ".join(part for part in (given, family) if part).strip()
        if name:
            names.append(name)
    return ", ".join(names)


def crossref_search(session: requests.Session, query: str, per_page: int, pages: int) -> list[Work]:
    works: list[Work] = []
    for page in range(1, pages + 1):
        payload = request_json(
            session,
            CROSSREF_URL,
            {
                "query.bibliographic": query,
                "rows": per_page,
                "offset": (page - 1) * per_page,
                "select": "DOI,title,abstract,URL,author,published-print,published-online,published,issued,created,is-referenced-by-count",
            },
            "Crossref",
        )
        for item in ((payload.get("message") or {}).get("items") or []):
            title = " ".join(item.get("title") or [])
            doi = item.get("DOI") or ""
            works.append(
                Work(
                    source="crossref",
                    source_id=doi,
                    title=normalize_text(title),
                    abstract=normalize_text(item.get("abstract") or ""),
                    year=crossref_year(item),
                    citations=int(item.get("is-referenced-by-count") or 0),
                    authors=crossref_authors(item),
                    paper_url=item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                    doi=f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
                    source_query=query,
                )
            )
        time.sleep(0.25)
    return works


def semantic_scholar_search(session: requests.Session, query: str, per_page: int, pages: int) -> list[Work]:
    works: list[Work] = []
    limit = min(per_page, 100)
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    for page in range(pages):
        try:
            response = session.get(
                SEMANTIC_SCHOLAR_URL,
                params={
                    "query": query,
                    "limit": limit,
                    "offset": page * limit,
                    "fields": "title,abstract,year,citationCount,authors,url,externalIds",
                },
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                try:
                    wait = int(float(response.headers.get("Retry-After") or 5))
                except ValueError:
                    wait = 5
                raise RuntimeError(f"Semantic Scholar rate limited for {wait}s")
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            print(f"Semantic Scholar query skipped ({query}): {exc}", file=sys.stderr)
            break

        for item in payload.get("data") or []:
            external = item.get("externalIds") or {}
            doi = external.get("DOI") or ""
            authors = ", ".join((author.get("name") or "") for author in (item.get("authors") or [])[:6])
            works.append(
                Work(
                    source="semantic_scholar",
                    source_id=item.get("paperId") or doi,
                    title=normalize_text(item.get("title") or ""),
                    abstract=normalize_text(item.get("abstract") or ""),
                    year=item.get("year"),
                    citations=int(item.get("citationCount") or 0),
                    authors=authors,
                    paper_url=item.get("url") or (f"https://doi.org/{doi}" if doi else ""),
                    doi=f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
                    source_query=query,
                )
            )
        time.sleep(1.1 if not api_key else 0.2)
    return works


def source_search(session: requests.Session, source: str, query: str, per_page: int, pages: int) -> list[Work]:
    if source == "openalex":
        return openalex_search(session, query, per_page, pages)
    if source == "crossref":
        return crossref_search(session, query, per_page, pages)
    if source == "semantic_scholar":
        return semantic_scholar_search(session, query, per_page, pages)
    raise ValueError(f"Unsupported source: {source}")


def deduplicate_candidates(candidates: Iterable[str]) -> list[str]:
    normalized_candidates = {canonical_dataset_name(candidate) for candidate in candidates}
    normalized_candidates = {candidate for candidate in normalized_candidates if looks_like_dataset_name(candidate)}

    filtered = []
    for candidate in normalized_candidates:
        candidate_key = compact_key(candidate)
        if any(
            candidate_key != compact_key(other) and candidate_key in compact_key(other)
            for other in normalized_candidates
        ):
            continue
        filtered.append(candidate)

    return sorted(filtered, key=str.casefold)


def looks_like_strong_dataset_name(candidate: str) -> bool:
    candidate = canonical_dataset_name(candidate)
    has_digit = bool(re.search(r"\d", candidate))
    has_separator = bool(re.search(r"[-_/'.+]", candidate))
    all_caps = bool(re.fullmatch(r"[A-Z0-9][A-Z0-9_./'+-]{2,}", candidate))
    ends_with_set = candidate.lower().endswith("set")
    return looks_like_dataset_name(candidate) and (has_digit or has_separator or all_caps or ends_with_set)


def extract_title_dataset_candidates(title: str) -> list[str]:
    if not has_phrase(title, DATASET_WORDS):
        return []

    candidates: set[str] = set()
    patterns = [
        r"^\s*([A-Z][A-Za-z0-9]{2,})\s*(?::|\s+-\s+)",
        r"^\s*([A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+)\s*[:,-]",
        r"^\s*([A-Z][A-Za-z0-9]{2,}\d{2,4})\s*[:,-]",
        r"^\s*([A-Z][A-Za-z0-9]{2,}set)\s*[,:\-]",
        r"\b([A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+)\s+(?:dataset|data set|benchmark|corpus|testbed)\b",
        r"\b([A-Z]{2,}[A-Z0-9_-]*\d{0,4})\s+(?:dataset|data set|benchmark|corpus|testbed)\b",
        r"\b([A-Z][A-Za-z0-9]{2,}set)\b",
        r"(?:dataset|data set|benchmark|corpus|testbed)\s+(?:called|named)\s+([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, title):
            candidate = canonical_dataset_name(match)
            if looks_like_dataset_name(candidate):
                candidates.add(candidate)

    return deduplicate_candidates(candidates)


def extract_abstract_dataset_candidates(abstract: str) -> list[str]:
    abstract = normalize_text(abstract[:2500])
    if not has_phrase(abstract, DATASET_WORDS):
        return []

    candidates: set[str] = set()
    name = r"[A-Z][A-Za-z0-9]*(?:[-_./'+][A-Za-z0-9]+)+|[A-Z]{2,}[A-Z0-9_./'+-]*\d[A-Z0-9_./'+-]*"
    dataset_word = r"(?:dataset|data set|benchmark|corpus|testbed|traffic traces|trace dataset)"
    patterns = [
        rf"\b({name})\b.{{0,80}}\b{dataset_word}\b",
        rf"\b{dataset_word}\b.{{0,80}}\b({name})\b",
        rf"\b(?:called|named|known as|referred to as)\s+({name})\b",
        rf"\b(?:we|this paper)\s+(?:present|presents|introduce|introduces|release|releases|propose|proposes)\s+(?:the\s+)?({name})\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, abstract):
            candidate = canonical_dataset_name(match)
            if looks_like_strong_dataset_name(candidate):
                candidates.add(candidate)

    return deduplicate_candidates(candidates)


def extract_dataset_candidates(work: Work) -> list[str]:
    candidates = set(extract_title_dataset_candidates(work.title))
    if candidates:
        return deduplicate_candidates(candidates)
    if title_looks_primary_dataset_publication(work.title) and not is_downstream_usage_title(work.title):
        candidates.update(extract_abstract_dataset_candidates(work.abstract))
    return deduplicate_candidates(candidates)


def infer_category(name: str, title: str, abstract: str) -> str:
    name_title = f"{name} {title}".lower()
    text = f"{name_title} {abstract}".lower()
    if any(term in name_title for term in ("kdd", "darpa", "nsl")):
        return "Classical"
    if any(term in name_title for term in ("dns", "doh", "exfiltration", "tunnel")):
        return "DNS"
    if "mqtt" in name_title:
        return "MQTT"
    if any(term in name_title for term in ("iiot", "industrial iot", "edge-iiot")):
        return "IIoT"
    if any(
        term in name_title
        for term in ("iot", "iomt", "internet of things", "internet of medical things", "smart agriculture")
    ):
        return "IoT"
    if any(term in name_title for term in ("wireless", "wifi", "wi-fi", "802.11", "wsn", "uav", "unmanned aerial")):
        return "Wireless"
    if any(term in text for term in ("scada", "industrial control", "water treatment", "water distribution", "swat", "wadi")):
        return "ICS/SCADA"
    if has_phrase(text, ("ics",)):
        return "ICS/SCADA"
    if "mqtt" in text:
        return "MQTT"
    if any(term in text for term in ("dns", "doh", "exfiltration", "tunnel")):
        return "DNS"
    if any(term in text for term in ("iot", "smart agriculture", "smart farm", "precision agriculture")):
        return "IoT"
    if any(term in text for term in ("wireless", "wifi", "wi-fi", "802.11", "wireless sensor", "uav", "unmanned aerial")):
        return "Wireless"
    if any(term in text for term in ("host", "system call", "event log", "windows event")):
        return "Host"
    if any(term in text for term in ("web attack", "web application", "sql injection", "http", "xss")):
        return "Web/App"
    if any(term in text for term in ("cloud", "container", "kubernetes")):
        return "Cloud"
    if any(term in text for term in ("can bus", "vehicular", "vehicle", "automotive")):
        return "Automotive/CAN"
    if any(term in text for term in ("botnet", "malware", "ransomware")):
        return "Malware/Botnet"
    if has_phrase(text, ("intrusion", "attack", "network security", "network traffic", "ddos", "dos")):
        return "Enterprise"
    return "Unknown"


def infer_mode(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    if any(term in text for term in ("real-time", "streaming", "continuous", "dynamic", "online", "testbed")):
        return "Dynamic"
    return "Static"


def infer_origin(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    if any(term in text for term in ("synthetic", "simulated", "simulation", "generated", "emulated")):
        return "Synthetic"
    if any(term in text for term in ("real-world", "real world", "real traffic", "testbed", "production", "captured", "honeypot", "laboratory", "lab")):
        return "Real-world / Lab"
    return "Unknown"


def title_looks_primary_dataset_publication(title: str) -> bool:
    if not has_phrase(title, DATASET_WORDS):
        return False
    if has_phrase(title, PRIMARY_DATASET_TITLE_WORDS):
        return True

    title_cmp = comparable(title)
    if re.search(r"\b(?:dataset|benchmark|corpus|testbed)\s+(?:of|for|to support)\b", title_cmp):
        return True
    if re.search(
        r"\b(?:new|realistic|real world|labeled|labelled|annotated|comprehensive)\b.{0,55}"
        r"\b(?:dataset|benchmark|corpus|testbed)\b",
        title_cmp,
    ):
        return True
    if re.search(
        r"\b(?:dataset|benchmark|corpus|testbed)\b.{0,70}"
        r"\b(?:intrusion|attack|malware|ddos|cyber|anomaly|botnet|exfiltration)\b",
        title_cmp,
    ):
        return True
    return False


def is_downstream_usage_title(title: str) -> bool:
    title_cmp = comparable(title)
    if has_phrase(title, REJECT_TITLE_WORDS):
        return True
    if re.search(r"\b(?:supp|suppl)\d+\b.*\bpdf\b", title_cmp):
        return True
    if re.search(r"\b(?:pdf|docx?|xlsx?|zip)\b$", title_cmp) and has_phrase(title, ("supp", "dataset")):
        return True
    if re.search(r"^(using|evaluating|evaluation|classification|detecting|predicting|re evaluation)\b", title_cmp):
        return True
    if has_phrase(title, ("intrusion detection using", "attack detection using", "based on the", "based on a")):
        return True
    if has_phrase(title, ("dataset based", "dataset-based", "imbalanced dataset", "noisy dataset")):
        return True
    if has_phrase(title, ("detection system for", "classifier for", "model for")) and has_phrase(title, ("dataset", "data set")):
        return True
    if re.search(r"\b(using|exposed to|on|with|based on|evaluated on|tested on)\b.{0,80}\bdataset\b", title_cmp):
        return True
    return sum(1 for pattern in DOWNSTREAM_TITLE_PATTERNS if re.search(pattern, title_cmp)) >= 2


def looks_like_supplement_record(work: Work) -> bool:
    title_cmp = comparable(work.title)
    doi = (work.doi or work.source_id or "").casefold()
    if has_phrase(
        work.title,
        (
            "supplemental information",
            "supplementary information",
            "supplemental material",
            "supplementary material",
            "supplemental file",
            "supplementary file",
        ),
    ):
        return True
    if re.search(r"\b(?:supp|suppl)\d+\b.*\bpdf\b", title_cmp):
        return True
    if "/supp-" in doi or re.search(r"/(?:mm|supp|suppl|s)\d+(?:$|[/?#])", doi):
        return True
    return False


def looks_like_off_scope_benchmark(title: str) -> bool:
    if has_phrase(title, ("llm", "llms", "large language model", "large language models")) and not has_phrase(
        title,
        (
            "intrusion detection",
            "attack detection",
            "malware detection",
            "network intrusion",
            "anomaly detection",
            "threat detection",
        ),
    ):
        return True
    return False


def candidate_near_dataset_word(candidate: str, text: str, distance: int = 100) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    candidate_pattern = re.escape(normalize_text(candidate))
    dataset_word = r"(?:dataset|data set|benchmark|corpus|testbed|traffic traces|trace dataset)"
    boundary_left = r"(?<![A-Za-z0-9])"
    boundary_right = r"(?![A-Za-z0-9])"
    return bool(
        re.search(
            rf"{boundary_left}{candidate_pattern}{boundary_right}.{{0,{distance}}}\b{dataset_word}\b",
            text,
            flags=re.I,
        )
        or re.search(
            rf"\b{dataset_word}\b.{{0,{distance}}}{boundary_left}{candidate_pattern}{boundary_right}",
            text,
            flags=re.I,
        )
    )


def candidate_has_abstract_introduction(candidate: str, abstract: str) -> bool:
    abstract = normalize_text(abstract[:2500])
    if not abstract:
        return False
    candidate_pattern = re.escape(normalize_text(candidate))
    boundary_left = r"(?<![A-Za-z0-9])"
    boundary_right = r"(?![A-Za-z0-9])"
    intro = (
        r"(?:we|this paper|the paper|the authors)?\s*"
        r"(?:present|presents|introduce|introduces|release|releases|propose|proposes|"
        r"develop|develops|create|creates|generate|generates|provide|provides|call|called|name|named)"
    )
    return bool(
        re.search(
            rf"\b{intro}\b.{{0,160}}{boundary_left}{candidate_pattern}{boundary_right}",
            abstract,
            flags=re.I,
        )
        or re.search(
            rf"{boundary_left}{candidate_pattern}{boundary_right}.{{0,100}}"
            rf"\b(?:is|called|named|referred to as|dataset introduced|dataset presented)\b",
            abstract,
            flags=re.I,
        )
    )


def title_introduces_candidate(candidate: str, title: str) -> bool:
    title_norm = normalize_text(title)
    candidate_norm = re.escape(normalize_text(candidate))
    if re.search(rf"^\s*{candidate_norm}\b\s*[:,-]", title_norm, flags=re.I):
        return True
    if re.search(rf"^\s*{candidate_norm}\b\s+(?:dataset|data set|benchmark|corpus|testbed)\b", title_norm, flags=re.I):
        return True
    if re.search(rf"[:;]\s*{candidate_norm}\b\s+(?:dataset|data set|benchmark|corpus|testbed)\b", title_norm, flags=re.I):
        return True
    if has_phrase(title, INTRODUCTION_WORDS) and re.search(
        rf"\b{candidate_norm}\b\s+(?:dataset|data set|benchmark|corpus|testbed)\b",
        title_norm,
        flags=re.I,
    ):
        return True
    return False


def work_introduces_candidate(candidate: str, work: Work) -> bool:
    if title_introduces_candidate(candidate, work.title):
        return True
    if not title_looks_primary_dataset_publication(work.title):
        return False
    if candidate_near_dataset_word(candidate, work.abstract[:2500]) and candidate_has_abstract_introduction(
        candidate,
        work.abstract,
    ):
        return True
    return False


def score_candidate(candidate: str, work: Work) -> dict | None:
    title = work.title
    abstract = work.abstract
    text = f"{title}\n{abstract}"

    if not candidate or not looks_like_dataset_name(candidate):
        return None
    if looks_like_supplement_record(work):
        return None
    if looks_like_off_scope_benchmark(title):
        return None
    if not has_phrase(title, DATASET_WORDS):
        return None
    if not has_phrase(title, STRONG_CYBERSECURITY_TITLE_WORDS):
        return None
    if not has_phrase(text, CYBERSECURITY_WORDS):
        return None
    if not work_introduces_candidate(candidate, work):
        return None
    if is_downstream_usage_title(title):
        return None

    score = 0
    reasons = []

    if has_phrase(title, DATASET_WORDS):
        score += 35
        reasons.append("dataset wording in title")
    if has_phrase(title, INTRODUCTION_WORDS) or has_phrase(abstract[:800], INTRODUCTION_WORDS):
        score += 25
        reasons.append("introduction/release wording")
    if has_phrase(title, CYBERSECURITY_WORDS):
        score += 20
        reasons.append("security context in title")
    elif has_phrase(abstract[:1200], CYBERSECURITY_WORDS):
        score += 10
        reasons.append("security context in abstract")
    if compact_key(candidate) in compact_key(title):
        score += 30
        reasons.append("dataset name in title")
    elif compact_key(candidate) in compact_key(abstract[:2500]):
        score += 20
        reasons.append("dataset name in abstract")
    if work.doi:
        score += 3
    if work.citations >= 10:
        score += 2
    if work.citations >= 100:
        score += 2

    if score < 60:
        return None

    return {
        "dataset": canonical_dataset_name(candidate),
        "category": infer_category(candidate, title, abstract),
        "mode": infer_mode(title, abstract),
        "origin": infer_origin(title, abstract),
        "year": work.year,
        "citations": work.citations,
        "paper_title": title,
        "authors": work.authors,
        "paper_url": work.paper_url,
        "doi": work.doi,
        "source": work.source,
        "source_id": work.source_id,
        "source_query": work.source_query,
        "match_score": score,
        "match_reason": "; ".join(reasons),
    }


def discover_rows(sources: list[str], queries: list[str], per_page: int, pages: int) -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "ids-dataset-crawler/1.0"})
    rows: list[dict] = []
    seen_work_candidates: set[tuple[str, str]] = set()
    disabled_sources: set[str] = set()

    for query in queries:
        print(f"Discovering via: {query}", flush=True)
        for source in sources:
            if source in disabled_sources:
                continue
            try:
                works = source_search(session, source, query, per_page, pages)
            except Exception as exc:
                print(f"  {source} failed, continuing with other sources: {exc}", file=sys.stderr)
                if "rate limited for" in str(exc):
                    disabled_sources.add(source)
                continue

            for work in works:
                candidates = extract_dataset_candidates(work)
                for candidate in candidates:
                    key = (compact_key(candidate), work.source_id or work.doi or work.paper_url or work.title)
                    if key in seen_work_candidates:
                        continue
                    seen_work_candidates.add(key)
                    row = score_candidate(candidate, work)
                    if row:
                        rows.append(row)
    return rows


def publication_sort_key(row: dict) -> tuple:
    return (
        -int(row.get("match_score") or 0),
        -int(row.get("citations") or 0),
        int(row["year"]) if row.get("year") else 9999,
        row.get("paper_title") or "",
    )


def select_best_publications(rows: list[dict], min_citations: int) -> list[dict]:
    best_by_dataset: dict[str, dict] = {}
    for row in rows:
        row = dict(row)
        row["citations"] = int(row.get("citations") or 0)
        row["match_score"] = int(row.get("match_score") or 0)
        if min_citations > 0 and row["citations"] < min_citations:
            continue
        dataset_key = compact_key(row["dataset"])
        current = best_by_dataset.get(dataset_key)
        if current is None or publication_sort_key(row) < publication_sort_key(current):
            best_by_dataset[dataset_key] = row

    selected = list(best_by_dataset.values())
    selected.sort(
        key=lambda row: (
            CATEGORY_ORDER.index(row["category"]) if row["category"] in CATEGORY_ORDER else len(CATEGORY_ORDER),
            int(row["year"]) if row.get("year") else 9999,
            row["dataset"].casefold(),
        )
    )
    return [{column: row.get(column, "") for column in output_columns()} for row in selected]


def output_columns() -> list[str]:
    return [
        "dataset",
        "category",
        "mode",
        "origin",
        "year",
        "citations",
        "paper_title",
        "authors",
        "paper_url",
        "doi",
        "source",
        "source_id",
        "source_query",
        "match_score",
        "match_reason",
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns())
        writer.writeheader()
        writer.writerows(rows)


def print_preview(rows: list[dict], limit: int = 120) -> None:
    columns = ["dataset", "category", "mode", "origin", "year", "citations", "match_score", "paper_title"]
    widths = {column: len(column) for column in columns}
    preview_rows = rows[:limit]
    for row in preview_rows:
        for column in columns:
            widths[column] = min(58, max(widths[column], len(str(row.get(column) or ""))))

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in preview_rows:
        values = []
        for column in columns:
            value = str(row.get(column) or "")
            if len(value) > widths[column]:
                value = value[: widths[column] - 1] + "..."
            values.append(value.ljust(widths[column]))
        print("  ".join(values))


def parse_sources(value: str) -> list[str]:
    sources = [source.strip() for source in value.split(",") if source.strip()]
    valid = {"openalex", "crossref", "semantic_scholar"}
    unknown = sorted(set(sources) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown source(s): {', '.join(unknown)}")
    return sources


def read_query_file(path: Path | None) -> list[str]:
    if not path:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover IDS and attack-detection dataset publications from scholarly APIs."
    )
    parser.add_argument("--out", type=Path, default=OUT, help="CSV output path.")
    parser.add_argument("--min-citations", type=int, default=DEFAULT_MIN_CITATIONS)
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES)
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=list(DEFAULT_SOURCES),
        help="Comma-separated APIs: openalex,crossref,semantic_scholar.",
    )
    parser.add_argument(
        "--extra-query",
        action="append",
        default=[],
        help="Additional discovery query. Can be provided multiple times.",
    )
    parser.add_argument("--query-file", type=Path, help="Optional newline-separated query file.")
    parser.add_argument("--list-queries", action="store_true", help="Print discovery queries and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = list(dict.fromkeys([*DISCOVERY_QUERIES, *read_query_file(args.query_file), *args.extra_query]))

    if args.list_queries:
        for query in queries:
            print(query)
        return

    rows = discover_rows(args.sources, queries, args.per_page, args.pages)
    selected = select_best_publications(rows, args.min_citations)
    if not selected:
        print("No dataset publications found.")
        return

    write_csv(args.out, selected)
    print(f"\nSaved: {args.out} ({len(selected)} datasets)")
    print_preview(selected)


if __name__ == "__main__":
    main()
