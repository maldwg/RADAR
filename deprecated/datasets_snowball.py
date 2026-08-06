#!/usr/bin/env python3
"""Survey and citation-graph based IDS dataset discovery.

This is intentionally a second crawler, separate from datasets.py. It does not
start from arbitrary dataset keywords. Instead it starts from trusted IDS/NIDS
dataset survey papers, extracts dataset-like names from those papers and their
reference neighborhoods, resolves each dataset entity to candidate base
publications, and ranks the most likely canonical publication for each entity.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests


OUT = Path("ids_dataset_snowball.csv")

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_RECOMMEND_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 3
MAX_RETRY_AFTER_SECONDS = 60

DEFAULT_SOURCES = ("openalex", "semantic_scholar")
DEFAULT_PER_PAGE = 25
DEFAULT_MAX_SEEDS = 6
DEFAULT_MAX_REFERENCES_PER_SEED = 80
DEFAULT_MAX_CITATIONS_PER_CANONICAL = 25
DEFAULT_MAX_REFERENCES_PER_CANONICAL = 25
DEFAULT_MAX_RECOMMENDATIONS = 10
DEFAULT_MAX_EXACT_MATCHES = 12
DEFAULT_MIN_SCORE = 0.35

# These seed queries are about trusted survey papers, not about datasets. Users
# can replace or extend them with --seed-title, --seed-doi, or --seed-query.
DEFAULT_SEED_SURVEY_QUERIES = [
    "Survey of intrusion detection systems techniques datasets and challenges Khraisat",
    "A survey of network-based intrusion detection data sets Ring",
    "network intrusion detection datasets survey",
    "intrusion detection datasets survey public datasets",
    "NIDS datasets survey benchmark intrusion detection",
    "IoT intrusion detection datasets survey",
    "industrial control system intrusion detection datasets survey",
]

# Exact dataset-name seeds supplement survey extraction. They are deliberately
# resolved through the same canonical-publication scoring path and can be
# disabled with --no-default-dataset-seeds.
DEFAULT_DATASET_NAME_SEEDS = {
    "FLAME": ("FLAME",),
    "IMPACT": ("IMPACT",),
    "ISCX-UNB": (
        "ISCX-UNB",
        "UNB ISCX",
        "ISCXIDS2012",
        "ISCX IDS 2012",
        "UNB ISCX IDS 2012",
        "ISCXIDS2012 Intrusion Detection Evaluation Dataset",
    ),
    "MAWI": ("MAWI", "MAWILab", "WIDE MAWI", "MAWI traffic archive", "MAWI working group traffic archive"),
    "ID2T": ("ID2T",),
}

# Search-only hints for legacy names whose canonical paper is often indexed
# under a descriptive title instead of under the dataset shorthand.
DEFAULT_DATASET_RESOLUTION_QUERIES = {
    "iscxunb": (
        "Toward developing a systematic approach to generate benchmark datasets for intrusion detection",
    ),
    "iscxids2012": (
        "Toward developing a systematic approach to generate benchmark datasets for intrusion detection",
    ),
}

# Short English-word aliases are too ambiguous to trust when they only appear
# in an abstract. They need stronger title evidence before becoming canonical.
LOW_PRECISION_ALIASES = {
    "flame",
    "impact",
}

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

DATASET_WORDS = (
    "dataset",
    "data set",
    "datasets",
    "data sets",
    "database",
    "databases",
    "benchmark",
    "benchmarks",
    "benchmarking",
    "corpus",
    "testbed",
    "traffic traces",
    "trace dataset",
    "traffic archive",
    "data archive",
    "traffic repository",
    "traffic data",
    "packet traces",
    "flow data",
    "dataset creation toolkit",
)

SURVEY_WORDS = (
    "survey",
    "review",
    "taxonomy",
    "systematic literature",
    "datasets and challenges",
    "data sets",
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
    "toward developing",
    "towards developing",
    "generate",
    "generation of",
    "creation of",
    "create",
    "generated",
    "generating",
    "release",
    "providing",
    "publicly available",
    "updated daily",
    "assist researchers",
    "released",
    "introduce",
    "introduced",
    "introducing",
    "present",
    "presents",
    "we present",
    "we release",
)

CYBERSECURITY_WORDS = (
    "intrusion",
    "attack",
    "cyber",
    "cybersecurity",
    "security",
    "malware",
    "botnet",
    "ddos",
    "dos",
    "phishing",
    "ransomware",
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
    "traffic archive",
    "traffic measurement",
    "backbone traffic",
    "packet trace",
    "packet traces",
    "system call",
    "event log",
    "honeypot",
    "pentesting",
    "penetration testing",
)

DOWNSTREAM_TITLE_WORDS = (
    "survey",
    "review",
    "taxonomy",
    "bibliometric",
    "classification",
    "classifier",
    "feature selection",
    "feature extraction",
    "comparative",
    "comparison",
    "performance analysis",
    "performance evaluation",
    "machine learning approach",
    "deep learning approach",
    "using machine learning",
    "using deep learning",
    "based on",
    "trained on",
    "tested on",
    "evaluated on",
    "applied to",
    "trained using",
    "experiment",
    "experiments",
    "implementation of",
    "analysis of",
    "study of",
    "optimized",
    "optimization",
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
    "HTTP",
    "HTTPS",
    "TCP",
    "UDP",
    "IP",
    "URL",
    "URI",
    "CAN",
    "ML",
    "DL",
    "AI",
    "CNN",
    "RNN",
    "LSTM",
    "SVM",
    "DNN",
    "RF",
    "KNN",
    "ROC",
    "AUC",
    "F1",
    "F1-score",
    "Dataset",
    "Data",
    "Benchmark",
    "Network",
    "Intrusion",
    "Detection",
    "Security",
    "Classification",
    "Evaluation",
    "Table",
    "Figure",
    "Appendix",
}

BAD_DATASET_NAME_KEYS = {item.casefold() for item in BAD_DATASET_NAMES}


class SourceRateLimited(RuntimeError):
    pass


@dataclass
class Publication:
    source: str
    source_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    citations: int = 0
    authors: str = ""
    url: str = ""
    doi: str = ""
    paper_type: str = ""
    references: list[str] = field(default_factory=list)
    pdf_url: str = ""
    relation: str = ""
    seed_title: str = ""

    @property
    def stable_key(self) -> str:
        return self.doi or f"{self.source}:{self.source_id}" or compact_key(self.title)


@dataclass
class DatasetMention:
    raw_name: str
    source_kind: str
    source_title: str
    source_id: str
    survey_title: str
    context: str


@dataclass
class ScoredPublication:
    publication: Publication
    score: float
    components: dict[str, float]
    reason: str


@dataclass
class DatasetEntity:
    normalized_name: str
    aliases: set[str] = field(default_factory=set)
    mentions: list[DatasetMention] = field(default_factory=list)
    candidate_publications: dict[str, Publication] = field(default_factory=dict)
    candidate_query_aliases: dict[str, set[str]] = field(default_factory=dict)
    resolution_queries: set[str] = field(default_factory=set)
    canonical_publication: ScoredPublication | None = None
    category: str = "Unknown"
    release_year: int | None = None

    def add_alias(self, name: str) -> None:
        name = canonical_dataset_name(name)
        if name:
            self.aliases.add(name)

    def add_publication(self, publication: Publication) -> None:
        self.candidate_publications[publication.stable_key] = publication

    def add_candidate_query_alias(self, publication: Publication, alias: str) -> None:
        if alias_is_high_precision(alias):
            self.candidate_query_aliases.setdefault(publication.stable_key, set()).add(alias)

    def add_resolution_query(self, query: str) -> None:
        query = normalize_text(query)
        if query:
            self.resolution_queries.add(query)

    @property
    def display_name(self) -> str:
        if not self.aliases:
            return self.normalized_name
        return sorted(self.aliases, key=lambda item: (len(item), item.casefold()))[0]

    @property
    def survey_sources(self) -> set[str]:
        return {mention.survey_title for mention in self.mentions if mention.survey_title}


def normalize_text(text: str | None) -> str:
    text = str(text or "")
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def comparable(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text).casefold()).strip()


def compact_key(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text).casefold())


def has_phrase(text: str | None, phrases: Iterable[str]) -> bool:
    normalized = f" {comparable(text)} "
    return any(f" {comparable(phrase)} " in normalized for phrase in phrases if comparable(phrase))


def canonical_dataset_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"\b(dataset|data set|benchmark|corpus|testbed|traffic traces)\b$", "", name, flags=re.I)
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)
    name = name.strip(" ,.;:()[]{}\"'")
    name = re.sub(r"\s*-\s*", "-", name)
    name = re.sub(r"\s+", " ", name)
    return name


def dataset_entity_key(name: str) -> str:
    name = canonical_dataset_name(name)
    key = compact_key(name)
    key = re.sub(r"(dataset|benchmark|corpus|testbed)$", "", key)
    return key


def default_seed_group_for_key(key: str) -> tuple[str, tuple[str, ...]] | None:
    for seed_name, aliases in DEFAULT_DATASET_NAME_SEEDS.items():
        names = (seed_name, *aliases)
        if any(dataset_entity_key(name) == key for name in names):
            return seed_name, aliases
    return None


def default_aliases_for_dataset_key(key: str) -> tuple[str, ...]:
    group = default_seed_group_for_key(key)
    if not group:
        return ()
    seed_name, aliases = group
    return tuple(dict.fromkeys((seed_name, *aliases)))


def default_resolution_queries_for_dataset_key(key: str) -> tuple[str, ...]:
    group = default_seed_group_for_key(key)
    if not group:
        return DEFAULT_DATASET_RESOLUTION_QUERIES.get(key, ())
    seed_name, aliases = group
    queries: list[str] = []
    for name in (seed_name, *aliases):
        queries.extend(DEFAULT_DATASET_RESOLUTION_QUERIES.get(dataset_entity_key(name), ()))
    return tuple(dict.fromkeys(queries))


def looks_like_dataset_name(name: str) -> bool:
    name = canonical_dataset_name(name)
    if len(name) < 3 or len(name) > 80:
        return False
    if name in BAD_DATASET_NAMES or name.casefold() in BAD_DATASET_NAME_KEYS:
        return False
    if re.fullmatch(r"\d{2,4}", name):
        return False
    if re.search(r"\b(score|accuracy|precision|recall|auc|roc|method|model|framework)\b", name, flags=re.I):
        return False
    if re.search(r"\b(based|enabled|oriented|specific|defined)\b$", name, flags=re.I):
        return False

    has_digit = bool(re.search(r"\d", name))
    has_separator = bool(re.search(r"[-_/+.' ]", name))
    has_mixed_caps = bool(re.search(r"[A-Z]", name)) and bool(re.search(r"[a-z]", name))
    all_caps = bool(re.fullmatch(r"[A-Z0-9][A-Z0-9_./'+-]{2,}", name))
    ends_with_set = name.lower().endswith("set")
    return has_digit or has_separator or has_mixed_caps or all_caps or ends_with_set


def reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    return " ".join(word for _, word in sorted(positions))


def request_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    source: str = "api",
    headers: dict | None = None,
) -> dict:
    backoff = 1.0
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                try:
                    wait = int(float(response.headers.get("Retry-After") or backoff))
                except ValueError:
                    wait = int(backoff)
                if wait > MAX_RETRY_AFTER_SECONDS:
                    raise SourceRateLimited(f"{source} rate limited for {wait}s")
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
        except SourceRateLimited:
            raise
        except requests.RequestException as exc:
            if attempt >= REQUEST_RETRIES:
                raise RuntimeError(f"{source} request failed for {url}: {exc}") from exc
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_RETRY_AFTER_SECONDS)
    raise RuntimeError(f"{source} request failed for {url}")


def openalex_authors(item: dict) -> str:
    authors = []
    for authorship in (item.get("authorships") or [])[:6]:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            authors.append(author["display_name"])
    return ", ".join(authors)


def publication_from_openalex(item: dict, relation: str = "", seed_title: str = "") -> Publication:
    title = item.get("title") or item.get("display_name") or ""
    location = item.get("primary_location") or {}
    best_oa = item.get("best_oa_location") or {}
    doi = item.get("doi") or ""
    return Publication(
        source="openalex",
        source_id=item.get("id") or "",
        title=normalize_text(title),
        abstract=normalize_text(reconstruct_openalex_abstract(item.get("abstract_inverted_index"))),
        year=item.get("publication_year"),
        citations=int(item.get("cited_by_count") or 0),
        authors=openalex_authors(item),
        url=location.get("landing_page_url") or item.get("doi") or item.get("id") or "",
        doi=doi,
        paper_type=item.get("type") or item.get("type_crossref") or "",
        references=list(item.get("referenced_works") or []),
        pdf_url=best_oa.get("pdf_url") or location.get("pdf_url") or "",
        relation=relation,
        seed_title=seed_title,
    )


def openalex_select_fields() -> str:
    return ",".join(
        [
            "id",
            "doi",
            "title",
            "display_name",
            "publication_year",
            "cited_by_count",
            "abstract_inverted_index",
            "primary_location",
            "best_oa_location",
            "authorships",
            "referenced_works",
            "type",
            "type_crossref",
        ]
    )


def openalex_search(
    session: requests.Session,
    query: str,
    per_page: int,
    pages: int,
    relation: str,
    seed_title: str = "",
) -> list[Publication]:
    publications = []
    for page in range(1, pages + 1):
        payload = request_json(
            session,
            OPENALEX_WORKS_URL,
            {
                "search": query,
                "per-page": per_page,
                "page": page,
                "select": openalex_select_fields(),
            },
            "OpenAlex",
        )
        for item in payload.get("results") or []:
            publications.append(publication_from_openalex(item, relation=relation, seed_title=seed_title))
        time.sleep(0.2)
    return publications


def openalex_fetch_work(
    session: requests.Session,
    work_id: str,
    relation: str,
    seed_title: str = "",
) -> Publication | None:
    if not work_id:
        return None
    if work_id.startswith("https://openalex.org/"):
        url = work_id
    elif work_id.startswith("W"):
        url = f"https://api.openalex.org/works/{work_id}"
    else:
        url = work_id
    if url.startswith("https://openalex.org/"):
        url = url.replace("https://openalex.org/", "https://api.openalex.org/works/")
    try:
        payload = request_json(session, url, {"select": openalex_select_fields()}, "OpenAlex")
    except Exception as exc:
        print(f"  OpenAlex work fetch skipped: {exc}", file=sys.stderr)
        return None
    return publication_from_openalex(payload, relation=relation, seed_title=seed_title)


def openalex_fetch_by_doi(session: requests.Session, doi: str, relation: str) -> list[Publication]:
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/")
    payload = request_json(
        session,
        OPENALEX_WORKS_URL,
        {
            "filter": f"doi:{doi}",
            "per-page": 5,
            "select": openalex_select_fields(),
        },
        "OpenAlex",
    )
    return [publication_from_openalex(item, relation=relation) for item in payload.get("results") or []]


def openalex_citations(
    session: requests.Session,
    publication: Publication,
    limit: int,
) -> list[Publication]:
    if not publication.source_id or limit <= 0:
        return []
    work_id = publication.source_id.rsplit("/", 1)[-1]
    try:
        payload = request_json(
            session,
            OPENALEX_WORKS_URL,
            {
                "filter": f"cites:{work_id}",
                "sort": "cited_by_count:desc",
                "per-page": min(limit, 200),
                "select": openalex_select_fields(),
            },
            "OpenAlex",
        )
    except Exception as exc:
        print(f"  OpenAlex citations skipped for {publication.title[:60]}: {exc}", file=sys.stderr)
        return []
    return [
        publication_from_openalex(item, relation="canonical_citation", seed_title=publication.title)
        for item in payload.get("results") or []
    ][:limit]


def semantic_headers() -> dict:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": api_key} if api_key else {}


def publication_from_semantic(item: dict, relation: str = "", seed_title: str = "") -> Publication:
    external = item.get("externalIds") or {}
    doi = external.get("DOI") or ""
    authors = ", ".join((author.get("name") or "") for author in (item.get("authors") or [])[:6])
    references = [
        ref.get("paperId")
        for ref in item.get("references") or []
        if ref and ref.get("paperId")
    ]
    return Publication(
        source="semantic_scholar",
        source_id=item.get("paperId") or doi,
        title=normalize_text(item.get("title") or ""),
        abstract=normalize_text(item.get("abstract") or ""),
        year=item.get("year"),
        citations=int(item.get("citationCount") or 0),
        authors=authors,
        url=item.get("url") or (f"https://doi.org/{doi}" if doi else ""),
        doi=f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
        paper_type=item.get("publicationTypes") or "",
        references=references,
        relation=relation,
        seed_title=seed_title,
    )


def semantic_search(
    session: requests.Session,
    query: str,
    per_page: int,
    pages: int,
    relation: str,
    seed_title: str = "",
) -> list[Publication]:
    publications = []
    fields = "title,abstract,year,citationCount,authors,url,externalIds,publicationTypes,references.paperId"
    for page in range(pages):
        payload = request_json(
            session,
            SEMANTIC_SEARCH_URL,
            {
                "query": query,
                "limit": min(per_page, 100),
                "offset": page * per_page,
                "fields": fields,
            },
            "Semantic Scholar",
            headers=semantic_headers(),
        )
        for item in payload.get("data") or []:
            publications.append(publication_from_semantic(item, relation=relation, seed_title=seed_title))
        time.sleep(0.4 if semantic_headers() else 1.1)
    return publications


def semantic_fetch_paper(
    session: requests.Session,
    paper_id: str,
    relation: str,
    seed_title: str = "",
) -> Publication | None:
    fields = "title,abstract,year,citationCount,authors,url,externalIds,publicationTypes,references.paperId"
    try:
        payload = request_json(
            session,
            f"{SEMANTIC_PAPER_URL}/{paper_id}",
            {"fields": fields},
            "Semantic Scholar",
            headers=semantic_headers(),
        )
    except Exception as exc:
        print(f"  Semantic Scholar paper fetch skipped: {exc}", file=sys.stderr)
        return None
    return publication_from_semantic(payload, relation=relation, seed_title=seed_title)


def semantic_paper_endpoint(
    session: requests.Session,
    paper_id: str,
    suffix: str,
    limit: int,
    relation: str,
    seed_title: str = "",
) -> list[Publication]:
    if not paper_id or limit <= 0:
        return []
    fields = "title,abstract,year,citationCount,authors,url,externalIds,publicationTypes,references.paperId"
    try:
        payload = request_json(
            session,
            f"{SEMANTIC_PAPER_URL}/{paper_id}/{suffix}",
            {"limit": min(limit, 100), "fields": fields},
            "Semantic Scholar",
            headers=semantic_headers(),
        )
    except Exception as exc:
        print(f"  Semantic Scholar {suffix} skipped: {exc}", file=sys.stderr)
        return []
    publications = []
    for item in payload.get("data") or []:
        paper = item.get("citingPaper") or item.get("citedPaper") or item
        if paper:
            publications.append(publication_from_semantic(paper, relation=relation, seed_title=seed_title))
    return publications[:limit]


def semantic_recommendations(
    session: requests.Session,
    publication: Publication,
    limit: int,
) -> list[Publication]:
    if not publication.source_id or limit <= 0:
        return []
    try:
        payload = request_json(
            session,
            f"{SEMANTIC_RECOMMEND_URL}/{publication.source_id}",
            {
                "limit": min(limit, 100),
                "fields": "title,abstract,year,citationCount,authors,url,externalIds,publicationTypes,references.paperId",
            },
            "Semantic Scholar",
            headers=semantic_headers(),
        )
    except Exception as exc:
        print(f"  Semantic Scholar recommendations skipped: {exc}", file=sys.stderr)
        return []
    return [
        publication_from_semantic(item, relation="recommended", seed_title=publication.title)
        for item in payload.get("recommendedPapers") or []
    ]


def source_search(
    session: requests.Session,
    source: str,
    query: str,
    per_page: int,
    pages: int,
    relation: str,
    seed_title: str = "",
) -> list[Publication]:
    if source == "openalex":
        return openalex_search(session, query, per_page, pages, relation=relation, seed_title=seed_title)
    if source == "semantic_scholar":
        return semantic_search(session, query, per_page, pages, relation=relation, seed_title=seed_title)
    raise ValueError(f"Unsupported source: {source}")


def looks_like_survey_publication(publication: Publication) -> bool:
    title = publication.title
    if not has_phrase(title, ("intrusion", "ids", "nids", "cyber", "network security", "iot security")):
        return False
    if has_phrase(title, SURVEY_WORDS):
        return True
    if publication.citations >= 100 and has_phrase(f"{title} {publication.abstract[:800]}", SURVEY_WORDS):
        return True
    return False


def resolve_seed_surveys(
    session: requests.Session,
    sources: list[str],
    seed_queries: list[str],
    seed_titles: list[str],
    seed_dois: list[str],
    max_seeds: int,
    per_page: int,
) -> list[Publication]:
    if max_seeds <= 0:
        return []

    seeds: dict[str, Publication] = {}
    disabled_sources: set[str] = set()

    for doi in seed_dois:
        if "openalex" not in disabled_sources and "openalex" in sources:
            try:
                for publication in openalex_fetch_by_doi(session, doi, relation="seed"):
                    seeds[publication.stable_key] = publication
            except SourceRateLimited as exc:
                print(f"OpenAlex disabled: {exc}", file=sys.stderr)
                disabled_sources.add("openalex")
            except Exception as exc:
                print(f"Seed DOI skipped ({doi}): {exc}", file=sys.stderr)
        if "semantic_scholar" not in disabled_sources and "semantic_scholar" in sources:
            doi_id = doi.strip().removeprefix("https://doi.org/")
            publication = semantic_fetch_paper(session, f"DOI:{doi_id}", relation="seed")
            if publication:
                seeds[publication.stable_key] = publication

    for query in [*seed_titles, *seed_queries]:
        print(f"Seed survey search: {query}", flush=True)
        for source in sources:
            if source in disabled_sources:
                continue
            try:
                publications = source_search(
                    session,
                    source,
                    query,
                    per_page=per_page,
                    pages=1,
                    relation="seed",
                )
            except SourceRateLimited as exc:
                print(f"{source} disabled: {exc}", file=sys.stderr)
                disabled_sources.add(source)
                continue
            except Exception as exc:
                print(f"  seed search failed on {source}: {exc}", file=sys.stderr)
                continue

            for publication in publications:
                if looks_like_survey_publication(publication):
                    seeds[publication.stable_key] = publication

    ordered = sorted(seeds.values(), key=lambda pub: (-pub.citations, pub.year or 9999, pub.title))
    return ordered[:max_seeds]


NAME_EXPR = (
    r"[A-Z][A-Za-z0-9]*(?:[-_./'+][A-Za-z0-9]+)+"
    r"|[A-Z]{2,}[A-Z0-9_./'+-]*\d[A-Z0-9_./'+-]*"
    r"|[A-Z][A-Za-z]+(?:[A-Z][A-Za-z0-9]+)+(?:\d{2,4})?"
)
DATASET_EXPR = r"(?:dataset|data set|benchmark|corpus|testbed|traffic traces|trace dataset)"


def extract_dataset_names(text: str) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    candidates: set[str] = set()
    patterns = [
        rf"^\s*({NAME_EXPR})\s*(?::|\s+-\s+).{{0,140}}\b{DATASET_EXPR}\b",
        rf"\b({NAME_EXPR})\b\s+(?:{DATASET_EXPR})\b",
        rf"\b(?:{DATASET_EXPR})\s+(?:called|named|known as|referred to as)\s+(?:the\s+)?({NAME_EXPR})\b",
        rf"\b(?:called|named|known as|referred to as)\s+(?:the\s+)?({NAME_EXPR})\b.{{0,80}}\b{DATASET_EXPR}\b",
        rf"[:;]\s*({NAME_EXPR})\s+(?:{DATASET_EXPR})\b",
        rf"\b({NAME_EXPR})\b.{{0,80}}\b(?:public|labeled|labelled|annotated|realistic|real-world)\s+{DATASET_EXPR}\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.I):
            candidate = canonical_dataset_name(match if isinstance(match, str) else match[0])
            if looks_like_dataset_name(candidate):
                candidates.add(candidate)
    return candidates


def short_context(text: str, name: str, width: int = 130) -> str:
    text = normalize_text(text)
    index = compact_key(text).find(compact_key(name))
    if index < 0:
        return text[:width]
    plain_index = text.casefold().find(name.casefold())
    if plain_index < 0:
        return text[:width]
    start = max(0, plain_index - width // 2)
    end = min(len(text), plain_index + len(name) + width // 2)
    return text[start:end].strip()


def fetch_seed_fulltext(session: requests.Session, publication: Publication, max_chars: int) -> str:
    if not publication.pdf_url or max_chars <= 0:
        return ""
    try:
        response = session.get(publication.pdf_url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  full text skipped for {publication.title[:60]}: {exc}", file=sys.stderr)
        return ""

    content_type = response.headers.get("content-type", "").casefold()
    body = response.content
    if "pdf" not in content_type and not publication.pdf_url.lower().endswith(".pdf"):
        return normalize_text(response.text)[:max_chars]

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
            print("  pdftotext unavailable or failed; full-text seed parsing skipped", file=sys.stderr)
            return ""
        return normalize_text(txt_path.read_text(encoding="utf-8", errors="ignore"))[:max_chars]


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
    if any(term in name_title for term in ("iot", "iomt", "internet of things", "smart agriculture")):
        return "IoT"
    if any(term in name_title for term in ("wireless", "wifi", "wi-fi", "802.11", "wsn", "uav")):
        return "Wireless"
    if any(term in text for term in ("scada", "industrial control", "water treatment", "water distribution", "swat", "wadi")):
        return "ICS/SCADA"
    if has_phrase(text, ("ics",)):
        return "ICS/SCADA"
    if "mqtt" in text:
        return "MQTT"
    if any(term in text for term in ("dns", "doh", "exfiltration", "tunnel")):
        return "DNS"
    if "iot" in text:
        return "IoT"
    if any(term in text for term in ("wireless", "wifi", "wi-fi", "802.11", "wireless sensor", "uav")):
        return "Wireless"
    if any(term in text for term in ("host", "system call", "event log", "windows event")):
        return "Host"
    if any(term in text for term in ("web attack", "web application", "sql injection", "xss")):
        return "Web/App"
    if any(term in text for term in ("cloud", "container", "kubernetes", "open radio access network", "oran")):
        return "Cloud"
    if any(term in text for term in ("can bus", "vehicular", "vehicle", "automotive")):
        return "Automotive/CAN"
    if any(term in text for term in ("botnet", "malware", "ransomware")):
        return "Malware/Botnet"
    if has_phrase(
        text,
        (
            "intrusion",
            "attack",
            "network security",
            "network traffic",
            "internet traffic",
            "traffic monitoring",
            "traffic traces",
            "packet traces",
            "anomaly detector",
            "anomaly detection",
            "ddos",
            "dos",
        ),
    ):
        return "Enterprise"
    return "Unknown"


def infer_release_year(name: str, publications: Iterable[Publication]) -> int | None:
    publications = list(publications)

    name_years: list[int] = []
    for year in re.findall(r"\b(19[8-9]\d|20[0-3]\d)\b", name or ""):
        name_years.append(int(year))
    for short in re.findall(r"(?<!\d)(9[89]|0[0-9]|1[0-9]|2[0-9])(?!\d)", name or ""):
        value = int(short)
        name_years.append(1900 + value if value >= 90 else 2000 + value)
    if name_years:
        return min(year for year in name_years if 1980 <= year <= 2035)

    candidate_years: list[int] = [pub.year for pub in publications if pub.year]
    for title in (pub.title for pub in publications):
        for year in re.findall(r"\b(19[8-9]\d|20[0-3]\d)\b", title or ""):
            candidate_years.append(int(year))
        for short in re.findall(r"(?<!\d)(9[89]|0[0-9]|1[0-9]|2[0-9])(?!\d)", title or ""):
            value = int(short)
            candidate_years.append(1900 + value if value >= 90 else 2000 + value)
    candidate_years = [year for year in candidate_years if 1980 <= year <= 2035]
    return min(candidate_years) if candidate_years else None


def aliases_in_text(entity: DatasetEntity, text: str) -> list[str]:
    text = normalize_text(text)
    matched: list[str] = []
    for alias in entity.aliases:
        alias = normalize_text(alias)
        alias_key = compact_key(alias)
        if not alias_key:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text, flags=re.I):
            matched.append(alias)
            continue
        if len(alias_key) >= 6 and alias_key not in LOW_PRECISION_ALIASES and alias_key in compact_key(text):
            matched.append(alias)
    return matched


def alias_is_high_precision(alias: str) -> bool:
    raw = alias.strip()
    alias_key = compact_key(raw)
    if not alias_key or alias_key in LOW_PRECISION_ALIASES:
        return False
    if any(char.isdigit() for char in raw):
        return True
    if any(char in raw for char in "-_/ "):
        return len(alias_key) >= 4
    if len(alias_key) >= 6:
        return True
    return raw.isupper() and len(alias_key) >= 4


def high_precision_aliases_in_text(entity: DatasetEntity, text: str) -> list[str]:
    return [alias for alias in aliases_in_text(entity, text) if alias_is_high_precision(alias)]


def high_precision_alias_evidence(entity: DatasetEntity, publication: Publication) -> set[str]:
    aliases = set(high_precision_aliases_in_text(entity, publication.abstract[:1800]))
    aliases.update(entity.candidate_query_aliases.get(publication.stable_key, set()))
    return aliases


def title_starts_with_alias(entity: DatasetEntity, title: str) -> bool:
    title_norm = normalize_text(title)
    for alias in sorted(entity.aliases, key=len, reverse=True):
        alias_pattern = re.escape(normalize_text(alias))
        if comparable(title_norm) == comparable(alias):
            return True
        if re.search(rf"^\s*{alias_pattern}\b\s*(?::|-|,|\()", title_norm, flags=re.I):
            return True
        if re.search(
            rf"^\s*{alias_pattern}\b\s+(?:dataset|data set|benchmark|corpus|testbed|traffic archive)\b",
            title_norm,
            flags=re.I,
        ):
            return True
    return False


def title_has_primary_dataset_wording(title: str) -> bool:
    if not has_phrase(title, DATASET_WORDS):
        return False
    if has_phrase(title, INTRODUCTION_WORDS):
        return True
    if has_phrase(title, ("anomaly labeling", "performance benchmarking")):
        return True
    title_cmp = comparable(title)
    return bool(
        re.search(
            r"\b(?:new|realistic|real world|realistic|labeled|labelled|annotated|comprehensive|public)\b"
            r".{0,80}\b(?:dataset|data set|benchmark|corpus|testbed|traffic archive)\b",
            title_cmp,
        )
        or re.search(
            r"\b(?:dataset|data set|benchmark|corpus|testbed|traffic archive)\b.{0,90}"
            r"\b(?:for|of|to support|for evaluating|for evaluation)\b",
            title_cmp,
        )
    )


def title_looks_like_downstream_use(entity: DatasetEntity, publication: Publication) -> bool:
    title = publication.title
    title_cmp = comparable(title)
    has_alias = bool(aliases_in_text(entity, title))

    if not has_alias:
        return False
    if title_starts_with_alias(entity, title) and title_has_primary_dataset_wording(title):
        return False
    if (
        title_has_primary_dataset_wording(title)
        and has_phrase(title, INTRODUCTION_WORDS)
        and not re.search(r"\b(?:from|using|based on|trained on|evaluated on|tested on)\b", title_cmp)
    ):
        return False
    if has_phrase(title, ("engine", "framework", "platform", "system")) and not title_has_primary_dataset_wording(title):
        return True

    downstream_near_alias = (
        r"\b(using|use of|with|from|on|based on|trained on|trained using|tested on|evaluated on|"
        r"applied to|implementation of|analysis of|study of|classification of|detecting|detection using)\b"
        r".{0,100}\b(dataset|data set|benchmark)?\b"
    )
    if re.search(downstream_near_alias, title_cmp):
        return True
    if re.search(
        r"\b(?:classification|classifier|model|algorithm|feature selection|feature extraction|"
        r"machine learning|deep learning|neural network|ensemble|optimization|optimized|"
        r"performance|evaluation|comparison|comparative|prediction)\b",
        title_cmp,
    ):
        return True
    return False


def is_probable_dataset_publication(entity: DatasetEntity, publication: Publication) -> bool:
    title = publication.title
    text = f"{publication.title} {publication.abstract[:800]}"
    if not title:
        return False
    if has_phrase(title, SURVEY_WORDS):
        return False
    if title_looks_like_downstream_use(entity, publication):
        return False
    if has_phrase(title, DOWNSTREAM_TITLE_WORDS) and not title_has_primary_dataset_wording(title):
        return False
    if not has_phrase(text, CYBERSECURITY_WORDS):
        return False
    exact_title = bool(aliases_in_text(entity, title))
    exact_abstract = bool(high_precision_alias_evidence(entity, publication))
    if title_starts_with_alias(entity, title) and title_has_primary_dataset_wording(title):
        return True
    if exact_title and title_has_primary_dataset_wording(publication.abstract[:1800]):
        return True
    if title_has_primary_dataset_wording(title) and (exact_title or exact_abstract):
        return True
    if exact_title and has_phrase(title, ("testbed", "traffic archive", "packet traces", "dataset creation toolkit")):
        return True
    return False


def official_link_score(publication: Publication) -> float:
    if publication.doi:
        return 0.5
    if "arxiv.org" in publication.url:
        return 0.4
    if publication.url and "doi.org" not in publication.url:
        return 0.5
    return 0.0


def venue_quality_score(publication: Publication) -> float:
    if publication.doi and publication.citations >= 50:
        return 1.0
    if publication.doi:
        return 0.7
    if "arxiv.org" in publication.url:
        return 0.4
    return 0.2


def score_publication(
    entity: DatasetEntity,
    publication: Publication,
    max_citations: int,
) -> ScoredPublication | None:
    if not is_probable_dataset_publication(entity, publication):
        return None

    exact_in_title = 1.0 if aliases_in_text(entity, publication.title) else 0.0
    exact_in_abstract = 1.0 if high_precision_alias_evidence(entity, publication) else 0.0
    dataset_title = 1.0 if has_phrase(publication.title, DATASET_WORDS) else 0.0
    official = official_link_score(publication)
    multi_survey = min(1.0, len(entity.survey_sources) / 3)
    venue = venue_quality_score(publication)

    citation_norm = 0.0
    if max_citations > 0:
        citation_norm = math.log(publication.citations + 1) / math.log(max_citations + 1)

    release_year = entity.release_year or infer_release_year(entity.display_name, [publication])
    year_proximity = 0.0
    if release_year and publication.year:
        year_proximity = max(0.0, 1.0 - min(abs(publication.year - release_year), 8) / 8)

    components = {
        "citation": citation_norm,
        "exact_title": exact_in_title,
        "dataset_title": dataset_title,
        "official_link": official,
        "multi_survey": multi_survey,
        "venue": venue,
        "year_proximity": year_proximity,
        "exact_abstract": 0.5 * exact_in_abstract,
    }
    score = (
        0.40 * components["citation"]
        + 0.20 * components["exact_title"]
        + 0.15 * components["dataset_title"]
        + 0.10 * components["official_link"]
        + 0.10 * components["multi_survey"]
        + 0.05 * components["venue"]
        + 0.05 * components["year_proximity"]
        + 0.05 * components["exact_abstract"]
    )
    reason = ", ".join(
        key for key, value in components.items() if value > 0
    )
    return ScoredPublication(publication=publication, score=round(score, 4), components=components, reason=reason)


def add_mention(
    entities: dict[str, DatasetEntity],
    name: str,
    source_kind: str,
    source_publication: Publication,
    survey_title: str,
    context: str,
) -> DatasetEntity | None:
    name = canonical_dataset_name(name)
    if not looks_like_dataset_name(name):
        return None
    key = dataset_entity_key(name)
    if not key:
        return None
    entity = entities.setdefault(key, DatasetEntity(normalized_name=key))
    entity.add_alias(name)
    entity.mentions.append(
        DatasetMention(
            raw_name=name,
            source_kind=source_kind,
            source_title=source_publication.title,
            source_id=source_publication.source_id,
            survey_title=survey_title,
            context=context,
        )
    )
    return entity


def add_dataset_name_seed(
    entities: dict[str, DatasetEntity],
    name: str,
    aliases: Iterable[str] = (),
    source_label: str = "trusted_dataset_seed",
) -> DatasetEntity | None:
    name = canonical_dataset_name(name)
    if not looks_like_dataset_name(name):
        return None
    key = dataset_entity_key(name)
    entity = entities.setdefault(key, DatasetEntity(normalized_name=key))
    for query in default_resolution_queries_for_dataset_key(key):
        entity.add_resolution_query(query)
    aliases = tuple(dict.fromkeys((*default_aliases_for_dataset_key(key), *aliases)))
    synthetic = Publication(
        source=source_label,
        source_id=key,
        title=f"Trusted dataset seed: {name}",
        relation=source_label,
    )
    for alias in [name, *aliases]:
        alias = canonical_dataset_name(alias)
        if looks_like_dataset_name(alias):
            entity.add_alias(alias)
            entity.mentions.append(
                DatasetMention(
                    raw_name=alias,
                    source_kind=source_label,
                    source_title=synthetic.title,
                    source_id=synthetic.source_id,
                    survey_title=source_label,
                    context=f"Exact dataset-name seed: {alias}",
                )
            )
    return entity


def add_default_dataset_name_seeds(
    entities: dict[str, DatasetEntity],
    include_defaults: bool,
    user_names: Iterable[str],
) -> None:
    if include_defaults:
        for name, aliases in DEFAULT_DATASET_NAME_SEEDS.items():
            add_dataset_name_seed(entities, name, aliases)
    for name in user_names:
        add_dataset_name_seed(entities, name, ())


def add_mentions_from_publication(
    entities: dict[str, DatasetEntity],
    publication: Publication,
    source_kind: str,
    survey_title: str,
    include_abstract: bool = True,
) -> None:
    text = publication.title
    if include_abstract:
        text = f"{publication.title}. {publication.abstract[:2500]}"
    for name in extract_dataset_names(text):
        entity = add_mention(
            entities,
            name,
            source_kind=source_kind,
            source_publication=publication,
            survey_title=survey_title,
            context=short_context(text, name),
        )
        if entity:
            entity.add_publication(publication)


def collect_seed_neighborhood(
    session: requests.Session,
    sources: list[str],
    seeds: list[Publication],
    entities: dict[str, DatasetEntity],
    args: argparse.Namespace,
) -> None:
    for seed in seeds:
        print(f"Seed: {seed.title} ({seed.year or 'n.d.'}, {seed.citations} cites)", flush=True)
        add_mentions_from_publication(entities, seed, "seed_metadata", seed.title)

        if args.fetch_fulltext:
            fulltext = fetch_seed_fulltext(session, seed, args.max_fulltext_chars)
            if fulltext:
                for name in extract_dataset_names(fulltext):
                    add_mention(entities, name, "seed_fulltext", seed, seed.title, short_context(fulltext, name))

        if seed.source == "openalex":
            for reference_id in seed.references[: args.max_references_per_seed]:
                reference = openalex_fetch_work(
                    session,
                    reference_id,
                    relation="seed_reference",
                    seed_title=seed.title,
                )
                if reference:
                    add_mentions_from_publication(entities, reference, "seed_reference", seed.title)
        elif seed.source == "semantic_scholar":
            refs = semantic_paper_endpoint(
                session,
                seed.source_id,
                "references",
                args.max_references_per_seed,
                relation="seed_reference",
                seed_title=seed.title,
            )
            for reference in refs:
                add_mentions_from_publication(entities, reference, "seed_reference", seed.title)


def resolve_candidate_publications(
    session: requests.Session,
    sources: list[str],
    entities: dict[str, DatasetEntity],
    args: argparse.Namespace,
) -> None:
    disabled_sources: set[str] = set()
    for entity in sorted(entities.values(), key=lambda item: item.display_name.casefold()):
        aliases = sorted(entity.aliases, key=lambda value: (len(value), value.casefold()))[: args.max_alias_searches]
        query_terms: list[tuple[str, str]] = []
        for alias in aliases:
            for query in (
                alias,
                f"{alias} dataset",
                f"{alias} intrusion detection dataset",
                f"{alias} network traffic",
            ):
                query_terms.append((query, alias))
        for query in sorted(entity.resolution_queries):
            query_terms.append((query, entity.display_name))

        seen_queries: set[str] = set()
        for query, evidence_alias in query_terms:
            query_key = comparable(query)
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            for source in sources:
                if source in disabled_sources:
                    continue
                try:
                    publications = source_search(
                        session,
                        source,
                        query,
                        per_page=args.max_exact_matches,
                        pages=1,
                        relation="exact_name_search",
                        seed_title=entity.display_name,
                    )
                except SourceRateLimited as exc:
                    print(f"{source} disabled during exact search: {exc}", file=sys.stderr)
                    disabled_sources.add(source)
                    continue
                except Exception as exc:
                    print(f"  exact search failed for {query} on {source}: {exc}", file=sys.stderr)
                    continue
                for publication in publications:
                    entity.add_publication(publication)
                    entity.add_candidate_query_alias(publication, evidence_alias)


def choose_canonical_publications(entities: dict[str, DatasetEntity], min_score: float) -> None:
    max_citations = max(
        [pub.citations for entity in entities.values() for pub in entity.candidate_publications.values()] or [1]
    )
    for entity in entities.values():
        entity.release_year = infer_release_year(entity.display_name, entity.candidate_publications.values())
        scored = [
            score_publication(entity, publication, max_citations)
            for publication in entity.candidate_publications.values()
        ]
        scored = [item for item in scored if item and item.score >= min_score]
        if not scored:
            continue
        scored.sort(key=lambda item: (-item.score, -item.publication.citations, item.publication.year or 9999))
        entity.canonical_publication = scored[0]
        pub = scored[0].publication
        entity.release_year = infer_release_year(entity.display_name, [pub])
        entity.category = infer_category(entity.display_name, pub.title, pub.abstract)


def snowball_from_canonical(
    session: requests.Session,
    sources: list[str],
    entities: dict[str, DatasetEntity],
    args: argparse.Namespace,
) -> None:
    canonical_entities = [
        entity
        for entity in entities.values()
        if entity.canonical_publication is not None
    ][: args.max_snowball_entities]

    for entity in canonical_entities:
        canonical = entity.canonical_publication.publication
        neighborhood: list[Publication] = []
        print(f"Snowball: {entity.display_name}", flush=True)

        if canonical.source == "openalex":
            for reference_id in canonical.references[: args.max_references_per_canonical]:
                reference = openalex_fetch_work(
                    session,
                    reference_id,
                    relation="canonical_reference",
                    seed_title=canonical.title,
                )
                if reference:
                    neighborhood.append(reference)
            neighborhood.extend(openalex_citations(session, canonical, args.max_citations_per_canonical))

        if canonical.source == "semantic_scholar":
            neighborhood.extend(
                semantic_paper_endpoint(
                    session,
                    canonical.source_id,
                    "references",
                    args.max_references_per_canonical,
                    relation="canonical_reference",
                    seed_title=canonical.title,
                )
            )
            neighborhood.extend(
                semantic_paper_endpoint(
                    session,
                    canonical.source_id,
                    "citations",
                    args.max_citations_per_canonical,
                    relation="canonical_citation",
                    seed_title=canonical.title,
                )
            )
            neighborhood.extend(semantic_recommendations(session, canonical, args.max_recommendations))

        for publication in neighborhood:
            add_mentions_from_publication(entities, publication, publication.relation or "snowball", canonical.title)


def entity_to_row(entity: DatasetEntity) -> dict:
    canonical = entity.canonical_publication
    publication = canonical.publication if canonical else Publication(source="", source_id="", title="")
    mention_sources = sorted({mention.source_kind for mention in entity.mentions})
    survey_sources = sorted(entity.survey_sources)
    aliases = sorted(entity.aliases, key=str.casefold)
    candidates = sorted(
        entity.candidate_publications.values(),
        key=lambda pub: (-pub.citations, pub.year or 9999, pub.title),
    )
    return {
        "dataset": entity.display_name,
        "normalized_name": entity.normalized_name,
        "aliases": "; ".join(aliases),
        "category": entity.category,
        "release_year": entity.release_year or "",
        "canonical_title": publication.title,
        "canonical_year": publication.year or "",
        "citation_count": publication.citations,
        "canonical_url": publication.url,
        "doi": publication.doi,
        "canonical_source": publication.source,
        "canonical_id": publication.source_id,
        "canonical_score": canonical.score if canonical else "",
        "score_reason": canonical.reason if canonical else "",
        "mention_count": len(entity.mentions),
        "survey_count": len(survey_sources),
        "survey_sources": "; ".join(survey_sources[:8]),
        "mention_sources": "; ".join(mention_sources),
        "candidate_publication_count": len(entity.candidate_publications),
        "candidate_publications": " | ".join(pub.title for pub in candidates[:5]),
    }


def output_columns() -> list[str]:
    return [
        "dataset",
        "normalized_name",
        "aliases",
        "category",
        "release_year",
        "canonical_title",
        "canonical_year",
        "citation_count",
        "canonical_url",
        "doi",
        "canonical_source",
        "canonical_id",
        "canonical_score",
        "score_reason",
        "mention_count",
        "survey_count",
        "survey_sources",
        "mention_sources",
        "candidate_publication_count",
        "candidate_publications",
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns())
        writer.writeheader()
        writer.writerows(rows)


def print_preview(rows: list[dict], limit: int = 80) -> None:
    columns = ["dataset", "category", "release_year", "citation_count", "canonical_score", "canonical_title"]
    widths = {column: len(column) for column in columns}
    for row in rows[:limit]:
        for column in columns:
            widths[column] = min(60, max(widths[column], len(str(row.get(column) or ""))))
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = str(row.get(column) or "")
            if len(value) > widths[column]:
                value = value[: widths[column] - 1] + "..."
            values.append(value.ljust(widths[column]))
        print("  ".join(values))


def parse_sources(value: str) -> list[str]:
    sources = [source.strip() for source in value.split(",") if source.strip()]
    valid = {"openalex", "semantic_scholar"}
    unknown = sorted(set(sources) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown source(s): {', '.join(unknown)}")
    return sources


def read_seed_file(path: Path | None) -> tuple[list[str], list[str], list[str]]:
    if not path:
        return [], [], []
    titles: list[str] = []
    dois: list[str] = []
    datasets: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("doi:") or line.startswith("https://doi.org/") or re.match(r"^10\.\d+/", line):
            dois.append(line.removeprefix("doi:").strip())
        elif line.startswith("dataset:"):
            datasets.append(line.removeprefix("dataset:").strip())
        else:
            titles.append(line)
    return titles, dois, datasets


def read_dataset_file(path: Path | None) -> list[str]:
    if not path:
        return []
    return [
        line.strip().removeprefix("dataset:").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover IDS dataset entities from survey seed papers and citation snowballing."
    )
    parser.add_argument("--out", type=Path, default=OUT, help="CSV output path.")
    parser.add_argument("--sources", type=parse_sources, default=list(DEFAULT_SOURCES))
    parser.add_argument("--per-page", type=int, default=DEFAULT_PER_PAGE)
    parser.add_argument("--max-seeds", type=int, default=DEFAULT_MAX_SEEDS)
    parser.add_argument("--max-references-per-seed", type=int, default=DEFAULT_MAX_REFERENCES_PER_SEED)
    parser.add_argument("--max-citations-per-canonical", type=int, default=DEFAULT_MAX_CITATIONS_PER_CANONICAL)
    parser.add_argument("--max-references-per-canonical", type=int, default=DEFAULT_MAX_REFERENCES_PER_CANONICAL)
    parser.add_argument("--max-recommendations", type=int, default=DEFAULT_MAX_RECOMMENDATIONS)
    parser.add_argument("--max-exact-matches", type=int, default=DEFAULT_MAX_EXACT_MATCHES)
    parser.add_argument("--max-alias-searches", type=int, default=4)
    parser.add_argument("--max-snowball-entities", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--seed-query", action="append", default=[], help="Additional survey seed query.")
    parser.add_argument("--seed-title", action="append", default=[], help="Trusted survey paper title.")
    parser.add_argument("--seed-doi", action="append", default=[], help="Trusted survey paper DOI.")
    parser.add_argument("--seed-file", type=Path, help="Line-based seed file: survey titles, DOI lines, or dataset: NAME.")
    parser.add_argument("--dataset-name", action="append", default=[], help="Exact dataset name to resolve through canonical-publication scoring.")
    parser.add_argument("--dataset-file", type=Path, help="Line-based file with exact dataset names to resolve.")
    parser.add_argument("--no-default-dataset-seeds", action="store_true", help="Do not include built-in exact seeds such as FLAME/MAWI/ID2T.")
    parser.add_argument("--fetch-fulltext", action="store_true", help="Try parsing open-access seed PDFs with pdftotext.")
    parser.add_argument("--max-fulltext-chars", type=int, default=120_000)
    parser.add_argument("--no-snowball", action="store_true", help="Stop after survey references and exact-name resolution.")
    parser.add_argument("--list-default-seeds", action="store_true", help="Print default survey seed queries and exit.")
    parser.add_argument("--list-default-dataset-seeds", action="store_true", help="Print built-in exact dataset-name seeds and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_default_seeds:
        for query in DEFAULT_SEED_SURVEY_QUERIES:
            print(query)
        return
    if args.list_default_dataset_seeds:
        for name, aliases in DEFAULT_DATASET_NAME_SEEDS.items():
            print("; ".join(dict.fromkeys([name, *aliases])))
        return

    file_titles, file_dois, file_dataset_names = read_seed_file(args.seed_file)
    seed_queries = list(dict.fromkeys([*DEFAULT_SEED_SURVEY_QUERIES, *args.seed_query]))
    seed_titles = list(dict.fromkeys([*args.seed_title, *file_titles]))
    seed_dois = list(dict.fromkeys([*args.seed_doi, *file_dois]))
    dataset_names = list(dict.fromkeys([*args.dataset_name, *file_dataset_names, *read_dataset_file(args.dataset_file)]))

    session = requests.Session()
    session.headers.update({"User-Agent": "ids-dataset-snowball-crawler/1.0"})

    seeds = resolve_seed_surveys(
        session,
        args.sources,
        seed_queries=seed_queries,
        seed_titles=seed_titles,
        seed_dois=seed_dois,
        max_seeds=args.max_seeds,
        per_page=args.per_page,
    )

    entities: dict[str, DatasetEntity] = {}
    add_default_dataset_name_seeds(entities, not args.no_default_dataset_seeds, dataset_names)

    if not seeds:
        print("No survey seed papers found; continuing with exact dataset-name seeds.", flush=True)
    else:
        collect_seed_neighborhood(session, args.sources, seeds, entities, args)
    print(f"Dataset-like entities after seed references: {len(entities)}", flush=True)

    if not entities:
        print("No dataset entities found.")
        return

    resolve_candidate_publications(session, args.sources, entities, args)
    choose_canonical_publications(entities, args.min_score)

    if not args.no_snowball:
        snowball_from_canonical(session, args.sources, entities, args)
        resolve_candidate_publications(session, args.sources, entities, args)
        choose_canonical_publications(entities, args.min_score)

    rows = [
        entity_to_row(entity)
        for entity in entities.values()
        if entity.canonical_publication is not None
    ]
    rows.sort(
        key=lambda row: (
            CATEGORY_ORDER.index(row["category"]) if row["category"] in CATEGORY_ORDER else len(CATEGORY_ORDER),
            -(float(row["canonical_score"]) if row["canonical_score"] else 0.0),
            row["dataset"].casefold(),
        )
    )

    if not rows:
        print("No canonical dataset publications found.")
        return

    write_csv(args.out, rows)
    print(f"\nSaved: {args.out} ({len(rows)} dataset entities)")
    print_preview(rows)


if __name__ == "__main__":
    main()
