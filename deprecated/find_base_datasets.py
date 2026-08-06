#!/usr/bin/env python3
"""
Direct API search and heuristic-based IDS dataset base paper discovery.

This script identifies the original "base papers" that introduce new Intrusion 
Detection System (IDS) datasets. It avoids hardcoding dataset names by querying
scientific APIs for dataset creation terminology and applying strong heuristics 
to filter out "downstream" papers (which merely evaluate on existing datasets).
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests

OUT = Path("ids_base_papers.csv")

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEMANTIC_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"

REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 3
MAX_RETRY_AFTER_SECONDS = 60

# --- Text Normalization ---

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

# --- Data Structures ---

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
    relation: str = ""

    @property
    def stable_key(self) -> str:
        return self.doi or f"{self.source}:{self.source_id}" or compact_key(self.title)

# --- API Wrappers ---

class SourceRateLimited(RuntimeError):
    pass

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

def reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    return " ".join(word for _, word in sorted(positions))

def openalex_authors(item: dict) -> str:
    authors = []
    for authorship in (item.get("authorships") or [])[:6]:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            authors.append(author["display_name"])
    return ", ".join(authors)

def publication_from_openalex(item: dict, relation: str = "") -> Publication:
    title = item.get("title") or item.get("display_name") or ""
    location = item.get("primary_location") or {}
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
        relation=relation,
    )

def openalex_select_fields() -> str:
    return ",".join([
        "id", "doi", "title", "display_name", "publication_year", "cited_by_count",
        "abstract_inverted_index", "primary_location", "authorships",
        "referenced_works", "type", "type_crossref"
    ])

def openalex_search(
    session: requests.Session,
    query: str,
    per_page: int,
    pages: int,
    relation: str,
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
            publications.append(publication_from_openalex(item, relation=relation))
        time.sleep(0.2)
    return publications

def semantic_headers() -> dict:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": api_key} if api_key else {}

def publication_from_semantic(item: dict, relation: str = "") -> Publication:
    external = item.get("externalIds") or {}
    doi = external.get("DOI") or ""
    authors = ", ".join((author.get("name") or "") for author in (item.get("authors") or [])[:6])
    references = [
        ref.get("paperId") for ref in item.get("references") or [] if ref and ref.get("paperId")
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
    )

def semantic_search(
    session: requests.Session,
    query: str,
    per_page: int,
    pages: int,
    relation: str,
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
            publications.append(publication_from_semantic(item, relation=relation))
        time.sleep(1.0 if semantic_headers() else 2.5) # Increased sleep to avoid rate limits
    return publications

def semantic_paper_endpoint(
    session: requests.Session,
    paper_id: str,
    suffix: str,
    limit: int,
    relation: str,
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
            publications.append(publication_from_semantic(paper, relation=relation))
    return publications[:limit]

# --- Heuristic Base Paper Verifier ---

def is_base_paper(publication: Publication) -> tuple[bool, float, str]:
    """
    Evaluates if a publication is a base paper introducing a new dataset.
    Returns (is_base_paper, score, reason).
    """
    title = comparable(publication.title)
    abstract = comparable(publication.abstract)
    full_text = f"{title} {abstract}"

    if not full_text.strip():
        return False, 0.0, "No text"

    # Must be somewhat related to the domain
    if not has_phrase(full_text, ["intrusion detection", "network security", "cybersecurity", "malware", "botnet", "ddos", "iot security", "traffic archive", "packet trace"]):
        return False, 0.0, "Out of domain"

    score = 0.0
    reasons = []

    # --- Positive Indicators (Dataset Creation) ---
    intro_phrases = [
        "we propose a new dataset", "we introduce a new dataset", "we present a new dataset",
        "we propose a novel dataset", "we introduce a novel dataset", "we present a novel dataset",
        "we propose the", "we introduce the", "we present the", "we present a comprehensive dataset",
        "we release a dataset", "we created a dataset", "generation of a new dataset",
        "a new dataset for", "towards a realistic dataset", "a novel dataset for",
        "we publish a dataset", "dataset creation toolkit"
    ]
    if has_phrase(abstract, intro_phrases):
        score += 5.0
        reasons.append("strong_intro_abstract")

    title_intro = [
        "a new dataset", "a novel dataset", "a realistic dataset", "towards a",
        "dataset generation", "creation of", "introducing", "presenting"
    ]
    if has_phrase(title, title_intro) and has_phrase(title, ["dataset", "benchmark", "corpus"]):
        score += 4.0
        reasons.append("strong_intro_title")

    if re.search(r"\bwe (present|introduce|propose|release|create|generate) (a|the) (new|novel|comprehensive|realistic )?(dataset|benchmark|corpus)\b", abstract):
        score += 3.0
        reasons.append("regex_intro_abstract")

    # --- Negative Indicators (Downstream / Detector / Survey) ---
    downstream_phrases = [
        "evaluated on the", "using the", "trained on the", "tested on the", "performance of",
        "applied to the", "our model outperforms", "accuracy of", "precision and recall",
        "machine learning approach for", "deep learning model", "we evaluate our",
        "compared with existing", "compared to existing", "state of the art",
        "dataset balancing", "dataset reduction", "feature selection", "feature extraction"
    ]
    if has_phrase(title, ["survey", "review", "taxonomy", "performance analysis", "comparison of", "balancing", "feature"]):
        score -= 10.0
        reasons.append("is_survey_or_review_or_method")

    if has_phrase(abstract, downstream_phrases):
        score -= 3.0
        reasons.append("downstream_phrases_abstract")

    # If it's presenting a model or system rather than a dataset
    if re.search(r"\b(propose|present|introduce) a (new |novel )?(model|framework|system|algorithm|method|architecture|approach)\b", abstract):
        score -= 2.0
        reasons.append("presents_model_not_dataset")

    # Many mentions of known datasets often imply evaluation on them
    known_datasets = ["nsl kdd", "kdd cup", "kdd99", "cicids", "cic ids", "unsw nb15", "unsw-nb15", "bot-iot", "ton_iot"]
    for kd in known_datasets:
        if kd in full_text:
            score -= 1.0
            reasons.append(f"mentions_known_dataset_{kd}")

    is_base = score >= 3.0
    return is_base, score, ", ".join(reasons)

def extract_dataset_name(publication: Publication) -> str:
    """Attempts to extract the dataset name from the base paper."""
    title = publication.title
    abstract = publication.abstract
    
    # Heuristic 1: Pre-colon acronym (e.g. "LITNET-2020: An Annotated...")
    match = re.search(r"^([A-Za-z0-9-]+)\s*:", title)
    if match:
        name = match.group(1)
        if len(name) > 3 and name.upper() == name or any(char.isdigit() for char in name):
            return name

    # Heuristic 2: dataset, NAME,
    match = re.search(r"dataset,\s*([A-Za-z0-9-]+)\s*,", title)
    if match:
        return match.group(1)

    # Heuristic 3: All caps word before "dataset" in abstract or title
    match = re.search(r"\b([A-Z0-9]+(-[A-Z0-9]+)*)\s+(dataset|benchmark|corpus)\b", title, re.IGNORECASE)
    if match:
        name = match.group(1)
        if len(name) > 2 and name.upper() == name:
             return name

    match = re.search(r"\b([A-Z0-9]+(-[A-Z0-9]+)*)\s+(dataset|benchmark|corpus)\b", abstract)
    if match:
         name = match.group(1)
         if len(name) > 2 and name.upper() == name:
             return name
             
    # Heuristic 4: "named X", "called X"
    match = re.search(r"\b(?:named|called|referred to as)\s+(?:the\s+)?([A-Z][a-zA-Z0-9-_]+)", abstract)
    if match:
        return match.group(1)

    return "Unknown"

# --- Main Logic ---

def direct_search_phase(session: requests.Session, pages: int) -> list[Publication]:
    print("--- Phase 1: Direct API Search ---")
    queries = [
        '"intrusion detection" AND "new dataset"',
        '"network security" AND "dataset generation"',
        '"intrusion detection" AND "novel dataset"',
        '"iot security" AND "new dataset"'
    ]
    
    candidates = []
    for query in queries:
        print(f"Querying OpenAlex: {query}")
        try:
            candidates.extend(openalex_search(session, query, per_page=25, pages=pages, relation="direct_search"))
        except Exception as e:
            print(f"OpenAlex error: {e}")
            
        print(f"Querying Semantic Scholar: {query}")
        try:
            candidates.extend(semantic_search(session, query, per_page=25, pages=pages, relation="direct_search"))
        except Exception as e:
            print(f"Semantic Scholar error: {e}")
        # timeout to not fall into RL
        time.sleep(30)

    return candidates

def snowball_phase(session: requests.Session, candidates: list[Publication]) -> list[Publication]:
    print("--- Phase 2: Snowballing from Surveys ---")
    survey_queries = [
        "Survey of intrusion detection systems techniques datasets and challenges Khraisat",
        "A survey of network-based intrusion detection data sets Ring",
        "network intrusion detection datasets survey",
        "IoT intrusion detection datasets survey",
        "industrial control system intrusion detection datasets survey"
    ]
    
    # 1. Find surveys
    surveys = []
    for query in survey_queries:
        print(f"Fetching survey: {query}")
        try:
            surveys.extend(semantic_search(session, query, per_page=1, pages=1, relation="survey"))
        except Exception as e:
            print(f"  Failed: {e}")
            
    print(f"Found {len(surveys)} surveys. Extracting references...")
    snowball_candidates = []
    
    # 2. Get references from surveys
    for survey in surveys:
        if not survey.source_id:
            continue
        try:
            refs = semantic_paper_endpoint(session, survey.source_id, "references", limit=100, relation="snowball_reference")
            for ref in refs:
                # Basic pre-filter before adding
                if has_phrase(ref.title, ["dataset", "data set", "benchmark", "corpus", "traffic"]):
                    snowball_candidates.append(ref)
        except Exception as e:
            print(f"  Failed to get refs for {survey.title}: {e}")

    print(f"Snowballed {len(snowball_candidates)} candidate references.")
    return snowball_candidates

def run_pipeline(args: argparse.Namespace):
    session = requests.Session()
    session.headers.update({"User-Agent": "IDS-Dataset-Discovery/1.0 (mailto:researcher@example.com)"})

    print("Starting IDS dataset base paper discovery...")
    candidates = direct_search_phase(session, pages=args.pages)
    
    # Phase 2: Snowball from surveys
    if args.snowball:
        snowball_candidates = snowball_phase(session, candidates)
        candidates.extend(snowball_candidates)
    
    print(f"Total candidates fetched: {len(candidates)}")
    
    base_papers: dict[str, Publication] = {} # Deduplicate by stable_key
    extracted_names: dict[str, str] = {}
    
    for pub in candidates:
        is_base, score, reason = is_base_paper(pub)
        if is_base:
            if pub.stable_key not in base_papers:
                base_papers[pub.stable_key] = pub
                extracted_names[pub.stable_key] = extract_dataset_name(pub)
            else:
                # Merge logic (e.g. keep one with most citations)
                if pub.citations > base_papers[pub.stable_key].citations:
                    base_papers[pub.stable_key] = pub

    print(f"Identified {len(base_papers)} base papers.")

    # Export to CSV
    print(f"Writing results to {OUT}...")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Extracted_Name", "Title", "Authors", "Year", "Citations", 
            "DOI", "URL", "Score", "Reason", "Abstract_Snippet"
        ])
        
        # Sort by citations descending
        sorted_pubs = sorted(base_papers.values(), key=lambda p: p.citations, reverse=True)
        
        for pub in sorted_pubs:
            name = extracted_names.get(pub.stable_key, "Unknown")
            _, score, reason = is_base_paper(pub)
            snippet = pub.abstract[:200] + "..." if pub.abstract else ""
            writer.writerow([
                name, pub.title, pub.authors, pub.year, pub.citations,
                pub.doi, pub.url, score, reason, snippet
            ])
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find IDS base papers")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to fetch per query per API")
    parser.add_argument("--snowball", action="store_true", help="Enable Phase 2 snowballing")
    args = parser.parse_args()
    run_pipeline(args)
