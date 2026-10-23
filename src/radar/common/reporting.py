"""Versioned relational reporting, separate from mutable inference caches."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any

from radar.common.csvio import read_csv
from radar.common.sources import SourceContract, content_hash, entity_identifier, source_contract

SCHEMA_VERSION = 1
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS snapshots(
 snapshot_id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_path TEXT NOT NULL,
 sha256 TEXT NOT NULL, contents BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS entities(entity_id TEXT PRIMARY KEY, entity_kind TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS identifiers(
 provider TEXT NOT NULL, entity_kind TEXT NOT NULL, external_id TEXT NOT NULL,
 entity_id TEXT NOT NULL REFERENCES entities(entity_id),
 PRIMARY KEY(provider,entity_kind,external_id));
CREATE TABLE IF NOT EXISTS runs(
 run_id TEXT PRIMARY KEY, contract_json TEXT NOT NULL, provenance_status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source_records(
 snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
 entity_id TEXT NOT NULL REFERENCES entities(entity_id), payload_json TEXT NOT NULL,
 PRIMARY KEY(snapshot_id,entity_id));
CREATE TABLE IF NOT EXISTS decisions(
 decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
 entity_id TEXT NOT NULL REFERENCES entities(entity_id),
 snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
 outcome TEXT NOT NULL, classifier_id TEXT NOT NULL, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence(
 evidence_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
 sha256 TEXT NOT NULL, payload_json TEXT NOT NULL, provenance_status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stage_memberships(
 run_id TEXT NOT NULL REFERENCES runs(run_id), entity_id TEXT NOT NULL REFERENCES entities(entity_id),
 stage TEXT NOT NULL, outcome TEXT NOT NULL, PRIMARY KEY(run_id,entity_id,stage));
CREATE TABLE IF NOT EXISTS relationships(
 relationship_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
 from_entity TEXT NOT NULL REFERENCES entities(entity_id), to_entity TEXT NOT NULL REFERENCES entities(entity_id),
 relation TEXT NOT NULL, basis_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reference_sets(reference_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reference_entities(
 reference_id TEXT NOT NULL REFERENCES reference_sets(reference_id), reference_key TEXT NOT NULL,
 expected_json TEXT NOT NULL, PRIMARY KEY(reference_id,reference_key));
CREATE TABLE IF NOT EXISTS evaluations(
 evaluation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
 reference_id TEXT NOT NULL REFERENCES reference_sets(reference_id), metrics_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reference_matches(
 evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id), reference_key TEXT NOT NULL,
 entity_id TEXT REFERENCES entities(entity_id), decision_id TEXT REFERENCES decisions(decision_id),
 match_method TEXT NOT NULL, status TEXT NOT NULL, instance_json TEXT NOT NULL,
 PRIMARY KEY(evaluation_id,reference_key));
CREATE TABLE IF NOT EXISTS figure_manifests(figure_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL);
"""


class ReportingDatabase:
    """Store immutable report snapshots and enforce explicit entity relationships."""
    def __init__(self, path: Path):
        """Create the additive reporting store without modifying source caches."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and "schema_meta" not in tables:
            self.db.close()
            raise ValueError("reporting database must be separate from the retrieval/inference cache")
        if "schema_meta" in tables and list(self.db.execute("SELECT version FROM schema_meta")) != [(SCHEMA_VERSION,)]:
            self.db.close()
            raise ValueError("unsupported reporting schema version")
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO schema_meta VALUES (?)", (SCHEMA_VERSION,))
        self.db.commit()

    def __enter__(self) -> ReportingDatabase:
        """Return the reporting store as a managed resource."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Release the connection after a successful or failed import."""
        self.db.close()

    def snapshot(self, path: Path, kind: str) -> str:
        """Preserve compressed bytes, detecting corruption on repeated imports."""
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        key = content_hash([kind, digest])
        existing = self.db.execute("SELECT contents FROM snapshots WHERE snapshot_id=?", (key,)).fetchone()
        if existing and zlib.decompress(existing[0]) != data:
            raise ValueError("corrupted reporting snapshot")
        self.db.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?)", (key, kind, str(path.resolve()), digest, zlib.compress(data)))
        return key

    def entity(self, contract: SourceContract, external: str) -> str:
        """Register a native identity without merging different providers by title."""
        if not external:
            raise ValueError("reporting entity requires a native identifier")
        key = entity_identifier(contract, external)
        self.db.execute("INSERT OR IGNORE INTO entities VALUES (?,?)", (key, contract.entity_kind))
        self.db.execute("INSERT OR IGNORE INTO identifiers VALUES (?,?,?,?)", (contract.name, contract.entity_kind, external, key))
        return key

    def relationship(self, run_id: str, origin: str, target: str, relation: str, basis: dict[str, Any]) -> None:
        """Record an explicit, attributable match or family link rather than an implicit merge."""
        if not relation or not basis:
            raise ValueError("relationships require a type and a documented basis")
        key = content_hash([run_id, origin, target, relation, basis])
        self.db.execute("INSERT OR IGNORE INTO relationships VALUES (?,?,?,?,?,?)", (key, run_id, origin, target, relation, json.dumps(basis, sort_keys=True)))

    def ingest_records(self, contract: SourceContract, path: Path, stage: str = "classification", run_id: str | None = None) -> str:
        """Persist a provider snapshot even when no evaluation reference set is supplied."""
        rows = read_csv(path)
        run_id = run_id or content_hash([contract.fingerprint, rows, stage])
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?)", (run_id, json.dumps(contract.__dict__, sort_keys=True), "frozen_inputs"))
            snapshot_id = self.snapshot(path, stage)
            seen: set[str] = set()
            for row in rows:
                external = row.get(contract.identity, "").strip()
                if not external or external in seen:
                    raise ValueError(f"duplicate or missing reporting identity: {external}")
                seen.add(external)
                entity = self.entity(contract, external)
                self.db.execute("INSERT OR IGNORE INTO source_records VALUES (?,?,?)", (snapshot_id, entity, json.dumps(row, sort_keys=True)))
                self.db.execute("INSERT OR IGNORE INTO stage_memberships VALUES (?,?,?,?)", (run_id, entity, stage, row.get("decision", "observed")))
                if stage == 'candidate_selection' and row.get('prefilter_band') in {'candidate', 'borderline'}:
                    self.db.execute('INSERT OR IGNORE INTO stage_memberships VALUES (?,?,?,?)', (run_id, entity, 'enrichment_prefilter', row['prefilter_band']))
                if stage == 'classification' and row.get('decision'):
                    decision_id = content_hash([run_id, entity, row])
                    self.db.execute('INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?)', (decision_id, run_id, entity, snapshot_id, row['decision'], row.get('classifier_id', ''), json.dumps(row, sort_keys=True)))
                    if row['decision'] in {'accept', 'accept_adjacent'} and not row.get('error'):
                        self.db.execute('INSERT OR IGNORE INTO stage_memberships VALUES (?,?,?,?)', (run_id, entity, 'inclusion', row['decision']))
        return run_id

    def ingest_evaluation(self, directory: Path, contract: SourceContract | None = None) -> str:
        """Import a current-format evaluation with verified input fingerprints."""
        metrics = json.loads((directory / "metrics.json").read_text())
        contract = source_contract(metrics["source"], contract)
        instances = read_csv(directory / "instances.csv")
        classifications = Path(metrics["classifications_path"])
        candidates = Path(metrics.get("candidates_path") or "")
        fingerprints = metrics.get("input_fingerprints", {})
        if not fingerprints.get("classifications") or not metrics.get("reference_fingerprint") or not metrics.get("taxonomy_fingerprint"):
            raise ValueError("evaluation import requires current reference, taxonomy and classification fingerprints")
        if metrics.get("candidates_path") and (not fingerprints.get("candidates") or not metrics.get("cohort_fingerprint")):
            raise ValueError("evaluation import requires candidate and cohort fingerprints")
        classification_source = Path(metrics.get("classification_source_path") or "")
        provenance = "frozen_inputs"
        for role, path in (("classifications", classifications), ("candidates", candidates), ("classification_source", classification_source)):
            if fingerprints.get(role) and (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != fingerprints[role]):
                raise ValueError(f"cannot import {role}: input changed since evaluation")
        run_id = content_hash([contract.fingerprint, fingerprints, metrics.get("classifier_id"), instances])
        reference_id = metrics["reference_fingerprint"]
        evaluation_id = content_hash([run_id, reference_id, metrics])
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?)", (run_id, json.dumps(contract.__dict__, sort_keys=True), provenance))
            self.db.execute("INSERT OR IGNORE INTO reference_sets VALUES (?,?)", (reference_id, json.dumps(metrics.get("reference_set", {}), sort_keys=True)))
            self.db.execute("INSERT OR IGNORE INTO evaluations VALUES (?,?,?,?)", (evaluation_id, run_id, reference_id, json.dumps(metrics, sort_keys=True)))
            for path in sorted(directory.glob("*.csv")) + [directory / "metrics.json"]:
                self.snapshot(path, "evaluation")
            for name, digest in metrics.get('reference_input_fingerprints', {}).items():
                path = Path(name)
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'cannot import reference input: missing or changed {path}')
                self.snapshot(path, 'reference_input')
            if metrics.get("classification_source_path"):
                self.snapshot(classification_source, "classification_source")
            records: dict[str, tuple[str, str]] = {}
            for role, path in (("candidates", candidates), ("classifications", classifications)):
                if not path.is_file():
                    continue
                snapshot_id = self.snapshot(path, role)
                seen: set[str] = set()
                for row in read_csv(path):
                    external = row.get(contract.identity, "").strip()
                    if not external or external in seen:
                        raise ValueError(f"duplicate or missing reporting identity in {path}: {external}")
                    seen.add(external)
                    entity = self.entity(contract, external)
                    self.db.execute("INSERT OR IGNORE INTO source_records VALUES (?,?,?)", (snapshot_id, entity, json.dumps(row, sort_keys=True)))
                    stage = "classification" if role == "classifications" else "candidate_selection"
                    self.db.execute("INSERT OR IGNORE INTO stage_memberships VALUES (?,?,?,?)", (run_id, entity, stage, row.get("decision", row.get("prefilter_band", "observed"))))
                    if role == "candidates" and row.get("prefilter_band") in {"candidate", "borderline"}:
                        self.db.execute("INSERT OR IGNORE INTO stage_memberships VALUES (?,?,?,?)", (run_id, entity, "enrichment_prefilter", row["prefilter_band"]))
                    if role != "classifications":
                        continue
                    decision_id = content_hash([run_id, entity, row])
                    self.db.execute("INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?)", (decision_id, run_id, entity, snapshot_id, row.get("decision", ""), row.get("classifier_id", ""), json.dumps(row, sort_keys=True)))
                    records[external] = (entity, decision_id)
                    if row.get("decision") in {"accept", "accept_adjacent"} and not row.get('error'):
                        self.db.execute("INSERT OR IGNORE INTO stage_memberships VALUES (?,?,?,?)", (run_id, entity, "inclusion", row["decision"]))
                    for field in ("evidence_json", "nli_evidence_chunks_json", "llm_nli_pair_scores_json"):
                        if not row.get(field):
                            continue
                        payload = json.loads(row[field])
                        evidence_hash = content_hash(payload)
                        self.db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)", (content_hash([decision_id, field, evidence_hash]), decision_id, evidence_hash, json.dumps(payload, sort_keys=True), "stored_classifier_audit:" + field))
                    if row.get("artifact_id"):
                        artifact_contract = SourceContract(contract.name, "artifact", "artifact_id", "artifact_name", {"id": "artifact_id"})
                        target = self.entity(artifact_contract, row["artifact_id"])
                        self.relationship(run_id, entity, target, "describes_artifact", {"field": "artifact_id", "snapshot": snapshot_id})
            for row in instances:
                reference_key = content_hash([row["match_field"], row["match_value"]])
                self.db.execute("INSERT OR IGNORE INTO reference_entities VALUES (?,?,?)", (reference_id, reference_key, row["expected_json"]))
                actual = json.loads(row["actual_json"])
                external = str(actual.get(contract.identity, ""))
                entity, decision = records.get(external, (None, None))
                if decision:
                    saved = json.loads(self.db.execute("SELECT payload_json FROM decisions WHERE decision_id=?", (decision,)).fetchone()[0])
                    if actual != saved:
                        raise ValueError("evaluation instance differs from its classification input")
                self.db.execute("INSERT OR IGNORE INTO reference_matches VALUES (?,?,?,?,?,?,?)", (evaluation_id, reference_key, entity, decision, row["match_method"], row["status"], json.dumps(row, sort_keys=True)))
        self.add_figures(directory / "figures")
        return evaluation_id

    def add_figures(self, directory: Path) -> None:
        """Index figure manifests while their independent frozen files remain portable."""
        with self.db:
            for path in sorted(directory.rglob("*.manifest.json")):
                manifest = json.loads(path.read_text())
                renderer = manifest.get('definition', {}).get('renderer_snapshot')
                for item in [*manifest['inputs'], *manifest['tables'], *manifest['artifacts'], *([renderer] if renderer else [])]:
                    archive = Path(item.get('snapshot', item['path']))
                    if not archive.is_file():
                        for parent in [path.parent, *path.parents]:
                            candidate = parent / '.provenance' / 'objects' / item['sha256']
                            if candidate.is_file():
                                archive = candidate
                                break
                    if not archive.is_file():
                        raise ValueError(f"cannot index missing figure object: {archive}")
                    if hashlib.sha256(archive.read_bytes()).hexdigest() != item['sha256']:
                        raise ValueError('cannot index changed figure input or artifact')
                    self.snapshot(archive, 'figure_object')
                self.db.execute("INSERT OR REPLACE INTO figure_manifests VALUES (?,?)", (manifest["figure_id"], json.dumps(manifest, sort_keys=True)))
