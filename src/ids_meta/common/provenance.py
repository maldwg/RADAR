"""Frozen inputs and explicit provenance for reproducible reporting bundles."""
from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from ids_meta.common.sources import content_hash


def file_hash(path: Path) -> str:
    """Hash bytes incrementally, including large classification audits."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_config(value: Any) -> Any:
    """Preserve plot settings without copying credentials into an audit manifest."""
    if isinstance(value, dict):
        return {str(k): _safe_config(v) for k, v in value.items()
            if not any(word in str(k).casefold() for word in ('password', 'secret', 'api_key', 'authorization'))
            and str(k).casefold() != 'token' and not str(k).casefold().endswith('_token')}
    if isinstance(value, (list, tuple)):
        return [_safe_config(v) for v in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def freeze_file(path: Path, root: Path) -> dict[str, str]:
    """Archive exact bytes under their content digest, independent of live filenames."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    archive = root / ".provenance" / "objects" / digest
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_bytes(data)
    elif file_hash(archive) != digest:
        raise ValueError(f"corrupted frozen input: {archive}")
    return {"path": str(path.resolve()), "sha256": digest, "snapshot": str(archive.resolve())}


def freeze_database(path: Path, root: Path) -> Path:
    """Capture a consistent SQLite view, including WAL contents, through a read-only source."""
    with tempfile.TemporaryDirectory(prefix='ids-meta-snapshot-') as temporary:
        target = Path(temporary) / 'snapshot.sqlite3'
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(target) as destination:
                source.backup(destination)
        frozen = freeze_file(target, root)
    archive = Path(frozen['snapshot'])
    register_input(archive)
    return archive


@dataclass
class BundleAudit:
    """Track inputs, table recipes and renderer identity for one plotting operation."""
    root: Path
    definition: dict[str, Any]
    inputs: dict[str, dict[str, str]] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)

    def input(self, path: Path) -> None:
        """Capture a source file once before any transformations are applied."""
        key = str(path.resolve())
        if path.is_file() and key not in self.inputs:
            self.inputs[key] = freeze_file(path, self.root)


_AUDIT: ContextVar[BundleAudit | None] = ContextVar("report_bundle_audit", default=None)


def register_input(path: Path) -> None:
    """Attach a CSV, profile or metrics file to the active figure bundle."""
    context = _AUDIT.get()
    if context is not None:
        context.input(path)


def register_table(path: Path, recipe: dict[str, Any] | None = None) -> None:
    """Capture exactly the exported table plus an optional independent replay recipe."""
    context = _AUDIT.get()
    if context is not None:
        if recipe is None and context.definition['renderer'] == 'ids_meta.plotting.evaluation.build':
            sources = {Path(item['path']).name: item['sha256'] for item in context.inputs.values()}
            if 'instances.csv' in sources and 'metrics.json' in sources:
                recipe = {'kind': 'evaluation_table', 'input_sha256': sources['instances.csv'],
                    'metrics_sha256': sources['metrics.json'], 'table': path.stem}
        context.tables[str(path.resolve())] = {**freeze_file(path, context.root), "recipe": recipe}


def capture_figure(paths: list[Path], out_dir: Path, stem: str) -> Path:
    """Link a figure's bytes to frozen numeric tables and explicit provenance limits."""
    context = _AUDIT.get()
    if context is None:
        raise ValueError("figure generation requires an audited rendering context")
    matched = [v for p, v in context.tables.items() if Path(p).stem == stem or Path(p).stem.startswith(stem + "_")]
    tables = matched or list(context.tables.values())
    manifest = {
        "schema_version": "1.0", "figure_id": content_hash([stem, context.definition, sorted(v['sha256'] for v in context.inputs.values()), sorted(v['sha256'] for v in tables)]),
        "definition": context.definition, "inputs": list(context.inputs.values()), "tables": tables,
        "artifacts": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in paths],
        "verification": "recomputable" if any(v.get("recipe") for v in tables) else "integrity_only",
        "lineage": "explicit_recipe" if any(v.get("recipe") for v in tables) else "input_snapshot; contribution mapping not declared",
    }
    result = out_dir / f"{stem}.manifest.json"
    result.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def _paths(value: Any) -> list[Path]:
    """Find explicit path arguments without interpreting arbitrary text as filenames."""
    if isinstance(value, Path):
        return [value]
    if isinstance(value, dict):
        return [p for item in value.values() for p in _paths(item)]
    if isinstance(value, (tuple, list)):
        return [p for item in value for p in _paths(item)]
    return []


def audited_bundle(function: Callable) -> Callable:
    """Add immutable input manifests to existing renderers without changing their API."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Freeze declared inputs, execute the renderer and reject mixed snapshots."""
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        root = values.get("out_dir") or values.get("output")
        if root is None and "input_dir" in values:
            root = values["input_dir"] / "figures"
        if root is None and "settings" in values:
            root = values["settings"].path("plots_dir") / values.get("source", "shared") / "pipeline_funnel"
        root = Path(root or values.get("root") or ".")
        settings = values.get('settings')
        configuration = values.get('config', {}) if settings is None else settings.values
        context = BundleAudit(root, {"renderer": function.__module__ + "." + function.__qualname__,
            "code_sha256": file_hash(Path(inspect.getfile(function))), "settings": _safe_config(configuration)})
        context.definition['renderer_snapshot'] = freeze_file(Path(inspect.getfile(function)), root)
        token = _AUDIT.set(context)
        try:
            for name, value in values.items():
                if name in {"out_dir", "output", "root", "config", "settings"}:
                    continue
                for path in _paths(value):
                    if path.is_file():
                        context.input(path)
                    elif path.is_dir():
                        for child in sorted(path.iterdir()):
                            if child.is_file() and child.suffix in {".csv", ".json", ".toml"} and not child.name.endswith(".manifest.json"):
                                context.input(child)
            result = function(*args, **kwargs)
            # Custom renderers (architecture) may not call save_figure.
            outputs = [p for p in _paths(result) if p.suffix in {".svg", ".pdf", ".png"} and p.is_file()]
            for directory, stem in sorted({(p.parent, p.stem) for p in outputs}):
                manifest = directory / f"{stem}.manifest.json"
                if manifest.is_file() and json.loads(manifest.read_text())["definition"].get("renderer") != context.definition["renderer"]:
                    continue
                capture_figure([p for p in outputs if p.parent == directory and p.stem == stem], directory, stem)
            for item in context.inputs.values():
                path = Path(item["path"])
                if not path.is_file() or file_hash(path) != item["sha256"]:
                    raise ValueError(f"reporting input changed while rendering: {path}")
            return result
        finally:
            _AUDIT.reset(token)
    return wrapped
