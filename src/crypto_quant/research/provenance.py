from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_KIND = "research_source_manifest"
CONFIG_FILES = (
    "pyproject.toml",
    "requirements-lock.txt",
    "Makefile",
    "README.md",
    "AGENTS.md",
    "RESEARCH_PLAYBOOK.md",
)
GOVERNANCE_DOCS = {
    "README.md",
    "AGENTS.md",
    "RESEARCH_PLAYBOOK.md",
}


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_manifest(
    project_root: Optional[Path] = None,
    relative_paths: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a deterministic manifest for the whole project or an explicit slice."""
    root = (project_root or default_project_root()).resolve()
    candidates: list[Path] = []
    if relative_paths is None:
        for filename in CONFIG_FILES:
            path = root / filename
            if path.is_file():
                candidates.append(path)
        for directory in ("src/crypto_quant", "tests"):
            directory_path = root / directory
            if directory_path.is_dir():
                candidates.extend(
                    path for path in directory_path.rglob("*.py") if path.is_file()
                )
    else:
        for value in relative_paths:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("manifest paths must be project-relative")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            candidates.append(path)

    files: Dict[str, Dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(set(candidates)):
        if path.is_symlink() or "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(root).as_posix()
        size_bytes = int(path.stat().st_size)
        files[relative_path] = {
            "size_bytes": size_bytes,
            "sha256": _sha256_file(path),
            "role": (
                "governance"
                if relative_path in GOVERNANCE_DOCS
                else "test"
                if relative_path.startswith("tests/")
                else "runtime"
            ),
        }
        total_bytes += size_bytes

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    return {
        **payload,
        "manifest_sha256": _object_hash(payload),
    }


def spot_klines_digest(db_path: Path) -> str:
    """Hash price/execution inputs without unrelated additive database tables."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT symbol, interval, open_time, open, high, low, close,
                   volume, close_time, quote_volume
            FROM klines
            ORDER BY symbol, interval, open_time
            """
        ).fetchall()
    finally:
        conn.close()
    digest = hashlib.sha256()
    for row in rows:
        values = [str(row[item]) for item in row.keys()]
        digest.update(("\x1f".join(values) + "\x1e").encode("utf-8"))
    return digest.hexdigest()
