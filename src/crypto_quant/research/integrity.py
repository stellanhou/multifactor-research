from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MANIFEST_NAME = "artifact_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
PAPER_APPEND_MANIFEST_SCHEMA_VERSION = 2
PAPER_APPEND_ARTIFACT_KIND = "append_only_paper_session"
HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _artifact_files(directory: Path) -> List[Path]:
    files: List[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name != MANIFEST_NAME:
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(directory).as_posix())


def build_artifact_manifest(directory: Path) -> Dict[str, Any]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {directory}")

    files: Dict[str, Dict[str, Any]] = {}
    total_bytes = 0
    for path in _artifact_files(directory):
        relative_path = path.relative_to(directory).as_posix()
        size_bytes = int(path.stat().st_size)
        files[relative_path] = {
            "size_bytes": size_bytes,
            "sha256": _sha256_file(path),
        }
        total_bytes += size_bytes

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "directory": directory.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    payload["manifest_sha256"] = _object_hash(payload)
    return payload


def write_artifact_manifest(directory: Path, manifest: Dict[str, Any]) -> None:
    _atomic_write_text(
        directory / MANIFEST_NAME,
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def load_artifact_manifest(directory: Path) -> Tuple[Dict[str, Any], Path]:
    path = directory / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest is not an object")
    return manifest, path


def seal_artifact_directory(
    directory: Path,
    method: str = "initial_seal",
    force: bool = False,
) -> Dict[str, Any]:
    """Seal a directory. Refuse to replace an existing valid or invalid manifest."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {directory}")
    if (directory / MANIFEST_NAME).exists() and not force:
        verification = verify_artifact_manifest(directory)
        if verification["status"] == "valid":
            return {
                "directory": str(directory),
                "status": "already_sealed",
                "file_count": verification["file_count"],
                "total_bytes": verification["total_bytes"],
            }
        if verification["status"] == "paper_append_pending":
            return {
                "directory": str(directory),
                "status": "paper_append_pending",
                "file_count": verification["file_count"],
                "total_bytes": verification["total_bytes"],
            }
        raise ValueError(
            f"refusing to reseal compromised artifacts in {directory}: "
            f"{verification['status']}"
        )

    manifest = build_artifact_manifest(directory)
    manifest["method"] = method
    # The method is metadata and must not participate in the content address.
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _object_hash(manifest)
    write_artifact_manifest(directory, manifest)
    return {
        "directory": str(directory),
        "status": "sealed",
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
    }


def append_evidence_to_sealed_directory(
    directory: Path,
    filename: str,
    content: str,
    reason: str,
) -> Dict[str, Any]:
    """Append a governance note to a sealed directory without altering old files."""
    directory = directory.resolve()
    if Path(filename).name != filename or not filename:
        raise ValueError("evidence filename must be a plain basename")
    verification = verify_artifact_manifest(directory)
    if verification["status"] not in {"valid", "extra_unmanifested_files"}:
        raise ValueError(
            f"refusing to append evidence to {directory}: {verification['status']}"
        )
    if verification["extra_files"] != [filename]:
        raise ValueError(
            "refusing to append evidence: unexpected unmanifested files "
            f"in {directory}: {verification['extra_files']}"
        )

    path = directory / filename
    wrote_new_file = not path.exists()
    if path.exists():
        # Late append registers the authoritative on-disk note; callers must not
        # replace governance content after the fact.
        content = path.read_text(encoding="utf-8")
    if wrote_new_file:
        _atomic_write_text(path, content)

    manifest, _ = load_artifact_manifest(directory)
    previous_manifest_sha256 = str(manifest["manifest_sha256"])
    expected_files = manifest["files"]
    try:
        relative_path = path.relative_to(directory).as_posix()
        size_bytes = int(path.stat().st_size)
        manifest["files"][relative_path] = {
            "size_bytes": size_bytes,
            "sha256": _sha256_file(path),
        }
        manifest["file_count"] = len(manifest["files"])
        manifest["total_bytes"] = int(manifest["total_bytes"]) + size_bytes
        manifest["method"] = "late_evidence_append"
        manifest["previous_manifest_sha256"] = previous_manifest_sha256
        manifest["append_reason"] = reason
        manifest["appended_at"] = datetime.now(timezone.utc).isoformat()
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = _object_hash(manifest)
        write_artifact_manifest(directory, manifest)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return {
        "directory": str(directory),
        "status": "evidence_appended",
        "filename": relative_path,
        "previous_manifest_sha256": previous_manifest_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def verify_artifact_manifest(directory: Path) -> Dict[str, Any]:
    directory = directory.resolve()
    try:
        manifest, manifest_path = load_artifact_manifest(directory)
    except FileNotFoundError as exc:
        return {
            "directory": str(directory),
            "status": "unsealed",
            "missing_files": [],
            "size_mismatches": [],
            "extra_files": [],
            "error": str(exc),
        }
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "directory": str(directory),
            "status": "manifest_invalid",
            "missing_files": [],
            "size_mismatches": [],
            "extra_files": [],
            "error": str(exc),
        }

    expected_files = manifest.get("files")
    recorded_hash = manifest.get("manifest_sha256")
    if not isinstance(expected_files, dict) or any(
        not isinstance(expected, dict) or "size_bytes" not in expected
        for expected in expected_files.values()
    ):
        return {
            "directory": str(directory),
            "status": "manifest_invalid",
            "missing_files": [],
            "size_mismatches": [],
            "extra_files": [],
            "error": "artifact manifest has invalid structure",
        }

    if (
        manifest.get("artifact_kind") == PAPER_APPEND_ARTIFACT_KIND
        or manifest.get("schema_version") == PAPER_APPEND_MANIFEST_SCHEMA_VERSION
    ):
        return _verify_paper_append_manifest(directory, manifest, recorded_hash)

    missing_files: List[str] = []
    size_mismatches: List[Dict[str, Any]] = []
    actual_files: Dict[str, Path] = {}
    for path in _artifact_files(directory):
        relative_path = path.relative_to(directory).as_posix()
        actual_files[relative_path] = path

    for relative_path, expected in sorted(expected_files.items()):
        path = directory / relative_path
        if not path.is_file() or path.is_symlink():
            missing_files.append(relative_path)
            continue
        actual_size = int(path.stat().st_size)
        if actual_size != int(expected["size_bytes"]):
            size_mismatches.append(
                {
                    "path": relative_path,
                    "expected_size_bytes": expected.get("size_bytes"),
                    "actual_size_bytes": actual_size,
                }
            )

    extra_files = sorted(
        relative_path
        for relative_path in actual_files
        if relative_path not in expected_files
    )

    if missing_files or size_mismatches:
        status = "compromised"
    elif extra_files:
        status = "extra_unmanifested_files"
    else:
        status = "valid"

    return {
        "directory": str(directory),
        "status": status,
        "manifest_sha256": recorded_hash,
        "file_count": int(len(expected_files)),
        "total_bytes": int(manifest.get("total_bytes", 0)),
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "extra_files": extra_files,
    }


def _verify_paper_append_manifest(
    directory: Path,
    manifest: Dict[str, Any],
    recorded_hash: Any,
) -> Dict[str, Any]:
    anchor = manifest.get("events_append_anchor")
    if manifest.get("schema_version") != PAPER_APPEND_MANIFEST_SCHEMA_VERSION:
        return {
            "directory": str(directory),
            "status": "manifest_invalid",
            "missing_files": [],
            "size_mismatches": [],
            "extra_files": [],
            "error": "paper append manifest has unsupported schema",
        }
    if not isinstance(anchor, dict):
        return {
            "directory": str(directory),
            "status": "manifest_invalid",
            "missing_files": [],
            "size_mismatches": [],
            "extra_files": [],
            "error": "paper append manifest lacks an event anchor",
        }

    expected_files = manifest["files"]
    missing_files: List[str] = []
    size_mismatches: List[Dict[str, Any]] = []
    actual_files: Dict[str, Path] = {}
    for path in _artifact_files(directory):
        relative_path = path.relative_to(directory).as_posix()
        actual_files[relative_path] = path

    for relative_path, expected in sorted(expected_files.items()):
        path = directory / relative_path
        if not path.is_file() or path.is_symlink():
            missing_files.append(relative_path)
            continue
        actual_size = int(path.stat().st_size)
        if actual_size != int(expected.get("size_bytes", -1)):
            size_mismatches.append(
                {
                    "path": relative_path,
                    "expected_size_bytes": expected.get("size_bytes"),
                    "actual_size_bytes": actual_size,
                }
            )

    extra_files = sorted(
        relative_path
        for relative_path in actual_files
        if relative_path not in expected_files
    )
    events_path = directory / "events.csv"
    events_anchor_length_ok = False
    events_size: Optional[int] = None
    if events_path.is_file() and not events_path.is_symlink():
        try:
            expected_byte_length = int(anchor["byte_length"])
            events_size = int(events_path.stat().st_size)
            events_anchor_length_ok = expected_byte_length >= 0 and events_size >= expected_byte_length
        except (KeyError, TypeError, ValueError, OSError):
            events_anchor_length_ok = False

    if missing_files or extra_files or not events_anchor_length_ok:
        status = "compromised"
    elif (
        size_mismatches
        and all(item["path"] == "events.csv" for item in size_mismatches)
        and events_size is not None
        and events_size > int(size_mismatches[0]["expected_size_bytes"])
    ):
        # A legitimate update appended bytes before the dedicated manifest
        # rotation ran. Keep this distinguishable from a history rewrite.
        status = "paper_append_pending"
    elif size_mismatches:
        status = "compromised"
    else:
        status = "valid"

    return {
        "directory": str(directory),
        "status": status,
        "manifest_sha256": recorded_hash,
        "file_count": int(len(expected_files)),
        "total_bytes": int(manifest.get("total_bytes", 0)),
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "extra_files": extra_files,
        "events_anchor_length_ok": events_anchor_length_ok,
    }


def capture_paper_append_anchor(directory: Path) -> Dict[str, Any]:
    """Capture the pre-update event length and manifest identity."""
    directory = directory.resolve()
    verification = verify_artifact_manifest(directory)
    if verification["status"] not in {"valid", "paper_append_pending"}:
        raise ValueError(
            f"refusing paper append for {directory}: "
            f"artifact status is {verification['status']}"
        )
    manifest, _ = load_artifact_manifest(directory)
    events_bytes = (directory / "events.csv").read_bytes()
    return {
        "previous_manifest_sha256": str(manifest["manifest_sha256"]),
        "byte_length": len(events_bytes),
        "sha256": _sha256_bytes(events_bytes),
        "expected_file_names": sorted(manifest["files"]),
    }


def rotate_paper_append_manifest(
    directory: Path,
    append_anchor: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    """Rotate a paper manifest after an event-only append."""
    directory = directory.resolve()
    actual_names = sorted(
        path.relative_to(directory).as_posix()
        for path in _artifact_files(directory)
    )
    expected_names = sorted(append_anchor.get("expected_file_names", []))
    if actual_names != expected_names:
        raise ValueError(
            "refusing paper manifest rotation: file set changed beyond the "
            f"sealed paper session contract in {directory}"
        )

    manifest = build_artifact_manifest(directory)
    manifest.pop("schema_version", None)
    manifest["schema_version"] = PAPER_APPEND_MANIFEST_SCHEMA_VERSION
    manifest["artifact_kind"] = PAPER_APPEND_ARTIFACT_KIND
    manifest["events_append_anchor"] = {
        key: append_anchor[key]
        for key in ("previous_manifest_sha256", "byte_length", "sha256")
    }
    manifest["previous_manifest_sha256"] = append_anchor[
        "previous_manifest_sha256"
    ]
    manifest["rotation_reason"] = reason
    manifest["rotated_at"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("method", None)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _object_hash(manifest)
    write_artifact_manifest(directory, manifest)
    return {
        "directory": str(directory),
        "status": "paper_append_rotated",
        "previous_manifest_sha256": manifest["previous_manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def enable_paper_append_manifest(directory: Path) -> Dict[str, Any]:
    """Convert a sealed genesis paper session to append-safe auditing."""
    directory = directory.resolve()
    verification = verify_artifact_manifest(directory)
    if verification["status"] != "valid":
        raise ValueError(
            f"refusing to enable paper append for {directory}: "
            f"artifact status is {verification['status']}"
        )
    anchor = capture_paper_append_anchor(directory)
    return rotate_paper_append_manifest(
        directory,
        anchor,
        "Enable append-safe forward-paper auditing before any new bars.",
    )


def seal_research_artifacts(output_root: Path) -> Dict[str, Any]:
    """Seal every experiment, audit, paper-session, and brief evidence directory."""
    output_root = output_root.resolve()
    roots: Iterable[Path] = (
        output_root / "runs",
        output_root / "audits",
        output_root / "paper_sessions",
        output_root / "briefs",
        output_root / "pinned_trials",
    )
    directories: List[Path] = []
    for root in roots:
        if root.is_dir():
            directories.extend(sorted(item for item in root.iterdir() if item.is_dir()))

    results = []
    for directory in directories:
        try:
            results.append(seal_artifact_directory(directory))
        except ValueError as exc:
            verification = verify_artifact_manifest(directory)
            if (
                verification["status"] == "extra_unmanifested_files"
                and verification["extra_files"] == ["INVALIDATION.md"]
            ):
                results.append(
                    append_evidence_to_sealed_directory(
                        directory,
                        "INVALIDATION.md",
                        (directory / "INVALIDATION.md").read_text(encoding="utf-8"),
                        "Register a post-seal invalidation note without changing original artifacts.",
                    )
                )
                continue
            raise
    return {
        "directories_scanned": len(directories),
        "sealed_now": sum(item["status"] == "sealed" for item in results),
        "already_sealed": sum(item["status"] == "already_sealed" for item in results),
        "results": results,
    }


def verify_ledger_append_only(
    current_bytes: bytes,
    expected_byte_length: int,
) -> Dict[str, Any]:
    """Check that a ledger has not fallen below its previously recorded size."""
    if len(current_bytes) < expected_byte_length:
        return {
            "status": "truncated",
            "expected_byte_length": expected_byte_length,
            "actual_byte_length": len(current_bytes),
        }
    return {
        "status": "length_preserved",
        "expected_byte_length": expected_byte_length,
        "actual_byte_length": len(current_bytes),
    }


def verify_research_artifact_manifests(
    output_root: Path,
    exclude_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify sealed evidence directories, optionally skipping an in-progress audit."""
    output_root = output_root.resolve()
    excluded = exclude_dir.resolve() if exclude_dir is not None else None
    roots = (
        output_root / "runs",
        output_root / "audits",
        output_root / "paper_sessions",
        output_root / "briefs",
        output_root / "pinned_trials",
    )
    results: List[Dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(item for item in root.iterdir() if item.is_dir()):
            if excluded is not None and directory.resolve() == excluded:
                continue
            verification = verify_artifact_manifest(directory)
            verification["artifact_group"] = root.name
            results.append(verification)

    counts = {
        "directories_scanned": len(results),
        "valid": sum(item["status"] == "valid" for item in results),
        "paper_append_pending": sum(
            item["status"] == "paper_append_pending" for item in results
        ),
        "unsealed": sum(item["status"] == "unsealed" for item in results),
        "extra_unmanifested_files": sum(
            item["status"] == "extra_unmanifested_files" for item in results
        ),
        "manifest_invalid": sum(
            item["status"] == "manifest_invalid" for item in results
        ),
        "compromised": sum(item["status"] == "compromised" for item in results),
    }
    failures = [
        item
        for item in results
        if item["status"]
        in {"compromised", "manifest_invalid", "missing_files"}
    ]
    return {"counts": counts, "results": results, "failures": failures}
