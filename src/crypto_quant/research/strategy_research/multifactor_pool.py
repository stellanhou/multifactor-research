"""Deterministic inventory and grouping for the factor idea-card pool."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from crypto_quant.features.factor_expressions import compile_expression
from crypto_quant.research.strategy_research.multifactor_factors import read_cards


_TIME_SERIES_OPERATORS = {
    "ts_delay", "ts_delta", "ts_return", "ts_rsi", "ts_mean", "ts_sum",
    "ts_min", "ts_max", "ts_std", "ts_corr", "ts_cov", "ts_rank",
}


def scan_idea_pool(
    path: Path,
    horizon_hours: int,
    max_lookback_hours: int,
    available_fields: Iterable[str],
) -> dict[str, Any]:
    """Scan every JSON file and group cards validated for one horizon.

    Invalid and non-card JSON files remain visible in ``entries``. A malformed
    JSON file has status ``error``; a well-formed file that does not satisfy the
    FM-v6 card and input requirements has status ``rejected``. Repeated copies
    of one card ID have status ``duplicate`` after the first sorted path.
    """
    if type(horizon_hours) is not int or horizon_hours not in {1, 4, 24}:
        raise ValueError("horizon_hours must be one of 1, 4, or 24")
    if type(max_lookback_hours) is not int or max_lookback_hours < 0:
        raise ValueError("max_lookback_hours must be a non-negative integer")
    if isinstance(available_fields, str):
        raise ValueError("available_fields must be an iterable of field names, not a string")
    try:
        field_values = list(available_fields)
    except TypeError as exc:
        raise ValueError("available_fields must be an iterable of field names") from exc
    if any(not isinstance(field, str) or not field.strip() for field in field_values):
        raise ValueError("available_fields must contain nonempty strings")
    fields = set(field_values)

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_dir():
        files = sorted(
            (item for item in root.rglob("*") if item.is_file() and item.suffix.lower() == ".json"),
            key=lambda item: item.relative_to(root).as_posix(),
        )
        display_root = root
    elif root.is_file() and root.suffix.lower() == ".json":
        files = [root]
        display_root = root.parent
    else:
        raise ValueError(f"path must be a directory or a JSON file: {root}")

    entries: list[dict[str, Any]] = []
    for listed_path in files:
        relative_path = listed_path.relative_to(root).as_posix() if root.is_dir() else listed_path.name
        file_path = listed_path.resolve()
        display_path = str(file_path)
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            entries.append({
                "path": display_path,
                "relative_path": relative_path,
                "status": "error",
                "reasons": [f"unable to read source file: {exc}"],
                "id": None,
                "card_snapshot": None,
                "card_sha256": None,
                "content_sha256": None,
            })
            continue
        card_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        try:
            snapshot = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            entries.append({
                "path": display_path,
                "relative_path": relative_path,
                "status": "error",
                "reasons": [f"unable to read valid JSON: {exc}"],
                "id": None,
                "card_snapshot": None,
                "card_sha256": card_sha256,
                "content_sha256": None,
            })
            continue

        entry: dict[str, Any] = {
            "path": display_path,
            "relative_path": relative_path,
            "status": "rejected",
            "reasons": [],
            "id": snapshot.get("id") if isinstance(snapshot, dict) else None,
            "source": snapshot.get("source") if isinstance(snapshot, dict) else None,
            "card_snapshot": snapshot,
            "card_sha256": card_sha256,
            "content_sha256": _snapshot_sha256(snapshot),
        }
        try:
            validated = read_cards(
                [file_path],
                horizon_hours=horizon_hours,
                max_lookback_hours=max_lookback_hours,
            )[0]
        except ValueError as exc:
            entry["reasons"].append(str(exc))
            entries.append(entry)
            continue
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            entry["status"] = "error"
            entry["reasons"].append(f"unable to validate card JSON: {exc}")
            entries.append(entry)
            continue

        validated_content_sha256 = _snapshot_sha256(validated["card_snapshot"])
        if validated_content_sha256 != entry["content_sha256"]:
            entry["status"] = "error"
            entry["reasons"].append("card contents changed while scan validation was in progress")
            entries.append(entry)
            continue

        missing_fields = sorted(set(validated["fields"]) - fields)
        if missing_fields:
            entry["reasons"].append(f"required input fields are unavailable: {missing_fields}")
            entries.append(entry)
            continue

        entry.update({
            "status": "admitted",
            "id": validated["id"],
            "source": validated["source"],
            "expression": validated["expression"],
            "expanded_expression": compile_expression(validated["expression"]).expanded_expression,
            "direction": validated["direction"],
            "horizon_hours": validated["horizon_hours"],
            "fields": validated["fields"],
            "lookback_hours": validated["lookback_hours"],
            "card_snapshot": validated["card_snapshot"],
            "content_sha256": validated_content_sha256,
        })
        entries.append(entry)

    _mark_duplicate_ids(entries)
    groups = _build_groups([entry for entry in entries if entry["status"] == "admitted"])
    return {
        "path": str(display_root.resolve()),
        "horizon_hours": horizon_hours,
        "max_lookback_hours": max_lookback_hours,
        "available_fields": sorted(fields),
        "counts": {
            "files": len(entries),
            "admitted": sum(entry["status"] == "admitted" for entry in entries),
            "rejected": sum(entry["status"] == "rejected" for entry in entries),
            "error": sum(entry["status"] == "error" for entry in entries),
            "duplicate": sum(entry["status"] == "duplicate" for entry in entries),
            "groups": len(groups),
            "runner_representatives": sum(
                len(group["runner_representatives"]) for group in groups
            ),
        },
        "entries": entries,
        "groups": groups,
    }


def _snapshot_sha256(snapshot: Any) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mark_duplicate_ids(entries: list[dict[str, Any]]) -> None:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        snapshot = entry["card_snapshot"]
        card_id = entry["id"]
        if (
            isinstance(snapshot, dict)
            and snapshot.get("source_type") == "factor_mining"
            and isinstance(card_id, str)
            and card_id.strip()
        ):
            by_id.setdefault(card_id, []).append(entry)

    for card_id, same_id_entries in by_id.items():
        if len(same_id_entries) == 1:
            continue
        hashes = {entry["content_sha256"] for entry in same_id_entries}
        if len(hashes) > 1:
            paths = [entry["path"] for entry in same_id_entries]
            reason = f"conflicting card contents share id {card_id}: {paths}"
            for entry in same_id_entries:
                entry["status"] = "error"
                entry["reasons"].append(reason)
                entry.pop("duplicate_of_path", None)
            continue

        first = same_id_entries[0]
        for duplicate in same_id_entries[1:]:
            duplicate["status"] = "duplicate"
            duplicate["duplicate_of_path"] = first["path"]
            duplicate["reasons"].append(f"duplicate card id; retained {first['path']}")


def _structure_fingerprint(expression: str) -> str:
    tree = compile_expression(expression).tree

    def describe(node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return ["name", node.id]
        if isinstance(node, ast.Constant):
            return ["constant", type(node.value).__name__, node.value]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return ["unary", "minus" if isinstance(node.op, ast.USub) else "plus", describe(node.operand)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            arguments = []
            for index, argument in enumerate(node.args):
                if node.func.id in _TIME_SERIES_OPERATORS and index == len(node.args) - 1:
                    arguments.append(["time_window"])
                else:
                    arguments.append(describe(argument))
            return ["call", node.func.id, arguments]
        raise ValueError(f"unexpected node in compiled factor expression: {type(node).__name__}")

    canonical = json.dumps(describe(tree), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _member_ref(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "path": entry["path"],
        "relative_path": entry["relative_path"],
        "card_sha256": entry["card_sha256"],
        "content_sha256": entry["content_sha256"],
        "expression": entry["expression"],
        "expanded_expression": entry["expanded_expression"],
        "direction": entry["direction"],
        "horizon_hours": entry["horizon_hours"],
        "fields": entry["fields"],
        "lookback_hours": entry["lookback_hours"],
    }


def _representative_key(entry: dict[str, Any]) -> tuple[int, str, str]:
    return entry["lookback_hours"], entry["id"], entry["path"]


def _build_groups(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = {}
    for entry in entries:
        key = entry["horizon_hours"], tuple(entry["fields"])
        by_group.setdefault(key, []).append(entry)

    groups: list[dict[str, Any]] = []
    for (horizon, fields), members in sorted(by_group.items()):
        members.sort(key=_representative_key)
        exact_buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
        family_buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for entry in members:
            exact_buckets.setdefault((entry["expanded_expression"], entry["direction"]), []).append(entry)
            family_buckets.setdefault(
                (_structure_fingerprint(entry["expression"]), entry["direction"]), []
            ).append(entry)

        exact_signals = []
        exact_key_by_member: dict[str, str] = {}
        for (expression, direction), signal_members in sorted(exact_buckets.items()):
            signal_members.sort(key=_representative_key)
            exact_key = _snapshot_sha256({
                "expanded_expression": expression,
                "direction": direction,
            })
            refs = [_member_ref(entry) for entry in signal_members]
            exact_signals.append({
                "key": exact_key,
                "expanded_expression": expression,
                "direction": direction,
                "representative": refs[0],
                "members": refs,
            })
            for entry in signal_members:
                exact_key_by_member[entry["path"]] = exact_key

        structure_families = []
        for (fingerprint, direction), family_members in sorted(family_buckets.items()):
            family_members.sort(key=_representative_key)
            refs = [_member_ref(entry) for entry in family_members]
            family_id = _snapshot_sha256({
                "fingerprint": fingerprint,
                "direction": direction,
            })
            structure_families.append({
                "kind": "structure_family",
                "family_id": family_id,
                "representative_id": refs[0]["id"],
                "representative_path": refs[0]["path"],
                "member_ids": [member["id"] for member in refs],
                "member_paths": [member["path"] for member in refs],
                "fingerprint": fingerprint,
                "direction": direction,
                "representative": refs[0],
                "members": refs,
                "exact_signal_keys": sorted({
                    exact_key_by_member[entry["path"]] for entry in family_members
                }),
            })

        groups.append({
            "horizon_hours": horizon,
            "fields": list(fields),
            "members": [_member_ref(entry) for entry in members],
            "exact_signals": exact_signals,
            "structure_families": structure_families,
            "runner_representatives": [family["representative"] for family in structure_families],
        })
    return groups
