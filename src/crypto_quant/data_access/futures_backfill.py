"""Resumable Binance USD-M USDT perpetual core-data backfill."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from xml.etree import ElementTree


S3_BUCKET_ROOT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PUBLIC_DOWNLOAD_ROOT = "https://data.binance.vision"
S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"

CATEGORY_SPECS: Tuple[Dict[str, str], ...] = (
    {"category": "klines", "period": "monthly", "interval": "1h", "kind": "price"},
    {"category": "markPriceKlines", "period": "monthly", "interval": "1h", "kind": "price"},
    {"category": "indexPriceKlines", "period": "monthly", "interval": "1h", "kind": "price"},
    {"category": "premiumIndexKlines", "period": "monthly", "interval": "1h", "kind": "price"},
    {"category": "fundingRate", "period": "monthly", "interval": "native", "kind": "funding"},
    {"category": "metrics", "period": "daily", "interval": "5m", "kind": "metrics"},
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:24]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _epoch_ms(value: Any) -> int:
    number = int(value)
    return number // 1_000 if abs(number) > 10**15 else number


def _metrics_time_ms(value: str) -> int:
    if str(value).lstrip("-").isdigit():
        return _epoch_ms(value)
    stamp = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1000)


class BinanceUMArchiveTransport:
    """Read-only official S3 listing and public archive transport."""

    def __init__(self, timeout: int = 45) -> None:
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            retry = Retry(
                total=7,
                connect=7,
                read=7,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
            self._local.session = session
        return session

    def _pages(self, prefix: str, delimiter: Optional[str] = None) -> Iterable[ElementTree.Element]:
        marker: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"prefix": prefix, "max-keys": 1000}
            if delimiter:
                params["delimiter"] = delimiter
            if marker:
                params["marker"] = marker
            response = self._session().get(S3_BUCKET_ROOT, params=params, timeout=self.timeout)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            yield root
            if root.findtext(f"{S3_NAMESPACE}IsTruncated", default="false") != "true":
                break
            marker = root.findtext(f"{S3_NAMESPACE}NextMarker")
            if not marker:
                keys = [node.findtext(f"{S3_NAMESPACE}Key") for node in root.findall(f"{S3_NAMESPACE}Contents")]
                marker = keys[-1] if keys else None
            if not marker:
                raise RuntimeError("truncated S3 listing has no continuation key")

    def list_symbols(self, root_prefix: str) -> List[str]:
        symbols: List[str] = []
        for root in self._pages(root_prefix, delimiter="/"):
            for node in root.findall(f"{S3_NAMESPACE}CommonPrefixes/{S3_NAMESPACE}Prefix"):
                if node.text:
                    symbols.append(node.text.rstrip("/").split("/")[-1])
        return sorted({symbol for symbol in symbols if symbol.endswith("USDT")})

    def list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        for root in self._pages(prefix):
            for node in root.findall(f"{S3_NAMESPACE}Contents"):
                key = node.findtext(f"{S3_NAMESPACE}Key") or ""
                if not key.endswith(".zip"):
                    continue
                objects.append(
                    {
                        "key": key,
                        "size": int(node.findtext(f"{S3_NAMESPACE}Size", default="0")),
                        "etag": (node.findtext(f"{S3_NAMESPACE}ETag") or "").strip('"'),
                    }
                )
        return sorted(objects, key=lambda item: item["key"])

    def download(self, key: str) -> bytes:
        response = self._session().get(f"{PUBLIC_DOWNLOAD_ROOT}/{key}", timeout=self.timeout)
        response.raise_for_status()
        return bytes(response.content)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS futures_archive_files (
            id INTEGER PRIMARY KEY,
            source_key TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            etag TEXT,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            issue_count INTEGER NOT NULL,
            processed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS futures_price_bars (
            data_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            quote_volume REAL NOT NULL,
            trades INTEGER NOT NULL,
            taker_buy_base_volume REAL NOT NULL,
            taker_buy_quote_volume REAL NOT NULL,
            source_file_id INTEGER NOT NULL,
            PRIMARY KEY (data_type, symbol, interval, open_time)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS futures_funding_rates (
            symbol TEXT NOT NULL,
            funding_time INTEGER NOT NULL,
            funding_interval_hours INTEGER NOT NULL,
            funding_rate REAL NOT NULL,
            source_file_id INTEGER NOT NULL,
            mark_price REAL,
            PRIMARY KEY (symbol, funding_time)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS futures_metrics (
            symbol TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            sum_open_interest REAL,
            sum_open_interest_value REAL,
            count_toptrader_long_short_ratio REAL,
            sum_toptrader_long_short_ratio REAL,
            count_long_short_ratio REAL,
            sum_taker_long_short_vol_ratio REAL,
            source_file_id INTEGER NOT NULL,
            PRIMARY KEY (symbol, open_time)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS futures_archive_issues (
            id INTEGER PRIMARY KEY,
            source_file_id INTEGER NOT NULL,
            row_number INTEGER,
            open_time INTEGER,
            reason TEXT NOT NULL,
            raw_preview TEXT
        );
        """
    )
    conn.commit()
    funding_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(futures_funding_rates)")
    }
    if "mark_price" not in funding_columns:
        conn.execute("ALTER TABLE futures_funding_rates ADD COLUMN mark_price REAL")
        conn.commit()
    definition_row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='futures_metrics'").fetchone()
    definition = str(definition_row[0]).upper() if definition_row else ""
    if "SUM_OPEN_INTEREST REAL NOT NULL" in definition:
        conn.executescript(
            """
            BEGIN IMMEDIATE;
            ALTER TABLE futures_metrics RENAME TO futures_metrics_strict;
            CREATE TABLE futures_metrics (
                symbol TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                sum_open_interest REAL,
                sum_open_interest_value REAL,
                count_toptrader_long_short_ratio REAL,
                sum_toptrader_long_short_ratio REAL,
                count_long_short_ratio REAL,
                sum_taker_long_short_vol_ratio REAL,
                source_file_id INTEGER NOT NULL,
                PRIMARY KEY (symbol, open_time)
            ) WITHOUT ROWID;
            INSERT INTO futures_metrics SELECT * FROM futures_metrics_strict;
            DROP TABLE futures_metrics_strict;
            COMMIT;
            """
        )


def _csv_rows(raw: bytes) -> List[List[str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if not names:
            return []
        with archive.open(names[0]) as handle:
            rows = list(csv.reader(io.TextIOWrapper(handle, encoding="utf-8")))
    return rows


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _parse_price(category: str, symbol: str, raw: bytes) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
    output: List[Tuple[Any, ...]] = []
    issues: List[Dict[str, Any]] = []
    rows = _csv_rows(raw)
    if rows and rows[0] and not rows[0][0].lstrip("-").isdigit():
        rows = rows[1:]
    for position, row in enumerate(rows, start=2):
        open_time: Optional[int] = None
        try:
            if len(row) < 12:
                raise ValueError("short futures kline row")
            open_time = _epoch_ms(row[0])
            close_time = _epoch_ms(row[6])
            values = [float(row[index]) for index in (1, 2, 3, 4, 5, 7, 9, 10)]
            open_price, high, low, close, volume, quote_volume, taker_base, taker_quote = values
            trades = int(float(row[8]))
            if not _finite(values):
                raise ValueError("non-finite futures kline value")
            if open_time < 0 or open_time % 3_600_000 != 0:
                raise ValueError("unaligned futures 1h open_time")
            duration = close_time - open_time
            if duration <= 0 or duration > 3_605_000:
                raise ValueError("invalid futures 1h close_time")
            if category != "premiumIndexKlines" and min(open_price, high, low, close) <= 0:
                raise ValueError("non-positive futures price")
            if high < max(open_price, close, low) or low > min(open_price, close, high):
                raise ValueError("invalid futures OHLC range")
            if min(volume, quote_volume, taker_base, taker_quote) < 0 or trades < 0:
                raise ValueError("negative futures volume or trade count")
            output.append((category, symbol, "1h", open_time, open_price, high, low, close, volume, close_time, quote_volume, trades, taker_base, taker_quote))
        except (TypeError, ValueError, IndexError) as exc:
            issues.append({"row_number": position, "open_time": open_time, "reason": str(exc), "raw_preview": ",".join(row[:12])[:500]})
    return output, issues


def _parse_funding(symbol: str, raw: bytes) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
    output: List[Tuple[Any, ...]] = []
    issues: List[Dict[str, Any]] = []
    rows = _csv_rows(raw)
    if rows and rows[0] and not rows[0][0].lstrip("-").isdigit():
        rows = rows[1:]
    for position, row in enumerate(rows, start=2):
        stamp: Optional[int] = None
        try:
            if len(row) < 3:
                raise ValueError("short funding row")
            stamp = _epoch_ms(row[0])
            interval = int(float(row[1]))
            rate = float(row[2])
            if stamp < 0 or interval <= 0 or not math.isfinite(rate):
                raise ValueError("invalid funding value")
            output.append((symbol, stamp, interval, rate))
        except (TypeError, ValueError, IndexError) as exc:
            issues.append({"row_number": position, "open_time": stamp, "reason": str(exc), "raw_preview": ",".join(row[:3])[:500]})
    return output, issues


def _parse_metrics(symbol: str, raw: bytes) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
    output: List[Tuple[Any, ...]] = []
    issues: List[Dict[str, Any]] = []
    rows = _csv_rows(raw)
    if rows and rows[0] and rows[0][0] == "create_time":
        rows = rows[1:]
    for position, row in enumerate(rows, start=2):
        stamp: Optional[int] = None
        try:
            if len(row) < 8:
                raise ValueError("short futures metrics row")
            stamp = _metrics_time_ms(row[0])
            row_symbol = str(row[1]).upper()
            if row_symbol != symbol:
                raise ValueError(f"metrics symbol {row_symbol} does not match {symbol}")
            values_list: List[Optional[float]] = []
            for raw_value in row[2:8]:
                if str(raw_value).strip() == "":
                    values_list.append(None)
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    values_list.append(None)
                    continue
                values_list.append(value)
            values = tuple(values_list)
            output.append((symbol, stamp, *values))
        except (TypeError, ValueError, IndexError) as exc:
            issues.append({"row_number": position, "open_time": stamp, "reason": str(exc), "raw_preview": ",".join(row[:8])[:500]})
    return output, issues


def _fetch_parse(transport: Any, task: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    raw = transport.download(item["key"])
    digest = hashlib.sha256(raw).hexdigest()
    if task["kind"] == "price":
        rows, issues = _parse_price(task["category"], task["symbol"], raw)
    elif task["kind"] == "funding":
        rows, issues = _parse_funding(task["symbol"], raw)
    else:
        rows, issues = _parse_metrics(task["symbol"], raw)
    return {"object": item, "sha256": digest, "rows": rows, "issues": issues}


def _task_state_path(output_root: Path, plan_id: str, task_id: str) -> Path:
    return Path(output_root) / "futures_backfill" / plan_id / "tasks" / f"{task_id}.json"


def _status_path(output_root: Path, plan_id: str) -> Path:
    return Path(output_root) / "futures_backfill" / plan_id / "status.json"


def build_um_futures_core_plan(
    output_root: Path = Path("experiments"),
    transport: Any = None,
    symbols: Optional[Sequence[str]] = None,
    max_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    """Discover every official archived USDT USD-M core-data prefix."""
    transport = transport or BinanceUMArchiveTransport()
    requested = {str(symbol).upper() for symbol in symbols or []}
    tasks: List[Dict[str, Any]] = []
    inventory: Dict[str, List[str]] = {}
    for spec in CATEGORY_SPECS:
        root = f"data/futures/um/{spec['period']}/{spec['category']}/"
        discovered = transport.list_symbols(root)
        if requested:
            discovered = [symbol for symbol in discovered if symbol in requested]
        if max_symbols is not None:
            discovered = discovered[: max(0, int(max_symbols))]
        inventory[spec["category"]] = discovered
        for symbol in discovered:
            suffix = f"{symbol}/{spec['interval']}/" if spec["kind"] == "price" else f"{symbol}/"
            identity = {"category": spec["category"], "symbol": symbol, "interval": spec["interval"]}
            tasks.append({"task_id": _digest("umt_", identity), **identity, "period": spec["period"], "kind": spec["kind"], "prefix": root + suffix})
    plan_identity = {"market": "futures_um", "quote": "USDT", "tasks": [task["task_id"] for task in tasks]}
    plan_id = _digest("umcore_", plan_identity)
    payload = {
        "plan_id": plan_id,
        "market": "binance_usdt_m_perpetual_archive",
        "scope": "all_historical_usdt_suffix_contracts",
        "created_at": _now(),
        "status": "planned",
        "task_count": len(tasks),
        "tasks": tasks,
        "inventory": inventory,
        "raw_zip_retention": False,
        "source": S3_BUCKET_ROOT,
    }
    root = Path(output_root) / "futures_backfill"
    _atomic_json(root / f"{plan_id}.json", payload)
    ready = 0
    for task in tasks:
        path = _task_state_path(Path(output_root), plan_id, task["task_id"])
        if path.exists():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete":
                    ready += 1
                    continue
            except (OSError, ValueError):
                pass
        _atomic_json(path, {"task_id": task["task_id"], "status": "requested", "files_completed": 0, "rows_stored": 0, "issues": 0, "updated_at": _now()})
    _atomic_json(_status_path(Path(output_root), plan_id), {"plan_id": plan_id, "status": "complete" if ready == len(tasks) else "planned", "task_count": len(tasks), "ready_count": ready, "failed_count": 0, "current_task_index": None, "last_error": None, "updated_at": _now()})
    return payload


def _store_artifacts(conn: sqlite3.Connection, task: Dict[str, Any], artifacts: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    rows_stored = issues_stored = bytes_stored = 0
    with conn:
        for artifact in artifacts:
            item = artifact["object"]
            cursor = conn.execute(
                """INSERT INTO futures_archive_files
                   (source_key,category,symbol,interval,byte_size,etag,sha256,row_count,issue_count,processed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_key) DO UPDATE SET sha256=excluded.sha256, byte_size=excluded.byte_size,
                       etag=excluded.etag, row_count=excluded.row_count, issue_count=excluded.issue_count,
                       processed_at=excluded.processed_at""",
                (item["key"], task["category"], task["symbol"], task["interval"], int(item.get("size", 0)), item.get("etag"), artifact["sha256"], len(artifact["rows"]), len(artifact["issues"]), _now()),
            )
            file_id = cursor.lastrowid or conn.execute("SELECT id FROM futures_archive_files WHERE source_key=?", (item["key"],)).fetchone()[0]
            if task["kind"] == "price":
                conn.executemany("INSERT OR REPLACE INTO futures_price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(*row, file_id) for row in artifact["rows"]])
            elif task["kind"] == "funding":
                conn.executemany(
                    """INSERT INTO futures_funding_rates
                       (symbol,funding_time,funding_interval_hours,funding_rate,source_file_id)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT(symbol,funding_time) DO UPDATE SET
                           funding_interval_hours=excluded.funding_interval_hours,
                           funding_rate=excluded.funding_rate,
                           source_file_id=excluded.source_file_id""",
                    [(*row, file_id) for row in artifact["rows"]],
                )
            else:
                conn.executemany("INSERT OR REPLACE INTO futures_metrics VALUES (?,?,?,?,?,?,?,?,?)", [(*row, file_id) for row in artifact["rows"]])
            conn.executemany(
                "INSERT INTO futures_archive_issues (source_file_id,row_number,open_time,reason,raw_preview) VALUES (?,?,?,?,?)",
                [(file_id, issue.get("row_number"), issue.get("open_time"), issue["reason"], issue.get("raw_preview")) for issue in artifact["issues"]],
            )
            rows_stored += len(artifact["rows"])
            issues_stored += len(artifact["issues"])
            bytes_stored += int(item.get("size", 0))
    return rows_stored, issues_stored, bytes_stored


def run_um_futures_core_plan(
    plan_id: str,
    db_path: Path = Path("market_data/crypto_quant.sqlite"),
    output_root: Path = Path("experiments"),
    transport: Any = None,
    authorize: bool = False,
    workers: int = 8,
    chunk_files: int = 24,
    max_tasks: Optional[int] = None,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Run or resume the official archive backfill."""
    if not authorize and os.environ.get("AUTHORIZED_PUBLIC_DOWNLOAD") != "1":
        raise PermissionError("USD-M core backfill requires AUTHORIZED_PUBLIC_DOWNLOAD=1")
    output_root = Path(output_root)
    plan_path = output_root / "futures_backfill" / f"{plan_id}.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    transport = transport or BinanceUMArchiveTransport()
    conn = _connect(Path(db_path))
    executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    states: Dict[str, Dict[str, Any]] = {}
    for task in plan["tasks"]:
        path = _task_state_path(output_root, plan_id, task["task_id"])
        states[task["task_id"]] = json.loads(path.read_text(encoding="utf-8"))
    ready = sum(state.get("status") == "complete" for state in states.values())
    status = {"plan_id": plan_id, "status": "running", "task_count": len(plan["tasks"]), "ready_count": ready, "failed_count": 0, "current_task_index": None, "last_error": None, "updated_at": _now()}
    _atomic_json(_status_path(output_root, plan_id), status)
    tasks_processed = files_processed = 0
    try:
        for index, task in enumerate(plan["tasks"]):
            state_path = _task_state_path(output_root, plan_id, task["task_id"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                continue
            if max_tasks is not None and tasks_processed >= max(0, int(max_tasks)):
                break
            if max_files is not None and files_processed >= max(0, int(max_files)):
                break
            status.update({"status": "running", "current_task_index": index, "current_task_id": task["task_id"], "last_error": None, "updated_at": _now()})
            _atomic_json(_status_path(output_root, plan_id), status)
            state.update({"status": "running", "last_error": None, "updated_at": _now()})
            _atomic_json(state_path, state)
            objects = transport.list_objects(task["prefix"])
            completed = {row[0] for row in conn.execute("SELECT source_key FROM futures_archive_files WHERE category=? AND symbol=? AND interval=?", (task["category"], task["symbol"], task["interval"]))}
            pending = [item for item in objects if item["key"] not in completed]
            if max_files is not None:
                pending = pending[: max(0, int(max_files) - files_processed)]
            for offset in range(0, len(pending), max(1, int(chunk_files))):
                chunk = pending[offset : offset + max(1, int(chunk_files))]
                artifacts = list(executor.map(lambda item: _fetch_parse(transport, task, item), chunk))
                rows, issues, byte_size = _store_artifacts(conn, task, artifacts)
                files_processed += len(chunk)
                state["files_completed"] = int(state.get("files_completed", 0)) + len(chunk)
                state["rows_stored"] = int(state.get("rows_stored", 0)) + rows
                state["issues"] = int(state.get("issues", 0)) + issues
                state["compressed_bytes"] = int(state.get("compressed_bytes", 0)) + byte_size
                state["files_total"] = len(objects)
                state["updated_at"] = _now()
                _atomic_json(state_path, state)
            remaining = conn.execute("SELECT COUNT(*) FROM futures_archive_files WHERE category=? AND symbol=? AND interval=?", (task["category"], task["symbol"], task["interval"])).fetchone()[0]
            if remaining >= len(objects):
                state["status"] = "complete"
                state["updated_at"] = _now()
                _atomic_json(state_path, state)
                tasks_processed += 1
                ready += 1
                status["ready_count"] = ready
            else:
                state["status"] = "partial"
                state["updated_at"] = _now()
                _atomic_json(state_path, state)
                break
            status["updated_at"] = _now()
            _atomic_json(_status_path(output_root, plan_id), status)
    except Exception as exc:
        status["status"] = "failed"
        status["failed_count"] = 1
        status["last_error"] = str(exc)
        status["updated_at"] = _now()
        _atomic_json(_status_path(output_root, plan_id), status)
        return status
    finally:
        executor.shutdown(wait=True)
        conn.close()
    status["ready_count"] = ready
    status["remaining_count"] = len(plan["tasks"]) - ready
    status["status"] = "complete" if ready == len(plan["tasks"]) else "partial"
    status["updated_at"] = _now()
    _atomic_json(_status_path(output_root, plan_id), status)
    return status


def um_futures_core_status(plan_id: str, db_path: Path = Path("market_data/crypto_quant.sqlite"), output_root: Path = Path("experiments")) -> Dict[str, Any]:
    status = json.loads(_status_path(Path(output_root), plan_id).read_text(encoding="utf-8"))
    if Path(db_path).exists():
        conn = _connect(Path(db_path))
        rows = conn.execute("SELECT category, COUNT(*), COALESCE(SUM(row_count),0), COALESCE(SUM(byte_size),0), COALESCE(SUM(issue_count),0) FROM futures_archive_files GROUP BY category ORDER BY category").fetchall()
        conn.close()
        status["categories"] = {row[0]: {"files": int(row[1]), "rows": int(row[2]), "compressed_bytes": int(row[3]), "issues": int(row[4])} for row in rows}
    return status


def repair_partial_metrics(
    plan_id: str,
    db_path: Path = Path("market_data/crypto_quant.sqlite"),
    output_root: Path = Path("experiments"),
) -> Dict[str, Any]:
    """Reset metric archives whose optional empty fields were previously rejected."""
    output_root = Path(output_root)
    plan = json.loads((output_root / "futures_backfill" / f"{plan_id}.json").read_text(encoding="utf-8"))
    conn = _connect(Path(db_path))
    affected = conn.execute("SELECT id,symbol FROM futures_archive_files WHERE category='metrics' AND issue_count>0").fetchall()
    affected_ids = [int(row[0]) for row in affected]
    affected_symbols = sorted({str(row[1]) for row in affected})
    if affected_ids:
        with conn:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS affected_metric_files (id INTEGER PRIMARY KEY)")
            conn.execute("DELETE FROM affected_metric_files")
            conn.executemany("INSERT INTO affected_metric_files VALUES (?)", [(item,) for item in affected_ids])
            conn.execute("DELETE FROM futures_metrics WHERE source_file_id IN (SELECT id FROM affected_metric_files)")
            conn.execute("DELETE FROM futures_archive_issues WHERE source_file_id IN (SELECT id FROM affected_metric_files)")
            conn.execute("DELETE FROM futures_archive_files WHERE id IN (SELECT id FROM affected_metric_files)")
    conn.close()
    tasks_by_symbol = {task["symbol"]: task for task in plan["tasks"] if task["category"] == "metrics"}
    for symbol in affected_symbols:
        task = tasks_by_symbol.get(symbol)
        if not task:
            continue
        state_path = _task_state_path(output_root, plan_id, task["task_id"])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"status": "requested", "files_completed": 0, "rows_stored": 0, "issues": 0, "compressed_bytes": 0, "last_error": None, "updated_at": _now()})
        _atomic_json(state_path, state)
    states = [json.loads(_task_state_path(output_root, plan_id, task["task_id"]).read_text(encoding="utf-8")) for task in plan["tasks"]]
    ready = sum(state.get("status") == "complete" for state in states)
    status = {"plan_id": plan_id, "status": "planned", "task_count": len(states), "ready_count": ready, "failed_count": 0, "current_task_index": None, "last_error": None, "updated_at": _now()}
    _atomic_json(_status_path(output_root, plan_id), status)
    return {"status": "reset", "affected_files": len(affected_ids), "affected_symbols": len(affected_symbols), "ready_count": ready}
