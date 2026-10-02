"""Append-only, per-factor SQLite evidence archives.

Each archive is identified by its expanded expression, direction, and explicit
expression-semantics version. A small registry assigns each identity a stable,
sequential file number. Evaluation inputs are immutable and compressed; large
factor-value tables remain available through stable pages.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class ArchiveIntegrityError(ValueError):
    """An archive's identity or stored evidence is structurally invalid."""


class EvaluationConflictError(ValueError):
    """An evaluation key already exists with different immutable evidence."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("archive evidence must be finite JSON data") from exc


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class FactorIdentity:
    expanded_expression: str
    direction: str
    semantics_version: str

    def __post_init__(self) -> None:
        _required_text(self.expanded_expression, "expanded_expression")
        _required_text(self.direction, "direction")
        _required_text(self.semantics_version, "semantics_version")

    def as_dict(self) -> dict[str, str]:
        return {"expanded_expression": self.expanded_expression,
                "direction": self.direction, "semantics_version": self.semantics_version}


@dataclass(frozen=True)
class EvaluationKey:
    data_version: str
    contract_version: str
    evaluator_version: str
    segment: str
    horizon: str

    def __post_init__(self) -> None:
        for field in ("data_version", "contract_version", "evaluator_version", "segment", "horizon"):
            _required_text(getattr(self, field), field)

    def as_dict(self) -> dict[str, str]:
        return {"data_version": self.data_version,
                "contract_version": self.contract_version,
                "evaluator_version": self.evaluator_version,
                "segment": self.segment, "horizon": self.horizon}


class FactorArchive:
    """One factor's append-only research evidence, stored in one SQLite file."""

    _SCHEMA_VERSION = 2
    _REGISTRY_SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, identity: FactorIdentity):
        self.path = Path(path)
        self.identity = identity
        self.factor_id: int | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.stat().st_size:
            self._validate_existing()
        else:
            self._initialize()

    @classmethod
    def open_for(cls, root: str | Path, identity: FactorIdentity) -> "FactorArchive":
        """Open or create the sequentially named archive for an identity."""
        root = Path(root)
        factor_id = cls._register_identity(root, identity)
        archive = cls(root / f"factor-{factor_id:06d}.sqlite3", identity)
        archive.factor_id = factor_id
        return archive

    @classmethod
    def open_existing(cls, root: str | Path, identity: FactorIdentity) -> "FactorArchive":
        """Open an existing archive read/write without creating a missing one."""
        root = Path(root)
        factor_id = cls._find_registered_identity(root, identity)
        path = root / f"factor-{factor_id:06d}.sqlite3"
        if not path.is_file():
            raise FileNotFoundError(path)
        archive = cls.__new__(cls)
        archive.path = path
        archive.identity = identity
        archive.factor_id = factor_id
        archive._validate_existing()
        return archive

    @classmethod
    def _register_identity(cls, root: Path, identity: FactorIdentity) -> int:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "registry.sqlite3"
        connection = sqlite3.connect(path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS registry_meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS factor_identities (
                    factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expanded_expression TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    semantics_version TEXT NOT NULL,
                    UNIQUE(expanded_expression, direction, semantics_version)
                );
            """)
            connection.execute("BEGIN IMMEDIATE")
            meta = connection.execute(
                "SELECT schema_version FROM registry_meta WHERE singleton=1").fetchone()
            if meta is None:
                connection.execute("INSERT INTO registry_meta VALUES (1, ?)",
                                   (cls._REGISTRY_SCHEMA_VERSION,))
            elif meta[0] != cls._REGISTRY_SCHEMA_VERSION:
                raise ArchiveIntegrityError("factor registry has an unsupported schema")
            identity_values = (identity.expanded_expression, identity.direction,
                               identity.semantics_version)
            row = connection.execute("""SELECT factor_id FROM factor_identities
                WHERE expanded_expression=? AND direction=? AND semantics_version=?""",
                identity_values).fetchone()
            if row is None:
                cursor = connection.execute("""INSERT INTO factor_identities
                    (expanded_expression, direction, semantics_version) VALUES (?, ?, ?)""",
                    identity_values)
                factor_id = int(cursor.lastrowid)
            else:
                factor_id = int(row[0])
            connection.commit()
            return factor_id
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise ArchiveIntegrityError("factor registry is invalid") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def _find_registered_identity(cls, root: Path, identity: FactorIdentity) -> int:
        path = root / "registry.sqlite3"
        if not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            meta = connection.execute(
                "SELECT schema_version FROM registry_meta WHERE singleton=1").fetchone()
            if meta is None or meta[0] != cls._REGISTRY_SCHEMA_VERSION:
                raise ArchiveIntegrityError("factor registry has an unsupported schema")
            row = connection.execute("""SELECT factor_id FROM factor_identities
                WHERE expanded_expression=? AND direction=? AND semantics_version=?""",
                (identity.expanded_expression, identity.direction,
                 identity.semantics_version)).fetchone()
        except sqlite3.DatabaseError as exc:
            raise ArchiveIntegrityError("factor registry is invalid") from exc
        finally:
            connection.close()
        if row is None:
            raise FileNotFoundError(f"factor identity is not registered in {path}")
        return int(row[0])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level="DEFERRED")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        identity_json = _canonical(self.identity.as_dict()).decode("utf-8")
        with self._connection() as db:
            db.execute("PRAGMA journal_mode = DELETE")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS archive_meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL,
                    identity_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blobs (
                    blob_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_size INTEGER NOT NULL CHECK(raw_size >= 0),
                    data_zlib BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS value_sets (
                    value_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_version TEXT NOT NULL,
                    computation_semantics TEXT NOT NULL,
                    blob_id INTEGER NOT NULL REFERENCES blobs(blob_id),
                    UNIQUE(data_version, computation_semantics)
                );
                CREATE TABLE IF NOT EXISTS value_set_sources (
                    value_set_id INTEGER NOT NULL REFERENCES value_sets(value_set_id),
                    source_json TEXT NOT NULL,
                    PRIMARY KEY(value_set_id, source_json)
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_version TEXT NOT NULL,
                    contract_version TEXT NOT NULL,
                    evaluator_version TEXT NOT NULL,
                    segment TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    payload_blob_id INTEGER NOT NULL REFERENCES blobs(blob_id),
                    value_set_id INTEGER REFERENCES value_sets(value_set_id),
                    factor_values_csv_blob_id INTEGER REFERENCES blobs(blob_id),
                    factor_value_count INTEGER NOT NULL CHECK(factor_value_count >= 0),
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    UNIQUE(data_version, contract_version, evaluator_version, segment, horizon)
                );
                CREATE TABLE IF NOT EXISTS evaluation_sources (
                    evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id),
                    source_json TEXT NOT NULL,
                    PRIMARY KEY(evaluation_id, source_json)
                );
                CREATE TABLE IF NOT EXISTS factor_values (
                    evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id),
                    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                    row_json TEXT NOT NULL,
                    PRIMARY KEY(evaluation_id, ordinal)
                );
            """)
            row = db.execute(
                "SELECT schema_version, identity_json FROM archive_meta WHERE singleton=1").fetchone()
            if row is None:
                db.execute("INSERT INTO archive_meta VALUES (1, ?, ?)",
                           (self._SCHEMA_VERSION, identity_json))
            elif row["schema_version"] != self._SCHEMA_VERSION or row["identity_json"] != identity_json:
                raise ArchiveIntegrityError("SQLite file belongs to a different factor identity or schema")

    def _validate_existing(self) -> None:
        try:
            connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute(
                    "SELECT schema_version, identity_json FROM archive_meta WHERE singleton=1").fetchone()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ArchiveIntegrityError("existing factor archive has no valid metadata") from exc
        identity_json = _canonical(self.identity.as_dict()).decode("utf-8")
        if row is None or row["schema_version"] != self._SCHEMA_VERSION or row["identity_json"] != identity_json:
            raise ArchiveIntegrityError("existing SQLite file belongs to a different factor identity or schema")

    @staticmethod
    def _normalize_values(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in values:
            if not isinstance(row, Mapping):
                raise ValueError("factor value rows must be mappings")
            normalized = json.loads(_canonical(dict(row)).decode("utf-8"))
            result.append(normalized)
        return result

    @staticmethod
    def _put_blob(db: sqlite3.Connection, raw: bytes) -> int:
        cursor = db.execute("INSERT INTO blobs (raw_size, data_zlib) VALUES (?, ?)",
                            (len(raw), zlib.compress(raw, level=9)))
        return int(cursor.lastrowid)

    @staticmethod
    def _load_blob(db: sqlite3.Connection, blob_id: int) -> bytes:
        row = db.execute("SELECT raw_size,data_zlib FROM blobs WHERE blob_id=?", (blob_id,)).fetchone()
        if row is None:
            raise ArchiveIntegrityError("referenced archive content is missing")
        return FactorArchive._read_blob(row["raw_size"], row["data_zlib"])

    def _append_value_set_db(self, db: sqlite3.Connection, data_version: str,
                             computation_semantics: str, raw: bytes,
                             provenance: Mapping[str, Any]) -> dict[str, str]:
        _required_text(data_version, "data_version")
        _required_text(computation_semantics, "computation_semantics")
        provenance_json = _canonical(dict(provenance)).decode("utf-8")
        existing = db.execute("""SELECT value_set_id,blob_id FROM value_sets
            WHERE data_version=? AND computation_semantics=?""",
            (data_version, computation_semantics)).fetchone()
        if existing:
            if self._load_blob(db, existing["blob_id"]) != raw:
                raise EvaluationConflictError("factor value set key already exists with different bytes")
            value_set_id = int(existing["value_set_id"])
        else:
            blob_id = self._put_blob(db, raw)
            cursor = db.execute("""INSERT INTO value_sets
                (data_version, computation_semantics, blob_id) VALUES (?, ?, ?)""",
                (data_version, computation_semantics, blob_id))
            value_set_id = int(cursor.lastrowid)
        self._append_source_db(db, "value_set_sources", "value_set_id", value_set_id,
                               provenance_json)
        return {"value_set_id": str(value_set_id)}

    @staticmethod
    def _append_source_db(db: sqlite3.Connection, table: str, foreign_key: str,
                          record_id: int, source_json: str) -> None:
        if table not in {"value_set_sources", "evaluation_sources"} or foreign_key not in {"value_set_id", "evaluation_id"}:
            raise ValueError("invalid source table")
        db.execute(f"INSERT OR IGNORE INTO {table} ({foreign_key}, source_json) VALUES (?, ?)",
                   (record_id, source_json))

    @staticmethod
    def _id_value(value: str | int, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a positive sequential ID")
        try:
            identifier = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a positive sequential ID") from exc
        if identifier < 1:
            raise ValueError(f"{label} must be a positive sequential ID")
        return identifier

    def append_value_set(self, data_version: str, computation_semantics: str,
                         factor_values_csv: bytes, *,
                         provenance: Mapping[str, Any] | None = None) -> dict[str, str]:
        """Persist calculated values before evaluation, keyed by their versions."""
        if not isinstance(factor_values_csv, bytes):
            raise TypeError("factor_values_csv must be bytes")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._append_value_set_db(db, data_version, computation_semantics,
                factor_values_csv, {} if provenance is None else provenance)

    def get_value_set(self, data_version: str, computation_semantics: str) -> dict[str, Any]:
        """Read an exact persisted value set by its calculation inputs."""
        _required_text(data_version, "data_version")
        _required_text(computation_semantics, "computation_semantics")
        with self._connection() as db:
            row = db.execute("""SELECT v.value_set_id,v.blob_id,b.raw_size,b.data_zlib
                FROM value_sets v JOIN blobs b ON b.blob_id=v.blob_id
                WHERE v.data_version=? AND v.computation_semantics=?""",
                (data_version, computation_semantics)).fetchone()
            if row is None:
                raise KeyError((data_version, computation_semantics))
            provenance_rows = db.execute("""SELECT source_json FROM value_set_sources
                WHERE value_set_id=? ORDER BY source_json""", (row["value_set_id"],)).fetchall()
        return {"value_set_id": str(row["value_set_id"]),
                "factor_values_csv": self._read_blob(row["raw_size"], row["data_zlib"]),
                "provenance": self._decode_sources(provenance_rows)}

    @staticmethod
    def _decode_sources(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [json.loads(row["source_json"]) for row in rows]

    @staticmethod
    def _read_blob(raw_size: int, compressed: bytes) -> bytes:
        try:
            raw = zlib.decompress(compressed)
        except zlib.error as exc:
            raise ArchiveIntegrityError("compressed archive content is invalid") from exc
        if len(raw) != raw_size:
            raise ArchiveIntegrityError("stored content size does not match its metadata")
        return raw

    @staticmethod
    def _key_columns() -> tuple[str, ...]:
        return ("data_version", "contract_version", "evaluator_version", "segment", "horizon")

    @classmethod
    def _key_predicate(cls) -> str:
        return " AND ".join(f"{field}=?" for field in cls._key_columns())

    @staticmethod
    def _key_values(key: EvaluationKey) -> tuple[str, ...]:
        fields = key.as_dict()
        return tuple(fields[name] for name in FactorArchive._key_columns())

    def append_evaluation(self, key: EvaluationKey, payload: Any, *,
                          factor_values: Iterable[Mapping[str, Any]] = (),
                          factor_values_csv: bytes | None = None,
                          value_set_id: str | None = None,
                          provenance: Mapping[str, Any] | None = None) -> dict[str, str]:
        """Append evidence atomically; duplicate identical writes are idempotent."""
        if not isinstance(key, EvaluationKey):
            raise TypeError("key must be an EvaluationKey")
        provenance_obj = {} if provenance is None else dict(provenance)
        values = self._normalize_values(factor_values)
        payload_raw = _canonical(payload)
        provenance_raw = _canonical(provenance_obj).decode("utf-8")
        if factor_values_csv is not None and not isinstance(factor_values_csv, bytes):
            raise TypeError("factor_values_csv must be bytes")
        requested_csv = factor_values_csv
        value_set_id_int: int | None = None
        csv_blob_id: int | None = None

        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if factor_values_csv is not None:
                value_set = self._append_value_set_db(db, key.data_version, key.evaluator_version,
                    factor_values_csv, provenance_obj)
                actual_value_set_id = self._id_value(value_set["value_set_id"], "value_set_id")
                if value_set_id is not None and self._id_value(value_set_id, "value_set_id") != actual_value_set_id:
                    raise ValueError("value_set_id does not match factor_values_csv")
                value_set_id_int = actual_value_set_id
                value_set_row = db.execute("SELECT blob_id FROM value_sets WHERE value_set_id=?",
                                           (value_set_id_int,)).fetchone()
                csv_blob_id = int(value_set_row["blob_id"])
            elif value_set_id is not None:
                value_set_id_int = self._id_value(value_set_id, "value_set_id")
                value_set = db.execute("""SELECT data_version,computation_semantics,blob_id
                    FROM value_sets WHERE value_set_id=?""", (value_set_id_int,)).fetchone()
                if value_set is None or value_set["data_version"] != key.data_version or \
                        value_set["computation_semantics"] != key.evaluator_version:
                    raise ValueError("value_set_id does not match evaluation data and evaluator version")
                csv_blob_id = int(value_set["blob_id"])
                requested_csv = self._load_blob(db, csv_blob_id)

            existing = db.execute(f"SELECT * FROM evaluations WHERE {self._key_predicate()}",
                                  self._key_values(key)).fetchone()
            if existing:
                old_payload = self._load_blob(db, existing["payload_blob_id"])
                old_values = [row["row_json"] for row in db.execute(
                    "SELECT row_json FROM factor_values WHERE evaluation_id=? ORDER BY ordinal",
                    (existing["evaluation_id"],))]
                old_csv = None
                if existing["factor_values_csv_blob_id"] is not None:
                    old_csv = self._load_blob(db, existing["factor_values_csv_blob_id"])
                new_values = [_canonical(row).decode("utf-8") for row in values]
                if (old_payload != payload_raw or old_values != new_values or
                        existing["factor_value_count"] != len(values) or old_csv != requested_csv):
                    raise EvaluationConflictError("evaluation key already exists with different evidence")
                evaluation_id = int(existing["evaluation_id"])
                self._append_source_db(db, "evaluation_sources", "evaluation_id", evaluation_id,
                                       provenance_raw)
                return {"evaluation_id": str(evaluation_id),
                        "value_set_id": None if existing["value_set_id"] is None
                        else str(existing["value_set_id"])}

            payload_blob_id = self._put_blob(db, payload_raw)
            cursor = db.execute("""INSERT INTO evaluations
                (data_version, contract_version, evaluator_version, segment, horizon,
                 payload_blob_id, value_set_id, factor_values_csv_blob_id, factor_value_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (*self._key_values(key), payload_blob_id, value_set_id_int, csv_blob_id, len(values)))
            evaluation_id = int(cursor.lastrowid)
            self._append_source_db(db, "evaluation_sources", "evaluation_id", evaluation_id,
                                   provenance_raw)
            db.executemany("INSERT INTO factor_values VALUES (?, ?, ?)",
                ((evaluation_id, i, _canonical(row).decode("utf-8"))
                 for i, row in enumerate(values)))
        return {"evaluation_id": str(evaluation_id),
                "value_set_id": None if value_set_id_int is None else str(value_set_id_int)}

    def _evaluation_for_key(self, db: sqlite3.Connection, key: EvaluationKey) -> sqlite3.Row | None:
        return db.execute(f"SELECT * FROM evaluations WHERE {self._key_predicate()}",
                          self._key_values(key)).fetchone()

    def get_factor_values_csv(self, key: EvaluationKey) -> bytes:
        """Read the original factor-values CSV bytes."""
        with self._connection() as db:
            row = self._evaluation_for_key(db, key)
            if row is None:
                raise KeyError(key.as_dict())
            if row["factor_values_csv_blob_id"] is None:
                raise KeyError("evaluation has no factor-values CSV")
            return self._load_blob(db, row["factor_values_csv_blob_id"])

    def get_evaluation(self, key: EvaluationKey) -> dict[str, Any]:
        """Retrieve complete evidence for an exact natural key."""
        with self._connection() as db:
            row = self._evaluation_for_key(db, key)
            if row is None:
                raise KeyError(key.as_dict())
            rows = db.execute("""SELECT ordinal,row_json FROM factor_values
                WHERE evaluation_id=? ORDER BY ordinal""", (row["evaluation_id"],)).fetchall()
            source_rows = db.execute("""SELECT source_json FROM evaluation_sources
                WHERE evaluation_id=? ORDER BY source_json""", (row["evaluation_id"],)).fetchall()
            payload_raw = self._load_blob(db, row["payload_blob_id"])
            if row["factor_values_csv_blob_id"] is not None:
                self._load_blob(db, row["factor_values_csv_blob_id"])
            if row["value_set_id"] is not None:
                value_set = db.execute("""SELECT data_version,computation_semantics
                    FROM value_sets WHERE value_set_id=?""", (row["value_set_id"],)).fetchone()
                if value_set is None or value_set["data_version"] != key.data_version or \
                        value_set["computation_semantics"] != key.evaluator_version:
                    raise ArchiveIntegrityError("evaluation value-set reference does not match")

        values = []
        for expected, value_row in enumerate(rows):
            if value_row["ordinal"] != expected:
                raise ArchiveIntegrityError("factor value rows are missing or reordered")
            values.append(json.loads(value_row["row_json"]))
        payload = json.loads(payload_raw.decode("utf-8"))
        if len(values) != row["factor_value_count"]:
            raise ArchiveIntegrityError("factor value rows are missing")
        return {"evaluation_id": str(row["evaluation_id"]),
                "value_set_id": None if row["value_set_id"] is None else str(row["value_set_id"]),
                "key": key.as_dict(),
                "payload": payload, "provenance": self._decode_sources(source_rows),
                "factor_values": values, "created_at": row["created_at"]}

    def page_factor_values(self, key: EvaluationKey, *, offset: int, limit: int) -> dict[str, Any]:
        """Return a page from stored rows or the lossless CSV value set."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit <= 0:
            raise ValueError("offset must be non-negative and limit must be positive integers")
        with self._connection() as db:
            evaluation = self._evaluation_for_key(db, key)
            if evaluation is None:
                raise KeyError(key.as_dict())
            evaluation_id = int(evaluation["evaluation_id"])
            total = db.execute("SELECT count(*) FROM factor_values WHERE evaluation_id=?",
                               (evaluation_id,)).fetchone()[0]
            expected_total = evaluation["factor_value_count"]
            if total != expected_total:
                raise ArchiveIntegrityError("factor value rows are missing")
            rows = db.execute("""SELECT ordinal,row_json FROM factor_values
                WHERE evaluation_id=? AND ordinal>=? ORDER BY ordinal LIMIT ?""",
                (evaluation_id, offset, limit)).fetchall()
            csv_raw = (self._load_blob(db, evaluation["factor_values_csv_blob_id"])
                       if total == 0 and evaluation["factor_values_csv_blob_id"] is not None else None)
        if csv_raw is not None:
            reader = csv.DictReader(io.StringIO(csv_raw.decode("utf-8-sig"), newline=""))
            if reader.fieldnames is None:
                raise ArchiveIntegrityError("factor-values CSV has no header")
            items = []
            count = 0
            for row in reader:
                if None in row:
                    raise ArchiveIntegrityError("factor-values CSV row has extra fields")
                if offset <= count < offset + limit:
                    items.append(dict(row))
                count += 1
            return {"total": count, "offset": offset, "items": items}
        items = []
        for expected_ordinal, row in enumerate(rows, start=offset):
            if row["ordinal"] != expected_ordinal:
                raise ArchiveIntegrityError("factor value page rows are missing or reordered")
            items.append(json.loads(row["row_json"]))
        return {"total": total, "offset": offset, "items": items}
