"""Local SQLite storage for experiment data.

Each run is stored as a single SQLite file. Data is written by the main
process during training and read by the HTTP server for visualization.

Thread-safe via RLock. WAL mode enables concurrent reads from the server
while the training process writes.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from goodseed.config import ensure_dir


def _get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS run_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS configs (
            path TEXT PRIMARY KEY,
            type_tag TEXT NOT NULL,
            value TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metric_series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS metric_points (
            series_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            y REAL NOT NULL,
            ts INTEGER NOT NULL,
            PRIMARY KEY (series_id, step)
        );

        CREATE TABLE IF NOT EXISTS string_series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS string_points (
            series_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            value TEXT NOT NULL,
            ts INTEGER NOT NULL,
            PRIMARY KEY (series_id, step)
        );
    """)
    conn.commit()


class LocalStorage:
    """Thread-safe SQLite storage for a single run."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._series_cache: Dict[str, int] = {}

        ensure_dir(self.db_path.parent)
        self._conn = _get_connection(self.db_path)
        _init_schema(self._conn)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _get_series_id(self, conn: sqlite3.Connection, path: str) -> int:
        """Get or create a series ID for a metric path. Must hold the lock."""
        sid = self._series_cache.get(path)
        if sid is not None:
            return sid
        conn.execute(
            "INSERT OR IGNORE INTO metric_series (path) VALUES (?)", (path,)
        )
        row = conn.execute(
            "SELECT id FROM metric_series WHERE path = ?", (path,)
        ).fetchone()
        sid = row["id"]
        self._series_cache[path] = sid
        return sid

    def set_meta(self, key: str, value: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            row = self._conn.execute(
                "SELECT value FROM run_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def log_configs(self, data: Dict[str, Tuple[str, str]]) -> None:
        """Log config values. data: {path: (type_tag, serialized_value)}"""
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as conn:
            for path, (type_tag, value) in data.items():
                conn.execute(
                    """INSERT OR REPLACE INTO configs
                       (path, type_tag, value, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (path, type_tag, value, now),
                )

    def get_configs(self) -> Dict[str, Tuple[str, str]]:
        """Get all configs as {path: (type_tag, value)}."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                "SELECT path, type_tag, value FROM configs"
            ).fetchall()
            return {row["path"]: (row["type_tag"], row["value"]) for row in rows}

    def log_metric_points(
        self, points: List[Tuple[str, int, float, int]]
    ) -> None:
        """Log metric points. Each tuple: (path, step, y, ts_unix)."""
        with self._transaction() as conn:
            for path, step, y, ts in points:
                series_id = self._get_series_id(conn, path)
                conn.execute(
                    """INSERT OR REPLACE INTO metric_points
                       (series_id, step, y, ts) VALUES (?, ?, ?, ?)""",
                    (series_id, step, y, ts),
                )

    def get_metric_points(
        self, path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get metric points, optionally filtered by path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            if path:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.y, p.ts
                       FROM metric_points p
                       JOIN metric_series s ON p.series_id = s.id
                       WHERE s.path = ?
                       ORDER BY p.step""",
                    (path,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.y, p.ts
                       FROM metric_points p
                       JOIN metric_series s ON p.series_id = s.id
                       ORDER BY s.path, p.step"""
                ).fetchall()
            return [dict(row) for row in rows]

    def get_metric_paths(self) -> List[str]:
        """Get all metric paths that have at least one point."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                """SELECT DISTINCT s.path FROM metric_series s
                   JOIN metric_points p ON s.id = p.series_id
                   ORDER BY s.path"""
            ).fetchall()
            return [row["path"] for row in rows]

    def _get_string_series_id(self, conn: sqlite3.Connection, path: str) -> int:
        """Get or create a series ID for a string series path. Must hold the lock."""
        conn.execute(
            "INSERT OR IGNORE INTO string_series (path) VALUES (?)", (path,)
        )
        row = conn.execute(
            "SELECT id FROM string_series WHERE path = ?", (path,)
        ).fetchone()
        return row["id"]

    def log_string_points(
        self, points: List[Tuple[str, int, str, int]]
    ) -> None:
        """Log string series points. Each tuple: (path, step, value, ts_unix)."""
        with self._transaction() as conn:
            for path, step, value, ts in points:
                series_id = self._get_string_series_id(conn, path)
                conn.execute(
                    """INSERT OR REPLACE INTO string_points
                       (series_id, step, value, ts) VALUES (?, ?, ?, ?)""",
                    (series_id, step, value, ts),
                )

    def get_string_points(
        self, path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get string series points, optionally filtered by path."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            if path:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.value, p.ts
                       FROM string_points p
                       JOIN string_series s ON p.series_id = s.id
                       WHERE s.path = ?
                       ORDER BY p.step""",
                    (path,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT s.path, p.step, p.value, p.ts
                       FROM string_points p
                       JOIN string_series s ON p.series_id = s.id
                       ORDER BY s.path, p.step"""
                ).fetchall()
            return [dict(row) for row in rows]

    def get_string_series_paths(self) -> List[str]:
        """Get all string series paths that have at least one point."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            rows = self._conn.execute(
                """SELECT DISTINCT s.path FROM string_series s
                   JOIN string_points p ON s.id = p.series_id
                   ORDER BY s.path"""
            ).fetchall()
            return [row["path"] for row in rows]

    def checkpoint_wal(self) -> None:
        """Checkpoint WAL to consolidate into a single .sqlite file."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Storage is closed")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def delete_db_file(self) -> None:
        """Close connection and delete the database file."""
        self.close()
        if self.db_path.exists():
            os.unlink(self.db_path)
        wal = Path(str(self.db_path) + "-wal")
        shm = Path(str(self.db_path) + "-shm")
        if wal.exists():
            os.unlink(wal)
        if shm.exists():
            os.unlink(shm)

    def __enter__(self) -> "LocalStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
