"""Local HTTP server for serving experiment data to the frontend.

Reads SQLite run files from the projects directory and exposes a JSON API.
The frontend (served from goodseed.ai or localhost) connects to this server.

Usage:
    goodseed serve [dir] [--port PORT]
"""

import json
import logging
import math
import re
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger("goodseed.server")

from goodseed.utils import deserialize_value


def _sanitize_for_json(obj: object) -> object:
    """Convert NaN/Infinity floats to strings for JSON spec compliance.

    JSON does not support NaN/Infinity literals, so we encode them as
    the strings ``"NaN"``, ``"Infinity"``, and ``"-Infinity"`` to
    preserve the information for the frontend.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


def _read_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    """Read all run_meta key-value pairs."""
    rows = conn.execute("SELECT key, value FROM run_meta").fetchall()
    return {row["key"]: row["value"] for row in rows}


def _scan_runs(projects_dir: Path) -> List[Dict[str, Any]]:
    """Scan the projects directory for run SQLite files.

    Supports nested project names (e.g. ``workspace/project``) by recursively
    searching for ``runs/*.sqlite`` at any depth under *projects_dir*.
    """
    runs = []
    if not projects_dir.exists():
        return runs

    for db_path in sorted(projects_dir.glob("**/runs/*.sqlite")):
        project_name = str(db_path.parent.parent.relative_to(projects_dir))
        try:
            conn = _open_readonly(db_path)
            try:
                meta = _read_meta(conn)
                try:
                    ss_rows = conn.execute(
                        """SELECT DISTINCT s.path FROM string_series s
                           JOIN string_points p ON s.id = p.series_id
                           ORDER BY s.path"""
                    ).fetchall()
                    string_series_paths = [row["path"] for row in ss_rows]
                except sqlite3.OperationalError as e:
                    logger.debug("string_series table not found in %s: %s", db_path, e)
                    string_series_paths = []
            finally:
                conn.close()

            runs.append({
                "project": project_name,
                "run_id": meta.get("run_name", db_path.stem),
                "experiment_name": meta.get("experiment_name"),
                "created_at": meta.get("created_at"),
                "closed_at": meta.get("closed_at"),
                "status": meta.get("status", "unknown"),
                "string_series_paths": string_series_paths,
            })
        except Exception:
            logger.warning("Failed to read run database: %s", db_path, exc_info=True)
            continue

    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs


def _get_configs(db_path: Path) -> Dict[str, Any]:
    """Read configs from a run database and deserialize values."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute("SELECT path, type_tag, value FROM configs").fetchall()
    finally:
        conn.close()

    configs = {}
    for row in rows:
        configs[row["path"]] = deserialize_value(row["type_tag"], row["value"])
    return configs


def _ts_to_iso(ts: int) -> str:
    """Convert a unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _get_metrics(db_path: Path, metric_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read metric points from a run database."""
    conn = _open_readonly(db_path)
    try:
        if metric_path:
            rows = conn.execute(
                """SELECT s.path, p.step, p.y, p.ts
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   WHERE s.path = ?
                   ORDER BY p.step""",
                (metric_path,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT s.path, p.step, p.y, p.ts
                   FROM metric_points p
                   JOIN metric_series s ON p.series_id = s.id
                   ORDER BY s.path, p.step"""
            ).fetchall()
    finally:
        conn.close()

    return [
        {
            "path": row["path"],
            "step": row["step"],
            "value": row["y"],
            "is_preview": False,
            "preview_completion": None,
            "logged_at": _ts_to_iso(row["ts"]),
        }
        for row in rows
    ]


def _get_metric_paths(db_path: Path) -> List[str]:
    """Read distinct metric paths from a run database."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute(
            """SELECT DISTINCT s.path FROM metric_series s
               JOIN metric_points p ON s.id = p.series_id
               ORDER BY s.path"""
        ).fetchall()
    finally:
        conn.close()
    return [row["path"] for row in rows]


def _get_string_series(
    db_path: Path,
    series_path: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    tail: Optional[int] = None,
) -> Dict[str, Any]:
    """Read string series points from a run database.

    Returns {"points": [...], "total": <int>} where total is the full count
    before limit/offset are applied.

    If ``tail`` is given, returns the last ``tail`` rows (overrides limit/offset).
    """
    conn = _open_readonly(db_path)
    try:
        where = "WHERE s.path = ?" if series_path else ""
        base_params: list = [series_path] if series_path else []

        total_row = conn.execute(
            f"""SELECT COUNT(*) as cnt
                FROM string_points p
                JOIN string_series s ON p.series_id = s.id
                {where}""",
            base_params,
        ).fetchone()
        total = total_row["cnt"] if total_row else 0

        order_col = "p.step" if series_path else "s.path, p.step"

        if tail is not None:
            # Fetch last N rows: use a subquery with DESC, then re-order ASC
            query = f"""SELECT * FROM (
                SELECT s.path, p.step, p.value, p.ts
                FROM string_points p
                JOIN string_series s ON p.series_id = s.id
                {where}
                ORDER BY {order_col} DESC
                LIMIT ?
            ) sub ORDER BY {"step" if series_path else "path, step"}"""
            rows = conn.execute(query, base_params + [tail]).fetchall()
        else:
            query = f"""SELECT s.path, p.step, p.value, p.ts
                        FROM string_points p
                        JOIN string_series s ON p.series_id = s.id
                        {where}
                        ORDER BY {order_col}"""
            params = list(base_params)
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as e:
        logger.debug("string_series table not available in %s: %s", db_path, e)
        return {"points": [], "total": 0}
    finally:
        conn.close()

    points = [
        {
            "path": row["path"],
            "step": row["step"],
            "value": row["value"],
            "logged_at": _ts_to_iso(row["ts"]),
        }
        for row in rows
    ]
    return {"points": points, "total": total}


def _scan_projects(projects_dir: Path) -> List[Dict[str, Any]]:
    """Scan the projects directory and return project metadata.

    Lightweight: uses file mtime instead of opening SQLite databases.
    """
    projects: Dict[str, Dict[str, Any]] = {}
    if not projects_dir.exists():
        return []

    for db_path in sorted(projects_dir.glob("**/runs/*.sqlite")):
        name = str(db_path.parent.parent.relative_to(projects_dir))
        if name not in projects:
            projects[name] = {"name": name, "run_count": 0, "last_modified": None}
        projects[name]["run_count"] += 1
        mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc).isoformat()
        if projects[name]["last_modified"] is None or mtime > projects[name]["last_modified"]:
            projects[name]["last_modified"] = mtime

    result = list(projects.values())
    result.sort(key=lambda p: p["last_modified"] or "", reverse=True)
    return result


def _resolve_run_db(projects_dir: Path, project: str, run_name: str) -> Optional[Path]:
    """Resolve a run database file path, return None if not found."""
    db_path = projects_dir / project / "runs" / f"{run_name}.sqlite"
    if db_path.exists():
        return db_path
    return None


# Route patterns
_ROUTE_PROJECTS = re.compile(r"^/api/projects$")
_ROUTE_RUNS = re.compile(r"^/api/runs$")
_ROUTE_CONFIGS = re.compile(r"^/api/runs/(.+)/([^/]+)/configs$")
_ROUTE_METRICS = re.compile(r"^/api/runs/(.+)/([^/]+)/metrics$")
_ROUTE_METRIC_PATHS = re.compile(r"^/api/runs/(.+)/([^/]+)/metric-paths$")
_ROUTE_STRING_SERIES = re.compile(r"^/api/runs/(.+)/([^/]+)/string_series$")


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Goodseed API."""

    # Set by the server
    projects_dir: Path

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to the appropriate handler."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        try:
            # GET /api/projects
            m = _ROUTE_PROJECTS.match(path)
            if m:
                projects = _scan_projects(self.projects_dir)
                self._send_json({"projects": projects})
                return

            # GET /api/runs
            m = _ROUTE_RUNS.match(path)
            if m:
                runs = _scan_runs(self.projects_dir)
                project_filter = query.get("project", [None])[0]
                if project_filter:
                    runs = [r for r in runs if r["project"] == project_filter]
                self._send_json({"runs": runs})
                return

            # GET /api/runs/<project>/<run_name>/configs
            m = _ROUTE_CONFIGS.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                configs = _get_configs(db_path)
                self._send_json({"configs": configs})
                return

            # GET /api/runs/<project>/<run_name>/metrics
            m = _ROUTE_METRICS.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                metric_path = query.get("path", [None])[0]
                metrics = _get_metrics(db_path, metric_path)
                self._send_json({"metrics": metrics})
                return

            # GET /api/runs/<project>/<run_name>/metric-paths
            m = _ROUTE_METRIC_PATHS.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                paths = _get_metric_paths(db_path)
                self._send_json({"paths": paths})
                return

            # GET /api/runs/<project>/<run_name>/string_series
            m = _ROUTE_STRING_SERIES.match(path)
            if m:
                project, run_name = m.group(1), m.group(2)
                db_path = _resolve_run_db(self.projects_dir, project, run_name)
                if not db_path:
                    self._send_error(404, f"Run not found: {project}/{run_name}")
                    return
                series_path = query.get("path", [None])[0]
                limit_str = query.get("limit", [None])[0]
                offset_str = query.get("offset", ["0"])[0]
                tail_str = query.get("tail", [None])[0]
                limit = int(limit_str) if limit_str else None
                offset = int(offset_str) if offset_str else 0
                tail = int(tail_str) if tail_str else None
                result = _get_string_series(db_path, series_path, limit=limit, offset=offset, tail=tail)
                self._send_json({
                    "string_series": result["points"],
                    "total": result["total"],
                })
                return

            self._send_error(404, "Not found")

        except Exception as e:
            self._send_error(500, str(e))

    def _send_json(self, data: Any) -> None:
        """Send a JSON response with CORS headers."""
        body = json.dumps(_sanitize_for_json(data)).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        """Send a JSON error response."""
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        """Add CORS headers to the response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging."""
        pass


def run_server(projects_dir: Path, port: int = 8765, verbose: bool = False) -> None:
    """Start the local HTTP server.

    Args:
        projects_dir: Directory containing project subdirectories with .sqlite files.
        port: Port to listen on (default: 8765).
        verbose: Print extra startup information.
    """
    _RequestHandler.projects_dir = projects_dir

    server = ThreadingHTTPServer(("127.0.0.1", port), _RequestHandler)

    if verbose:
        print(f"Goodseed server running at http://localhost:{port}")
        print(f"Data directory: {projects_dir}")
    print(f"View your runs at https://goodseed.ai/app/local?port={port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
