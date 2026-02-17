"""Run class for experiment tracking.

The Run object is the main interface for logging experiment data.
All data is written to a local SQLite database that persists permanently.
A local HTTP server (goodseed serve) reads these files for visualization.
"""

import atexit
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from goodseed.config import (
    get_default_project,
    get_run_db_path,
)
from goodseed.storage import LocalStorage
from goodseed.utils import (
    flatten_dict,
    generate_run_name,
    normalize_path,
    serialize_value,
)


def _resolve_db_path(
    run_name: str,
    project: str,
    auto_name: bool,
    log_dir: Optional[Union[str, Path]] = None,
    goodseed_home: Optional[Union[str, Path]] = None,
) -> tuple:
    """Resolve a unique (run_name, db_path) pair.

    For auto-generated names, appends -2, -3, etc. on collision.
    For explicit names, raises if the file already exists.

    Returns (run_name, db_path).
    """
    def _path_for(name: str) -> Path:
        if log_dir:
            return Path(log_dir) / f"{name}.sqlite"
        return get_run_db_path(project, name, goodseed_home)

    db_path = _path_for(run_name)
    if not db_path.exists():
        return run_name, db_path

    if not auto_name:
        raise RuntimeError(
            f"Database already exists: {db_path}\n"
            f"Choose a different run_name or delete the file:\n"
            f"  rm {db_path}"
        )

    base = run_name
    for i in range(2, 1000):
        candidate = f"{base}-{i}"
        db_path = _path_for(candidate)
        if not db_path.exists():
            return candidate, db_path

    raise RuntimeError("Could not find a unique run name after retries")


class Run:
    """An experiment run for logging metrics and configs.

    Data is written to a local SQLite file that persists after the run closes.
    Use ``goodseed serve`` to visualize runs in the browser.

    Args:
        experiment_name: Human-readable name for this experiment.
        project: Project name (defaults to GOODSEED_PROJECT or 'default').
        run_name: Unique run name. Auto-generated if not provided.
        goodseed_home: Override for GOODSEED_HOME.
        log_dir: Override directory for the run database file.
            If provided, the database is stored at ``log_dir/{run_name}.sqlite``.
            Otherwise uses ``~/.goodseed/projects/{project}/{run_name}.sqlite``.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        goodseed_home: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        created_at: Optional[str] = None,
    ):
        self.project = project or get_default_project()
        self.experiment_name = experiment_name

        self._lock = threading.RLock()
        self._closed = False

        self.run_name, db_path = _resolve_db_path(
            run_name=run_name or generate_run_name(),
            project=self.project,
            auto_name=run_name is None,
            log_dir=log_dir,
            goodseed_home=goodseed_home,
        )

        # Initialize local storage
        self._storage = LocalStorage(db_path)
        self._db_path = db_path
        self._storage.set_meta("run_name", self.run_name)
        self._storage.set_meta("project", self.project)
        self._storage.set_meta(
            "created_at", created_at or datetime.now(timezone.utc).isoformat()
        )
        self._storage.set_meta("status", "running")
        if experiment_name:
            self._storage.set_meta("experiment_name", experiment_name)

        atexit.register(self._cleanup)

        print(f"Goodseed run: {self.run_name}")
        print(f"  Data: {db_path}")

    def log_configs(
        self,
        data: Dict[str, Any],
        flatten: bool = False,
    ) -> None:
        """Log configuration values.

        Args:
            data: Dictionary of path -> value mappings.
            flatten: If True, flatten nested dictionaries.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Run is closed")

            if flatten:
                data = flatten_dict(data)

            # Normalize paths and serialize
            serialized = {}
            for k, v in data.items():
                path = normalize_path(k)
                type_tag, value = serialize_value(v)
                serialized[path] = (type_tag, value)

            self._storage.log_configs(serialized)

    def log_metrics(
        self,
        data: Dict[str, float],
        step: int,
    ) -> None:
        """Log metric values at a given step.

        Args:
            data: Dictionary of metric_path -> float_value.
            step: The step number (integer).
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Run is closed")

            step = int(step)
            ts = int(datetime.now(timezone.utc).timestamp())

            points = []
            for k, v in data.items():
                path = normalize_path(k)
                points.append((path, step, float(v), ts))

            self._storage.log_metric_points(points)

    def log_string_series(
        self,
        data: Dict[str, str],
        step: int,
    ) -> None:
        """Log string series values at a given step.

        Args:
            data: Dictionary of series_path -> string_value.
            step: The step number (integer).
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Run is closed")

            step = int(step)
            ts = int(datetime.now(timezone.utc).timestamp())

            points = []
            for k, v in data.items():
                path = normalize_path(k)
                points.append((path, step, str(v), ts))

            self._storage.log_string_points(points)

    def close(self, status: str = "finished") -> None:
        """Close the run.

        Args:
            status: Run status to set ('finished' or 'failed').

        The WAL is checkpointed so the run is a single .sqlite file.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._storage.set_meta("status", status)
        self._storage.set_meta("closed_at", datetime.now(timezone.utc).isoformat())
        self._storage.checkpoint_wal()
        self._storage.close()
        print(f"Goodseed run closed: {self.run_name}")

    def _cleanup(self) -> None:
        if not self._closed:
            self.close()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "failed" if exc_type is not None else "finished"
        self.close(status=status)
