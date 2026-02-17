# Goodseed - Architecture & API Reference

## Architecture

### Storage

Each run is a single SQLite file at `~/.goodseed/projects/<project>/runs/<run_name>.sqlite`.

SQLite WAL mode allows the training process to write while the server reads concurrently. On `close()`, the WAL is checkpointed so the result is one file.

Schema:

```sql
run_meta(key TEXT PRIMARY KEY, value TEXT)
configs(path TEXT PRIMARY KEY, type_tag TEXT, value TEXT, updated_at TEXT)
metric_series(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE)
metric_points(series_id INTEGER, step INTEGER, y REAL, ts INTEGER, PRIMARY KEY (series_id, step))
string_series(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL UNIQUE)
string_points(series_id INTEGER, step INTEGER, value TEXT, ts INTEGER, PRIMARY KEY (series_id, step))
```

### Server

`goodseed serve` starts a local HTTP server (`http.server.ThreadingHTTPServer`, no dependencies) that scans the projects directory and serves run data as JSON.

Endpoints:

```
GET /api/projects                                -- list all projects
GET /api/runs                                    -- list all runs
GET /api/runs?project=<name>                     -- list runs filtered by project
GET /api/runs/<project>/<run_name>/configs       -- config key-value pairs
GET /api/runs/<project>/<run_name>/metrics       -- all metric points
GET /api/runs/<project>/<run_name>/metrics?path=loss  -- filtered by path
GET /api/runs/<project>/<run_name>/metric-paths  -- list of metric names
GET /api/runs/<project>/<run_name>/string_series -- all string series points
GET /api/runs/<project>/<run_name>/string_series?path=<path>           -- filtered by path
GET /api/runs/<project>/<run_name>/string_series?limit=50&offset=0     -- paginated
GET /api/runs/<project>/<run_name>/string_series?tail=20               -- last N entries
```

CORS is enabled (`Access-Control-Allow-Origin: *`) so the frontend at goodseed.ai can connect.

### Run Status

Runs have a `status` field in `run_meta`: `running` (default), `finished`, or `failed`. Status is set when `close()` is called. The context manager sets `failed` on exception, `finished` otherwise.

### Data Types

Configs are stored with type tags:

| Python Type | type_tag | Stored as |
|-------------|----------|-----------|
| `bool` | `"bool"` | `"true"` / `"false"` |
| `int` | `"int"` | `"42"` |
| `float` | `"float"` | `"3.14"` |
| `str` | `"str"` | `"hello"` |
| `datetime` | `"datetime"` | ISO 8601 string |
| `None` | `"null"` | `""` |

Metrics are always `float`.

## API Reference

### `goodseed.Run`

```python
Run(
    experiment_name: str | None = None,
    project: str | None = None,        # default: GOODSEED_PROJECT or "default"
    run_name: str | None = None,        # auto-generated if not provided
    goodseed_home: str | Path | None = None,  # override for ~/.goodseed
    log_dir: str | Path | None = None,  # override directory for the .sqlite file
)
```

#### `run.log_configs(data, flatten=False)`

Log configuration key-value pairs.

```python
run.log_configs({"learning_rate": 0.001, "optimizer": "adam"})

# Flatten nested dicts
run.log_configs({"model": {"hidden": 256, "layers": 4}}, flatten=True)
# Stores: "model/hidden" = 256, "model/layers" = 4
```

#### `run.log_metrics(data, step)`

Log metric values at a given step.

```python
run.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)
```

- `step` is coerced to `int`
- Same step + path overwrites the previous value
- Values are coerced to `float`

#### `run.log_string_series(data, step)`

Log string series values at a given step.

```python
run.log_string_series({"output": "Generated text here..."}, step=100)
```

- `step` is coerced to `int`
- Same step + path overwrites the previous value
- Values are coerced to `str`

#### `run.close(status='finished')`

Close the run. Checkpoints the WAL and closes the database connection.

#### Context Manager

```python
with goodseed.Run(experiment_name="exp") as run:
    run.log_metrics({"loss": 0.5}, step=1)
# status='finished' on normal exit, 'failed' on exception
```

## CLI Reference

### `goodseed serve [dir] [--port PORT]`

Start the local HTTP server. `dir` defaults to `~/.goodseed/projects`, port defaults to 8765.

### `goodseed list [dir] [--project NAME]`

List projects, or runs within a specific project when `--project` / `-p` is provided.

## File Layout

```
goodseed/
  src/goodseed/
    __init__.py       # exports Run
    run.py            # Run class
    storage.py        # LocalStorage (SQLite read/write)
    server.py         # HTTP server and read-only query functions
    config.py         # Environment config and path helpers
    utils.py          # Name generation, serialization, flattening
    cli.py            # CLI entry point
    _sync_legacy.py   # Legacy Supabase sync (not used in current workflow)
  tests/
    conftest.py
    test_storage.py
    test_run.py
    test_utils.py
    test_integration.py     # full workflow + HTTP server tests
    test_cli.py
    test_neptune_proxy.py
  examples/
    mlp.py                  # PyTorch MLP grid search on Iris (requires torch, sklearn)
```
