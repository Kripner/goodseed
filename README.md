# Goodseed

ML experiment tracker. Logs metrics and configs to local SQLite files, serves them via a built-in HTTP server, and visualizes them in the browser.

## Install

```bash
cd goodseed
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import goodseed

with goodseed.Run(experiment_name="my-experiment") as run:
    run.log_configs({"learning_rate": 0.001, "batch_size": 32})

    for step in range(100):
        loss = train_step()
        run.log_metrics({"loss": loss}, step=step)
```

Then view your runs:

```bash
goodseed serve
```

## How It Works

Each `Run()` creates a SQLite file at `~/.goodseed/projects/<project>/runs/<run_name>.sqlite`. Metrics and configs are written there during training. After the run closes, the WAL is checkpointed so the result is a single `.sqlite` file.

The `goodseed serve` command starts a local HTTP server that reads these files and exposes a JSON API. The frontend at [goodseed.ai](https://goodseed.ai/app/local) connects to this server to display your runs.

## Configuration

| Variable | Description |
|----------|-------------|
| `GOODSEED_HOME` | Data directory (default: `~/.goodseed`) |
| `GOODSEED_PROJECT` | Default project name (default: `default`) |

## CLI

```bash
goodseed                   # Start the server (default command)
goodseed serve [dir]       # Start the server, optionally from a specific directory
goodseed serve --port 9000 # Use a custom port
goodseed list              # List local runs
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

See [DOCS.md](DOCS.md) for architecture details and API reference.
