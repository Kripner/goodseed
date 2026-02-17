"""Goodseed - ML experiment tracker.

Example:
    >>> import goodseed
    >>> run = goodseed.Run(experiment_name="my-experiment")
    >>> run.log_configs({"learning_rate": 0.001})
    >>> run.log_metrics({"loss": 0.5}, step=1)
    >>> run.close()
"""

from goodseed.run import Run

__version__ = "0.2.3"
__all__ = ["Run"]
