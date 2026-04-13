"""Shared MLflow setup helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import mlflow


def configure_tracking(*, tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking destination and active experiment."""

    mlflow.set_tracking_uri(tracking_uri)
    _ = mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(*, tracking_uri: str, experiment_name: str, run_name: str) -> Iterator[object]:
    """Configure tracking and yield an active MLflow run context."""

    configure_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name) as active_run:
        yield active_run
