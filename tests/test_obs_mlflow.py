from importlib.util import find_spec
from types import SimpleNamespace

from aiinfra_e2e.obs import mlflow as obs_mlflow


def test_obs_mlflow_module_exists() -> None:
    spec = find_spec("aiinfra_e2e.obs.mlflow")

    assert spec is not None


def test_configure_tracking_sets_uri_and_experiment(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _set_tracking_uri(uri: str) -> None:
        calls.append(("tracking_uri", uri))

    def _set_experiment(name: str) -> str:
        calls.append(("experiment", name))
        return name

    monkeypatch.setattr(
        obs_mlflow,
        "mlflow",
        SimpleNamespace(set_tracking_uri=_set_tracking_uri, set_experiment=_set_experiment),
        raising=False,
    )

    obs_mlflow.configure_tracking(tracking_uri="file:///tmp/mlruns", experiment_name="demo-exp")

    assert calls == [("tracking_uri", "file:///tmp/mlruns"), ("experiment", "demo-exp")]


def test_start_run_configures_tracking_before_entering_context(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _set_tracking_uri(uri: str) -> None:
        calls.append(("tracking_uri", uri))

    def _set_experiment(name: str) -> str:
        calls.append(("experiment", name))
        return name

    class _Run:
        def __enter__(self) -> str:
            calls.append(("enter", "run"))
            return "active-run"

        def __exit__(self, exc_type, exc, tb) -> None:
            calls.append(("exit", "run"))
            return None

    def _start_run(*, run_name: str):
        calls.append(("run_name", run_name))
        return _Run()

    monkeypatch.setattr(
        obs_mlflow,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=_set_tracking_uri,
            set_experiment=_set_experiment,
            start_run=_start_run,
        ),
        raising=False,
    )

    with obs_mlflow.start_run(
        tracking_uri="file:///tmp/mlruns",
        experiment_name="demo-exp",
        run_name="demo-run",
    ) as active_run:
        assert active_run == "active-run"
        calls.append(("inside", "context"))

    assert calls == [
        ("tracking_uri", "file:///tmp/mlruns"),
        ("experiment", "demo-exp"),
        ("run_name", "demo-run"),
        ("enter", "run"),
        ("inside", "context"),
        ("exit", "run"),
    ]
