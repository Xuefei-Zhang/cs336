"""Offline evaluation helpers for generated samples and golden prompts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow

from aiinfra_e2e.config import EvalConfig
from aiinfra_e2e.eval.golden import GoldenValidationResult, validate_golden_case

Generator = Callable[[str], str]


@dataclass(frozen=True)
class OfflineEvalResult:
    run_id: str
    run_dir: Path
    report_path: Path
    metrics: dict[str, dict[str, float]]
    golden_results: list[GoldenValidationResult]


def _resolve_run_id(eval_config: EvalConfig) -> str:
    if eval_config.run_name:
        return eval_config.run_name
    return f"offline-{eval_config.sample_count}"


def _summary(values: Sequence[int | float]) -> dict[str, float]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(numeric_values),
        "mean": sum(numeric_values) / len(numeric_values),
        "max": max(numeric_values),
    }


def _log_mlflow_report(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_id: str,
    report_path: Path,
    sample_count: int,
    passed: int,
    failed: int,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    _ = mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_id):
        _ = mlflow.log_params({"sample_count": str(sample_count)})
        _ = mlflow.log_metrics({"golden_passed": passed, "golden_failed": failed})
        mlflow.log_artifact(str(report_path))


def run_offline_eval(
    *,
    eval_config: EvalConfig,
    obs_tracking_uri: str,
    obs_experiment_name: str,
    generator: Generator,
) -> OfflineEvalResult:
    """Generate offline samples, score them with simple heuristics, and log a report."""

    run_id = _resolve_run_id(eval_config)
    run_dir = Path(eval_config.output_dir) / run_id
    _ = run_dir.mkdir(parents=True, exist_ok=True)

    golden_results: list[GoldenValidationResult] = []
    char_lengths: list[int] = []
    repetition_ratios: list[float] = []

    for golden_case in eval_config.golden_prompts[: eval_config.sample_count]:
        prompt = golden_case.prompt
        output = generator(prompt)
        validation = validate_golden_case(
            name=golden_case.name,
            prompt=prompt,
            output=output,
            constraints=golden_case.constraints,
        )
        golden_results.append(validation)
        char_lengths.append(len(output))
        repetition_ratios.append(float(validation.metrics["repetition_ratio"]))

    passed = sum(1 for result in golden_results if result.passed)
    failed = len(golden_results) - passed
    metrics = {
        "char_length": _summary(char_lengths),
        "repetition_ratio": _summary(repetition_ratios),
    }
    report_payload = {
        "run_id": run_id,
        "sample_count": len(golden_results),
        "metrics": metrics,
        "golden_summary": {
            "total": len(golden_results),
            "passed": passed,
            "failed": failed,
        },
        "golden_results": [result.to_dict() for result in golden_results],
    }
    report_path = run_dir / "eval_report.json"
    _ = report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _log_mlflow_report(
        tracking_uri=obs_tracking_uri,
        experiment_name=obs_experiment_name,
        run_id=run_id,
        report_path=report_path,
        sample_count=len(golden_results),
        passed=passed,
        failed=failed,
    )

    return OfflineEvalResult(
        run_id=run_id,
        run_dir=run_dir,
        report_path=report_path,
        metrics=metrics,
        golden_results=golden_results,
    )
