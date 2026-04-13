import json
from pathlib import Path

import mlflow

from aiinfra_e2e.config import EvalConfig, GoldenPromptConfig, load_yaml
from aiinfra_e2e.eval.golden import validate_golden_case
from aiinfra_e2e.eval.offline import run_offline_eval


def test_validate_golden_case_reports_failed_constraints() -> None:
    result = validate_golden_case(
        prompt="Explain caching in one sentence.",
        output="cache cache cache",
        constraints={
            "min_length": 20,
            "must_contain": ["cache"],
            "must_not_contain": ["error"],
            "max_repetition_ratio": 0.34,
        },
    )

    assert result.passed is False
    assert sorted(result.failed_constraints) == ["max_repetition_ratio", "min_length"]
    assert result.metrics["char_length"] == len("cache cache cache")


def test_run_offline_eval_writes_report_and_logs_mlflow_artifact(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(str(tracking_dir))
    _ = mlflow.set_experiment("offline-eval-test")

    run_dir = tmp_path / "artifacts" / "runs" / "offline-eval-smoke"

    prompts_seen: list[str] = []

    def generate(prompt: str) -> str:
        prompts_seen.append(prompt)
        if "repeat" in prompt:
            return "repeat repeat repeat"
        return f"answer::{prompt}"

    eval_result = run_offline_eval(
        eval_config=EvalConfig(
            sample_count=2,
            run_name="offline-eval-smoke",
            output_dir=str(tmp_path / "artifacts" / "runs"),
            golden_prompts=[
                GoldenPromptConfig(
                    name="length-check",
                    prompt="Say hello politely",
                    constraints={"min_length": 8, "must_contain": ["answer::"]},
                ),
                GoldenPromptConfig(
                    name="repeat-check",
                    prompt="repeat once without repeat loops",
                    constraints={"max_repetition_ratio": 0.34},
                ),
            ],
        ),
        obs_tracking_uri=str(tracking_dir),
        obs_experiment_name="offline-eval-test",
        generator=generate,
    )

    report_path = run_dir / "eval_report.json"
    assert eval_result.report_path == report_path
    assert prompts_seen == ["Say hello politely", "repeat once without repeat loops"]
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "offline-eval-smoke"
    assert payload["sample_count"] == 2
    assert payload["metrics"]["char_length"]["min"] > 0
    assert payload["metrics"]["char_length"]["max"] >= payload["metrics"]["char_length"]["mean"]
    assert payload["golden_summary"]["total"] == 2
    assert payload["golden_summary"]["passed"] == 1
    assert payload["golden_summary"]["failed"] == 1
    assert payload["golden_results"][1]["failed_constraints"] == ["max_repetition_ratio"]

    experiment = mlflow.get_experiment_by_name("offline-eval-test")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list")
    assert len(runs) == 1

    run_id = runs[0].to_dictionary()["info"]["run_id"]
    artifact_dir = tracking_dir / experiment.experiment_id / run_id / "artifacts" / "eval_report.json"
    assert artifact_dir.exists()


def test_load_yaml_supports_offline_eval_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "offline.yaml"
    _ = config_path.write_text(
        (
            "metric: offline\n"
            "sample_count: 3\n"
            "output_dir: artifacts/runs\n"
            "run_name: sample-offline\n"
            "golden_prompts:\n"
            "  - name: polite\n"
            "    prompt: Say hello\n"
            "    constraints:\n"
            "      min_length: 4\n"
            "      must_contain:\n"
            "        - hello\n"
        ),
        encoding="utf-8",
    )

    config = load_yaml(config_path, EvalConfig)

    assert config.sample_count == 3
    assert config.output_dir == "artifacts/runs"
    assert config.run_name == "sample-offline"
    assert config.golden_prompts[0].constraints["must_contain"] == ["hello"]
