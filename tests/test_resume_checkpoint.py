import json
import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
import types

from datasets import Dataset
import torch

from aiinfra_e2e.config import DataConfig, ObsConfig, TrainConfig
from aiinfra_e2e.train.checkpointing import latest_checkpoint, should_resume
from aiinfra_e2e.train import sft as sft_module
from aiinfra_e2e.train.sft import run_sft


def _build_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"instruction": "Say hi", "input": "", "output": "hi"},
            {"instruction": "Say bye", "input": "", "output": "bye"},
            {"instruction": "Count", "input": "1 2", "output": "3"},
        ]
    )


def _base_configs(tmp_path: Path) -> tuple[DataConfig, TrainConfig, ObsConfig]:
    run_root = tmp_path / "artifacts" / "runs"
    tracking_dir = tmp_path / "mlruns"
    cache_dir = tmp_path / "hf-cache"
    return (
        DataConfig(
            dataset_id="unit-test-dataset",
            split="train",
            cache_dir=str(cache_dir),
            train_ratio=0.8,
            val_ratio=0.1,
            split_key_fields=["instruction", "output"],
        ),
        TrainConfig(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            output_dir=str(run_root),
            run_name="resume-test",
            seed=13,
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            logging_steps=1,
            save_steps=1,
            gradient_checkpointing=False,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
        ),
        ObsConfig(
            tracking_uri=str(tracking_dir),
            experiment_name="resume-tests",
        ),
    )


def test_should_resume_accepts_boolean_and_path_inputs() -> None:
    assert should_resume(TrainConfig(model_id="model", resume_from_checkpoint=True)) is True
    assert (
        should_resume(TrainConfig(model_id="model", resume_from_checkpoint="checkpoint-9")) is True
    )
    assert should_resume(TrainConfig(model_id="model", resume_from_checkpoint=False)) is False


def test_latest_checkpoint_returns_highest_checkpoint_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_one = run_dir / "checkpoint-1"
    checkpoint_ten = run_dir / "checkpoint-10"
    _ = checkpoint_one.mkdir(parents=True)
    _ = (run_dir / "notes").mkdir()
    _ = checkpoint_ten.mkdir()

    assert latest_checkpoint(run_dir) == checkpoint_ten


def test_load_trl_sft_objects_supports_public_imports(monkeypatch) -> None:
    spec = importlib.util.find_spec("aiinfra_e2e.train.trl_compat")
    assert spec is not None

    compat_module = importlib.import_module("aiinfra_e2e.train.trl_compat")
    fake_trl = types.ModuleType("trl")
    fake_sft_trainer = type("FakePublicSFTTrainer", (), {})
    fake_sft_config = type("FakePublicSFTConfig", (), {})
    setattr(fake_trl, "SFTTrainer", fake_sft_trainer)
    setattr(fake_trl, "SFTConfig", fake_sft_config)

    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    trainer_cls, config_cls = compat_module.load_trl_sft_objects(force_reload=True)

    assert trainer_cls is fake_sft_trainer
    assert config_cls is fake_sft_config


def test_load_trl_sft_objects_rejects_private_trl_fallbacks(monkeypatch) -> None:
    compat_module = importlib.import_module("aiinfra_e2e.train.trl_compat")
    fake_trl = types.ModuleType("trl")
    original_import = builtins.__import__

    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("trl.trainer"):
            raise AssertionError(f"private TRL import attempted: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    try:
        compat_module.load_trl_sft_objects(force_reload=True)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError when TRL public exports are unavailable")

    assert "trl.SFTTrainer" in message
    assert "trl.SFTConfig" in message


def test_run_sft_resumes_from_latest_checkpoint_without_restarting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("AIINFRA_E2E_CPU_SMOKE", "1")

    data_config, train_config, obs_config = _base_configs(tmp_path)
    dataset = _build_dataset()
    monkeypatch.setattr("aiinfra_e2e.train.sft.load_hf_dataset", lambda config: dataset)
    monkeypatch.setattr("aiinfra_e2e.train.sft._load_tokenizer", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("aiinfra_e2e.train.sft._load_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(
        "aiinfra_e2e.train.sft._prepare_train_split",
        lambda *args, **kwargs: _build_dataset(),
    )
    monkeypatch.setattr("aiinfra_e2e.train.sft.SFTTrainer", _ResumeTrainer)
    monkeypatch.setattr("aiinfra_e2e.train.sft.SFTConfig", _DummyTrainingArgs)
    monkeypatch.setattr("aiinfra_e2e.train.sft.build_lora_config", lambda config: None)
    monkeypatch.setattr("aiinfra_e2e.train.sft._set_seed", lambda seed: None)

    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text("placeholder: true\n", encoding="utf-8")

    run_dir = run_sft(
        data_config=data_config,
        train_config=train_config,
        obs_config=obs_config,
        data_config_path=config_path,
        train_config_path=config_path,
        obs_config_path=config_path,
    )

    checkpoint_one = run_dir / "checkpoint-1"
    assert checkpoint_one.exists()

    trainer_cls = sft_module.SFTTrainer
    assert trainer_cls is not None
    original_training_step = trainer_cls.training_step
    training_step_calls = 0

    def counting_training_step(self, *args, **kwargs):
        nonlocal training_step_calls
        training_step_calls += 1
        return original_training_step(self, *args, **kwargs)

    monkeypatch.setattr(
        trainer_cls,
        "training_step",
        counting_training_step,
    )

    resumed_train_config = train_config.model_copy(
        update={
            "max_steps": 2,
            "resume_from_checkpoint": True,
        }
    )
    resumed_run_dir = run_sft(
        data_config=data_config,
        train_config=resumed_train_config,
        obs_config=obs_config,
        data_config_path=config_path,
        train_config_path=config_path,
        obs_config_path=config_path,
    )

    assert resumed_run_dir == run_dir
    assert training_step_calls == 1

    checkpoint_two = run_dir / "checkpoint-2"
    trainer_state = json.loads((checkpoint_two / "trainer_state.json").read_text(encoding="utf-8"))
    assert trainer_state["global_step"] == 2


class _FakeTrainResult:
    training_loss = 0.25


class _FakeTrainer:
    attempts = 0
    batch_sizes: list[int] = []
    max_lengths: list[int | None] = []
    resume_args: list[str | bool | None] = []
    saved_model_dirs: list[str] = []

    def __init__(self, *, args, **kwargs) -> None:
        type(self).batch_sizes.append(args.per_device_train_batch_size)
        type(self).max_lengths.append(args.max_length)

    def train(self, resume_from_checkpoint=None):
        type(self).resume_args.append(resume_from_checkpoint)
        type(self).attempts += 1
        if type(self).attempts < 3:
            raise torch.cuda.OutOfMemoryError("simulated cuda oom")
        return _FakeTrainResult()

    def save_model(self, output_dir: str) -> None:
        type(self).saved_model_dirs.append(output_dir)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"


class _FakeModelConfig:
    use_cache = True


class _FakeModel:
    config = _FakeModelConfig()


class _DummyTrainingArgs:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ResumeTrainResult:
    training_loss = 0.125


class _ResumeTrainer:
    def __init__(self, *, args, **kwargs) -> None:
        self.args = args

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)

    def train(self, resume_from_checkpoint=None):
        _ = self.training_step()
        if resume_from_checkpoint is None:
            next_step = 1
        else:
            next_step = int(str(resume_from_checkpoint).rsplit("-", maxsplit=1)[1]) + 1
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{next_step}"
        _ = checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _ = (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps({"global_step": next_step}),
            encoding="utf-8",
        )
        return _ResumeTrainResult()

    def save_model(self, output_dir: str) -> None:
        _ = Path(output_dir).mkdir(parents=True, exist_ok=True)


def test_run_sft_retries_cuda_oom_with_bounded_fallback(tmp_path: Path, monkeypatch) -> None:
    data_config, train_config, obs_config = _base_configs(tmp_path)
    oom_train_config = train_config.model_copy(
        update={
            "run_name": "oom-test",
            "oom_retries": 2,
            "max_seq_len": 128,
            "resume_from_checkpoint": True,
        }
    )
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text("placeholder: true\n", encoding="utf-8")

    run_dir = Path(oom_train_config.output_dir) / "oom-test"
    checkpoint_dir = run_dir / "checkpoint-4"
    _ = checkpoint_dir.mkdir(parents=True)
    _ = (checkpoint_dir / "trainer_state.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("aiinfra_e2e.train.sft.load_hf_dataset", lambda config: _build_dataset())
    monkeypatch.setattr(
        "aiinfra_e2e.train.sft._load_tokenizer", lambda *args, **kwargs: _FakeTokenizer()
    )
    monkeypatch.setattr("aiinfra_e2e.train.sft._load_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(
        "aiinfra_e2e.train.sft._prepare_train_split", lambda *args, **kwargs: _build_dataset()
    )
    monkeypatch.setattr("aiinfra_e2e.train.sft.SFTTrainer", _FakeTrainer)
    monkeypatch.setattr("aiinfra_e2e.train.sft.SFTConfig", _DummyTrainingArgs)
    monkeypatch.setattr("aiinfra_e2e.train.sft.build_lora_config", lambda config: None)
    monkeypatch.setattr("aiinfra_e2e.train.sft._set_seed", lambda seed: None)
    monkeypatch.setattr("aiinfra_e2e.train.sft._should_enable_qlora", lambda: False)
    monkeypatch.setattr("aiinfra_e2e.train.sft.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("aiinfra_e2e.train.sft.torch.cuda.is_bf16_supported", lambda: False)

    _FakeTrainer.attempts = 0
    _FakeTrainer.batch_sizes = []
    _FakeTrainer.max_lengths = []
    _FakeTrainer.resume_args = []
    _FakeTrainer.saved_model_dirs = []

    returned_run_dir = run_sft(
        data_config=data_config,
        train_config=oom_train_config,
        obs_config=obs_config,
        data_config_path=config_path,
        train_config_path=config_path,
        obs_config_path=config_path,
    )

    assert returned_run_dir == run_dir
    assert _FakeTrainer.attempts == 3
    assert _FakeTrainer.batch_sizes == [1, 1, 1]
    assert _FakeTrainer.max_lengths == [128, 64, 32]
    assert _FakeTrainer.resume_args == [
        str(checkpoint_dir),
        str(checkpoint_dir),
        str(checkpoint_dir),
    ]
    assert _FakeTrainer.saved_model_dirs == [str(run_dir / "lora_adapter")]
