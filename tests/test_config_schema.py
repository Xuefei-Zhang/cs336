from pathlib import Path

import pytest
from pydantic import ValidationError

from aiinfra_e2e.config import DataConfig, load_yaml


def test_load_yaml_validates_unknown_fields_with_file_path(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    _ = config_path.write_text(
        "dataset_id: hfl/alpaca_zh_51k\nsplit: train\nunexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=str(config_path)):
        _ = load_yaml(config_path, DataConfig)


def test_load_yaml_returns_validated_model(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    _ = config_path.write_text(
        "dataset_id: hfl/alpaca_zh_51k\nsplit: train\ntext_field: output\n",
        encoding="utf-8",
    )

    config = load_yaml(config_path, DataConfig)

    assert config.dataset_id == "hfl/alpaca_zh_51k"
    assert config.split == "train"
    assert config.text_field == "output"
