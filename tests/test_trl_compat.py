import importlib
import sys
import types

import pytest


def test_disable_torchvision_if_broken_patches_transformers_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trl_compat = importlib.import_module("aiinfra_e2e.train.trl_compat")
    import transformers.utils
    import transformers.utils.import_utils as import_utils

    original_import_module = importlib.import_module
    original_utils_check = transformers.utils.is_torchvision_available
    original_import_utils_check = import_utils.is_torchvision_available
    original_submodule = sys.modules.get("torchvision._meta_registrations")

    sys.modules["torchvision._meta_registrations"] = types.ModuleType(
        "torchvision._meta_registrations"
    )

    def broken_import_module(name: str, package: str | None = None):
        if name == "torchvision":
            raise AttributeError(
                "partially initialized module 'torchvision' has no attribute 'extension'"
            )
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", broken_import_module)
    monkeypatch.setattr(transformers.utils, "is_torchvision_available", lambda: True)
    monkeypatch.setattr(import_utils, "is_torchvision_available", lambda: True)
    trl_compat.disable_torchvision_if_broken.cache_clear()

    assert transformers.utils.is_torchvision_available() is True
    assert import_utils.is_torchvision_available() is True

    trl_compat.disable_torchvision_if_broken()

    assert transformers.utils.is_torchvision_available() is False
    assert import_utils.is_torchvision_available() is False
    assert "torchvision._meta_registrations" not in sys.modules
    trl_compat.disable_torchvision_if_broken.cache_clear()
    monkeypatch.setattr(transformers.utils, "is_torchvision_available", original_utils_check)
    monkeypatch.setattr(import_utils, "is_torchvision_available", original_import_utils_check)
    if original_submodule is None:
        sys.modules.pop("torchvision._meta_registrations", None)
    else:
        sys.modules["torchvision._meta_registrations"] = original_submodule
