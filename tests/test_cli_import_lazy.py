import builtins
import importlib
import sys

import pytest


def test_cli_import_does_not_import_training_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked_prefixes = ("peft", "torchvision")
    original_import = builtins.__import__
    module_names = [
        "aiinfra_e2e.cli",
        "aiinfra_e2e.train",
        "aiinfra_e2e.train.sft",
        "peft",
        "torchvision",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    for module_name in module_names:
        sys.modules.pop(module_name, None)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(blocked_prefixes):
            raise AssertionError(f"unexpected heavy dependency import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    cli = importlib.import_module("aiinfra_e2e.cli")

    assert cli.__name__ == "aiinfra_e2e.cli"
    assert hasattr(cli, "app")
    assert "peft" not in sys.modules
    assert "torchvision" not in sys.modules

    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, module in previous_modules.items():
        if module is not None:
            sys.modules[module_name] = module
