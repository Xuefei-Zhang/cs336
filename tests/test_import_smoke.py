import importlib


def test_package_imports() -> None:
    package = importlib.import_module("aiinfra_e2e")
    assert package.__name__ == "aiinfra_e2e"


def test_cli_module_imports() -> None:
    cli = importlib.import_module("aiinfra_e2e.cli")
    assert cli.__name__ == "aiinfra_e2e.cli"
