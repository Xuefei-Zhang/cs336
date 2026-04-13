# Learnings


- Task 1 scaffold: setuptools build backend worked for editable installs in this environment, while hatchling was not importable during pip editable resolution.
- Task 3 CLI: Typer subcommands can stay forward-compatible by sharing a single `--config` loader helper and keeping non-implemented commands as validation-only stubs until later tasks land.
