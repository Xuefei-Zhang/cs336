# repo review docs

This directory contains generated walkthrough material for understanding the current repo and especially the `make e2e` path.

## Files

- `make-e2e-overview.md` — shortest useful introduction to the e2e pipeline
- `make-e2e-codepath-review.md` — ordered codepath trace from `make e2e` down into Python runtime stages
- `repo-structure-map.md` — file-by-file guide for where the important logic lives
- `manual-serve-command-codepath-review.md` — exact runtime path for the manual `aiinfra_e2e.cli serve --config artifacts/tmp/interactive-serve.yaml` command
- `qwen3.5-122b-a10b-gptq-int4-config-review.md` — source-grounded explanation of the local Qwen3.5 model config and what it implies for this repo's serve path

## Diagrams

- `assets/make-e2e-overview.svg`
- `assets/make-e2e-stage-mapping.svg`

## Recommended reading order

1. `make-e2e-overview.md`
2. `make-e2e-codepath-review.md`
3. `repo-structure-map.md`
4. `manual-serve-command-codepath-review.md`
5. `qwen3.5-122b-a10b-gptq-int4-config-review.md`
