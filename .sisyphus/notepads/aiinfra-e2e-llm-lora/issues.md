# Issues


- Task 1 issue: persistent shell session did not resolve newly installed console scripts (`aiinfra-e2e`, `ruff`) on PATH, so verification used the active Python bin path directly; LSP diagnostics also could not discover basedpyright-langserver despite installation.
