"""Process-local Python startup customizations for spawned subprocesses."""

import os

from aiinfra_e2e.serve.tokenizer_compat import ensure_all_special_tokens_extended
from aiinfra_e2e.serve.tqdm_compat import ensure_vllm_tqdm_compat

if os.environ.get("AIINFRA_E2E_ENABLE_VLLM_TOKENIZER_COMPAT") == "1":
    ensure_all_special_tokens_extended()

if os.environ.get("AIINFRA_E2E_ENABLE_VLLM_TQDM_COMPAT") == "1":
    ensure_vllm_tqdm_compat()
