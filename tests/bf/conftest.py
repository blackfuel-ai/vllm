# SPDX-License-Identifier: Apache-2.0
"""Shared guards for the BF feature-test layer (`tests/bf/`).

Tier-0 (bf-cpu-smoke) runs this whole tree with no GPU and no model
download. A BF test that genuinely needs either still lives here but must
opt out of Tier-0 by depending on one of these guards, so the smoke run
skips it cleanly instead of failing on a missing accelerator or network
fetch. See bf-docs/adr/0007-functional-test-strategy.md (Tier-0/Tier-1).
"""

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _cuda_available(),
    reason="BF Tier-1: needs a GPU; skipped on the CPU-only cpu-smoke run",
)

requires_model_download = pytest.mark.skipif(
    True,
    reason="BF Tier-1: needs a model/tokenizer download; offline on cpu-smoke",
)
