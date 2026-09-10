# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conformance of the in-tree DeepSeek-V4.1 encoder against the checkpoint.

The reference is a byte-verbatim copy of ``encoding/encoding.py`` from
``deepseek-ai/DeepSeek-V4.1-Flash`` at revision ``dba1be0a``, vendored under
``reference/``. It is loaded by path rather than imported as a package so the
copy stays a data file: nothing edits it to satisfy a test.

The alias table is the reason this exists. ``reasoning_effort`` is an integer
budget in 1-100, and the string aliases are a convenience mapping on top. When
the two implementations disagree on that mapping, the same alias renders a
different prompt in vLLM than in the vendor's own reference, silently, with no
error at any layer -- and the default (``"high"``) is the case every
unconfigured deployment hits.
"""

import importlib.util
from pathlib import Path

from vllm.tokenizers.deepseek_v41_encoding import (
    REASONING_EFFORT_MAPPINGS,
    REASONING_EFFORT_TEMPLATE,
)

REFERENCE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"

# The checkpoint defines no `xhigh`; vLLM adds it. The guard therefore asserts
# agreement on the shared aliases and pins the extra one separately, so an
# upstream change to either is caught without asserting a value the vendor
# never published.
VLLM_ONLY_ALIASES = {"xhigh"}


def _load_reference():
    path = Path(__file__).parent / "reference" / "encoding_dsv41.py"
    spec = importlib.util.spec_from_file_location("_dsv41_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load_reference()


def test_shared_aliases_match_reference():
    shared = set(REASONING_EFFORT_MAPPINGS) - VLLM_ONLY_ALIASES
    assert shared == set(reference.REASONING_EFFORT_MAPPINGS), (
        "the in-tree alias set diverged from the checkpoint's; if the vendor "
        "added or removed an alias, refresh the reference and this guard"
    )
    for alias in sorted(shared):
        assert REASONING_EFFORT_MAPPINGS[alias] == (
            reference.REASONING_EFFORT_MAPPINGS[alias]
        ), (
            f"alias {alias!r} renders "
            f"{REASONING_EFFORT_MAPPINGS[alias]} in vLLM but "
            f"{reference.REASONING_EFFORT_MAPPINGS[alias]} in the checkpoint "
            f"reference -- the same request would produce different prompts"
        )


def test_default_effort_matches_reference():
    from vllm.tokenizers.deepseek_v41_encoding import DEFAULT_REASONING_EFFORT

    assert DEFAULT_REASONING_EFFORT == reference.DEFAULT_REASONING_EFFORT
    assert REASONING_EFFORT_MAPPINGS[DEFAULT_REASONING_EFFORT] == (
        reference.REASONING_EFFORT_MAPPINGS[reference.DEFAULT_REASONING_EFFORT]
    ), "the unconfigured path must render the vendor's own default budget"


def test_vllm_only_alias_is_ordered_and_pinned():
    # `xhigh` has no vendor counterpart, so it is pinned here rather than
    # compared. It must still sit strictly between `high` and `max` for the
    # ladder to read monotonically.
    assert REASONING_EFFORT_MAPPINGS["xhigh"] == 88
    assert (
        REASONING_EFFORT_MAPPINGS["high"]
        < REASONING_EFFORT_MAPPINGS["xhigh"]
        < REASONING_EFFORT_MAPPINGS["max"]
    )


def test_effort_template_matches_reference_verbatim():
    assert REASONING_EFFORT_TEMPLATE == reference.REASONING_EFFORT_TEMPLATE


def test_every_alias_is_in_the_documented_range():
    for alias, budget in REASONING_EFFORT_MAPPINGS.items():
        assert 1 <= budget <= 100, f"{alias!r} maps outside the 1-100 range"
