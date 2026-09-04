# SPDX-License-Identifier: Apache-2.0
"""gfx942 sparse-MLA decode path selection.

On gfx942 (MI300/MI325) the sparse-MLA decode takes the split-K path tuned
for gfx950 unless ``VLLM_ROCM_SPARSE_DECODE_TUNED_GFX942=0``; gfx950 always
takes it and every other arch keeps the generic path. These tests pin the
selection truth table, the lenient spelling of the switch, and the
import-time binding of ``_TUNED_SPARSE_DECODE`` in the ops module, without
a GPU.
"""

import importlib
import logging
import sys
import types

import pytest

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

# These tests run on CPU-only Tier-0; the default cleanup fixture tries to
# empty an accelerator cache that does not exist.
pytestmark = pytest.mark.skip_global_cleanup

SWITCH = "VLLM_ROCM_SPARSE_DECODE_TUNED_GFX942"


@pytest.mark.parametrize(
    ("on_gfx942", "on_gfx950", "gfx942_tuned", "expected"),
    [
        (False, True, False, True),  # gfx950: native, switch irrelevant
        (False, True, True, True),
        (True, False, True, True),  # gfx942 with the switch on (default)
        (True, False, False, False),  # gfx942 opted out
        (False, False, True, False),  # other archs never take the tuned path
        (False, False, False, False),
    ],
)
def test_resolve_tuned_sparse_decode(on_gfx942, on_gfx950, gfx942_tuned, expected):
    assert (
        ops._resolve_tuned_sparse_decode(on_gfx942, on_gfx950, gfx942_tuned) is expected
    )


def _read_switch() -> bool:
    return envs.environment_variables[SWITCH]()


def test_switch_defaults_on(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert _read_switch() is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("0", False),
        ("false", False),
        # Lenient like every other VLLM_* bool: unknown spellings read as off.
        ("yes", False),
        ("on", False),
        ("2", False),
        ("", False),
    ],
)
def test_switch_reads_leniently(monkeypatch, value, expected):
    monkeypatch.setenv(SWITCH, value)
    assert _read_switch() is expected


def _fake_rocm_platform_module(*, on_gfx942: bool, on_gfx950: bool):
    """Stand in for ``vllm.platforms.rocm``, which needs a GPU to import."""
    mod = types.ModuleType("vllm.platforms.rocm")
    mod.__dict__["_ON_GFX942"] = on_gfx942
    mod.__dict__["_ON_GFX950"] = on_gfx950
    return mod


@pytest.fixture
def reload_ops(monkeypatch, caplog):
    """Reload the ops module as if imported on the given arch with the given env.

    Returns a callable ``(on_gfx942, on_gfx950, switch) -> module``; the ops
    module is reloaded unpatched on teardown so later tests see the real
    CPU-only binding.
    """
    module_logger = logging.getLogger(ops.__name__)
    # The vllm logger hierarchy does not propagate to the root logger.
    module_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=ops.__name__)

    def _reload(on_gfx942: bool, on_gfx950: bool, switch: str | None):
        with monkeypatch.context() as m:
            m.setattr(current_platform, "is_rocm", lambda: True)
            m.setitem(
                sys.modules,
                "vllm.platforms.rocm",
                _fake_rocm_platform_module(on_gfx942=on_gfx942, on_gfx950=on_gfx950),
            )
            if switch is None:
                m.delenv(SWITCH, raising=False)
            else:
                m.setenv(SWITCH, switch)
            return importlib.reload(ops)

    yield _reload

    module_logger.removeHandler(caplog.handler)
    importlib.reload(ops)


def _selection_lines(caplog) -> list[str]:
    return sorted(
        {
            r.getMessage()
            for r in caplog.records
            if "Sparse-MLA decode" in r.getMessage()
        }
    )


def test_import_binding_gfx942_default_takes_tuned_path(reload_ops, caplog):
    module = reload_ops(on_gfx942=True, on_gfx950=False, switch=None)
    assert module._TUNED_SPARSE_DECODE is True
    assert _selection_lines(caplog) == [
        f"Sparse-MLA decode on gfx942 uses the gfx950-tuned split-K path ({SWITCH}=1)."
    ]


def test_import_binding_gfx942_opt_out_takes_generic_path(reload_ops, caplog):
    module = reload_ops(on_gfx942=True, on_gfx950=False, switch="0")
    assert module._TUNED_SPARSE_DECODE is False
    assert _selection_lines(caplog) == [
        f"Sparse-MLA decode on gfx942 uses the generic path ({SWITCH}=0)."
    ]


def test_import_binding_gfx950_ignores_switch(reload_ops, caplog):
    module = reload_ops(on_gfx942=False, on_gfx950=True, switch="0")
    assert module._TUNED_SPARSE_DECODE is True
    assert _selection_lines(caplog) == []


def test_import_binding_other_arch_never_takes_tuned_path(reload_ops, caplog):
    module = reload_ops(on_gfx942=False, on_gfx950=False, switch="1")
    assert module._TUNED_SPARSE_DECODE is False
    assert _selection_lines(caplog) == []
