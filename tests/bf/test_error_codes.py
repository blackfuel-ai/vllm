# SPDX-License-Identifier: Apache-2.0
"""Seed BF feature test: the OpenAI-standard structured error-code surface.

Worked reference for the `tests/bf/` convention (bf-docs/AGENTS.md). Guards
the [bf-patch] behaviour shipped in PR #84: a stable string `code` on error
bodies, decoupled from the HTTP status, threaded from a raising exception's
`error_code` through the single `create_error_response` chokepoint. Offline,
CPU-safe, no model download — so it runs in Tier-0 (bf-cpu-smoke).
"""

from http import HTTPStatus

from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import (
    APIErrorCode,
    VLLMNotFoundError,
    VLLMValidationError,
)


def test_context_length_exceeded_threads_stable_code():
    """A coded validation error surfaces its OpenAI string code + 400."""
    exc = VLLMValidationError(
        "This model's maximum context length is 100 tokens.",
        parameter="input_tokens",
        error_code=APIErrorCode.CONTEXT_LENGTH_EXCEEDED,
    )

    resp = create_error_response(exc)

    assert resp.error.code == "context_length_exceeded"
    assert resp.error.type == "BadRequestError"
    assert resp.error.param == "input_tokens"
    assert resp.error.status_code == HTTPStatus.BAD_REQUEST.value


def test_status_code_stays_out_of_the_wire_body():
    """`code` is the OpenAI string; the HTTP status never leaks into JSON."""
    exc = VLLMValidationError(
        "too long", error_code=APIErrorCode.CONTEXT_LENGTH_EXCEEDED
    )

    body = create_error_response(exc).model_dump()["error"]

    assert body["code"] == "context_length_exceeded"
    assert "status_code" not in body


def test_uncoded_validation_error_has_null_code():
    """An error with no `error_code` reports a null `code`, not a status int."""
    resp = create_error_response(VLLMValidationError("bad value"))

    assert resp.error.code is None
    assert resp.error.status_code == HTTPStatus.BAD_REQUEST.value


def test_not_found_maps_to_404_without_a_code():
    resp = create_error_response(VLLMNotFoundError("missing"))

    assert resp.error.code is None
    assert resp.error.type == "NotFoundError"
    assert resp.error.status_code == HTTPStatus.NOT_FOUND.value


def test_string_message_path_carries_no_code():
    """The non-exception entry point keeps `code` null (no exception to read)."""
    resp = create_error_response("plain message", param="model")

    assert resp.error.code is None
    assert resp.error.message == "plain message"
    assert resp.error.param == "model"


def test_only_known_code_is_defined():
    """Lock the shipped enum surface so a new code is a deliberate edit."""
    assert {c.name for c in APIErrorCode} == {"CONTEXT_LENGTH_EXCEEDED"}
    assert APIErrorCode.CONTEXT_LENGTH_EXCEEDED.value == "context_length_exceeded"
