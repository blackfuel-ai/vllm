# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OpenAI-shaped error responses, including stable error codes."""

from http import HTTPStatus

from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import APIErrorCode, VLLMNotFoundError, VLLMValidationError


def test_context_length_exceeded_emits_stable_code():
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


def test_body_omits_status_code_and_exposes_string_code():
    exc = VLLMValidationError(
        "too long", error_code=APIErrorCode.CONTEXT_LENGTH_EXCEEDED
    )

    body = create_error_response(exc).model_dump()["error"]

    assert body["code"] == "context_length_exceeded"
    assert "status_code" not in body


def test_validation_error_without_code_has_null_code():
    resp = create_error_response(VLLMValidationError("bad value"))

    assert resp.error.code is None
    assert resp.error.status_code == HTTPStatus.BAD_REQUEST.value


def test_not_found_maps_to_404():
    resp = create_error_response(VLLMNotFoundError("missing"))

    assert resp.error.code is None
    assert resp.error.type == "NotFoundError"
    assert resp.error.status_code == HTTPStatus.NOT_FOUND.value
