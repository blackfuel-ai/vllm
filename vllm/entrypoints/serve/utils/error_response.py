# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    GenerationError,
)
from vllm.entrypoints.serve.utils.api_utils import sanitize_message
from vllm.logger import init_logger

logger = init_logger(__name__)


def _classify_exception(exc: Exception) -> tuple[str, HTTPStatus, str | None]:
    """Map an exception to its (err_type, HTTP status, param)."""
    from vllm.exceptions import VLLMNotFoundError, VLLMValidationError

    if isinstance(exc, VLLMValidationError):
        return "BadRequestError", HTTPStatus.BAD_REQUEST, exc.parameter
    if isinstance(exc, VLLMNotFoundError):
        return "NotFoundError", HTTPStatus.NOT_FOUND, None
    if isinstance(exc, (ValueError, TypeError, OverflowError)):
        return "BadRequestError", HTTPStatus.BAD_REQUEST, None
    if isinstance(exc, NotImplementedError):
        return "NotImplementedError", HTTPStatus.NOT_IMPLEMENTED, None
    if isinstance(exc, GenerationError):
        return "InternalServerError", exc.status_code, None
    if any(cls.__name__ == "TemplateError" for cls in type(exc).__mro__):
        # jinja2.TemplateError and its subclasses (avoid importing jinja2)
        return "BadRequestError", HTTPStatus.BAD_REQUEST, None
    return "InternalServerError", HTTPStatus.INTERNAL_SERVER_ERROR, None


def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse:
    exc: Exception | None = None
    code: str | None = None

    if isinstance(message, Exception):
        exc = message
        logger.debug(
            "create_error_response called with %s: %s", type(exc).__name__, exc
        )

        err_type, status_code, param = _classify_exception(exc)
        error_code = getattr(exc, "error_code", None)
        if error_code is not None:
            code = error_code.value
        message = str(exc)

    return ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=err_type,
            code=code,
            status_code=status_code.value,
            param=param,
        )
    )
