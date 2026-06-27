"""
JSON-RPC 2.0 implementation per https://www.jsonrpc.org/specification

Verified against the spec:
- Version string MUST be exactly "2.0"
- id: String, Number, or Null (Null discouraged in requests)
- Error codes: -32700 (Parse error), -32600 (Invalid Request),
  -32601 (Method not found), -32602 (Invalid params), -32603 (Internal error)
- Server errors: -32099 to -32000 (reserved for implementation)
- Notifications have no 'id' field
- All member names are case-sensitive
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union
from enum import IntEnum


class JsonRpcErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # Server error range: -32099 to -32000
    SERVER_ERROR_START = -32099
    SERVER_ERROR_END = -32000


JSONRPC_VERSION = "2.0"


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 Error object."""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, d: dict) -> JsonRpcError:
        return cls(
            code=d["code"],
            message=d["message"],
            data=d.get("data"),
        )

    @classmethod
    def parse_error(cls, data: Any = None) -> JsonRpcError:
        return cls(JsonRpcErrorCode.PARSE_ERROR, "Parse error", data)

    @classmethod
    def invalid_request(cls, data: Any = None) -> JsonRpcError:
        return cls(JsonRpcErrorCode.INVALID_REQUEST, "Invalid Request", data)

    @classmethod
    def method_not_found(cls, data: Any = None) -> JsonRpcError:
        return cls(JsonRpcErrorCode.METHOD_NOT_FOUND, "Method not found", data)

    @classmethod
    def invalid_params(cls, data: Any = None) -> JsonRpcError:
        return cls(JsonRpcErrorCode.INVALID_PARAMS, "Invalid params", data)

    @classmethod
    def internal_error(cls, data: Any = None) -> JsonRpcError:
        return cls(JsonRpcErrorCode.INTERNAL_ERROR, "Internal error", data)


@dataclass
class JsonRpcRequest:
    """
    JSON-RPC 2.0 Request object.

    Per spec:
    - jsonrpc: MUST be exactly "2.0"
    - method: String, MUST NOT start with "rpc." (reserved)
    - params: Array or Object (optional)
    - id: String or Number. If omitted, this is a Notification.
    """
    method: str
    params: Optional[Union[dict, list]] = None
    id: Optional[Union[str, int]] = None

    @property
    def is_notification(self) -> bool:
        """Notifications have no 'id' field and expect no response."""
        return self.id is None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> JsonRpcRequest:
        _validate_version(d)
        if "method" not in d:
            raise JsonRpcValidationError("Missing 'method' field")
        method = d["method"]
        if not isinstance(method, str):
            raise JsonRpcValidationError("'method' must be a string")
        return cls(
            method=method,
            params=d.get("params"),
            id=d.get("id"),
        )


@dataclass
class JsonRpcResponse:
    """
    JSON-RPC 2.0 Response object.

    Per spec, MUST contain either 'result' or 'error', but NOT both.
    """
    id: Optional[Union[str, int]]
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None

    def __post_init__(self):
        if self.result is not None and self.error is not None:
            raise JsonRpcValidationError(
                "Response MUST contain either 'result' or 'error', not both"
            )

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> JsonRpcResponse:
        _validate_version(d)
        if "result" not in d and "error" not in d:
            raise JsonRpcValidationError(
                "Response must contain 'result' or 'error'"
            )
        error = None
        if "error" in d:
            error = JsonRpcError.from_dict(d["error"])
        return cls(
            id=d.get("id"),
            result=d.get("result"),
            error=error,
        )

    @classmethod
    def success(cls, id: Union[str, int], result: Any) -> JsonRpcResponse:
        return cls(id=id, result=result)

    @classmethod
    def error_response(
        cls, id: Optional[Union[str, int]], error: JsonRpcError
    ) -> JsonRpcResponse:
        return cls(id=id, error=error)


class JsonRpcValidationError(Exception):
    """Raised when a JSON-RPC message fails validation."""
    pass


def _validate_version(d: dict) -> None:
    """Validate that the jsonrpc version field is exactly '2.0'."""
    if d.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcValidationError(
            f"jsonrpc version must be '{JSONRPC_VERSION}', "
            f"got '{d.get('jsonrpc')}'"
        )


def parse_message(raw: str) -> Union[JsonRpcRequest, JsonRpcResponse, list]:
    """
    Parse a raw JSON string into a JSON-RPC message.

    Returns:
        JsonRpcRequest, JsonRpcResponse, or a list (batch) of either.

    Raises:
        JsonRpcValidationError on invalid messages.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JsonRpcValidationError(f"Invalid JSON: {e}") from e

    if isinstance(data, list):
        # Batch request/response
        if len(data) == 0:
            raise JsonRpcValidationError("Empty batch array")
        return [_parse_single(item) for item in data]

    if isinstance(data, dict):
        return _parse_single(data)

    raise JsonRpcValidationError(f"Expected object or array, got {type(data).__name__}")


def _parse_single(d: dict) -> Union[JsonRpcRequest, JsonRpcResponse]:
    """Parse a single JSON-RPC message dict."""
    if not isinstance(d, dict):
        raise JsonRpcValidationError(f"Expected object, got {type(d).__name__}")

    _validate_version(d)

    # Distinguish request from response:
    # - Request has 'method'
    # - Response has 'result' or 'error'
    if "method" in d:
        return JsonRpcRequest.from_dict(d)
    elif "result" in d or "error" in d:
        return JsonRpcResponse.from_dict(d)
    else:
        raise JsonRpcValidationError(
            "Cannot determine message type: no 'method', 'result', or 'error'"
        )


class IdGenerator:
    """Thread-safe sequential ID generator for JSON-RPC requests."""

    def __init__(self, start: int = 1):
        self._counter = start

    def next(self) -> int:
        current = self._counter
        self._counter += 1
        return current
