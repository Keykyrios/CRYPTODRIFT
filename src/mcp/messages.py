"""
MCP message types per the MCP specification (2025-03-26).

Verified against: https://modelcontextprotocol.io/specification/2025-03-26

Key MCP message patterns:
- initialize: Client -> Server (request), Server -> Client (response)
- initialized: Client -> Server (notification, no response expected)
- tools/list: Client -> Server (request)
- tools/call: Client -> Server (request)

All messages use JSON-RPC 2.0 as the wire format.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .jsonrpc import JsonRpcRequest, JsonRpcResponse, IdGenerator

logger = logging.getLogger(__name__)


# ── MCP Protocol Constants ─────────────────────────────────────────
MCP_PROTOCOL_VERSION = "2025-03-26"

# MCP method names (from the spec)
METHOD_INITIALIZE = "initialize"
METHOD_INITIALIZED = "notifications/initialized"
METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"
METHOD_PING = "ping"


# ── MCP Capability Declarations ────────────────────────────────────
# NOTE: Deliberately omitting capability attestation per arXiv:2601.17549
# to reproduce the MCP vulnerability for research purposes.

def make_server_capabilities() -> dict:
    """
    Server capabilities declaration.
    INTENTIONALLY omits attestation — reproducing MCP spec vulnerability.
    """
    return {
        "tools": {
            "listChanged": False,
        },
    }


def make_client_capabilities() -> dict:
    """Client capabilities declaration."""
    return {}


# ── Initialize Messages ───────────────────────────────────────────

def make_initialize_request(id_gen: IdGenerator) -> JsonRpcRequest:
    """Create an MCP initialize request (client -> server)."""
    return JsonRpcRequest(
        method=METHOD_INITIALIZE,
        params={
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": make_client_capabilities(),
            "clientInfo": {
                "name": "CryptoDrift",
                "version": "0.1.0",
            },
        },
        id=id_gen.next(),
    )


def make_initialize_response(request_id: Union[str, int]) -> JsonRpcResponse:
    """Create an MCP initialize response (server -> client)."""
    return JsonRpcResponse.success(
        id=request_id,
        result={
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": make_server_capabilities(),
            "serverInfo": {
                "name": "CryptoDrift-MCP-Server",
                "version": "0.1.0",
            },
        },
    )


def make_initialized_notification() -> JsonRpcRequest:
    """
    Create an MCP initialized notification (client -> server).
    This is a notification (no id), so no response is expected.
    """
    return JsonRpcRequest(
        method=METHOD_INITIALIZED,
        params={},
        id=None,  # Notification — no id
    )


# ── Tool Messages ─────────────────────────────────────────────────

def make_tools_list_request(id_gen: IdGenerator) -> JsonRpcRequest:
    """Create a tools/list request (client -> server)."""
    return JsonRpcRequest(
        method=METHOD_TOOLS_LIST,
        params={},
        id=id_gen.next(),
    )


def make_tools_list_response(
    request_id: Union[str, int], tools: list[dict]
) -> JsonRpcResponse:
    """Create a tools/list response (server -> client)."""
    return JsonRpcResponse.success(
        id=request_id,
        result={"tools": tools},
    )


def make_tool_call_request(
    id_gen: IdGenerator,
    tool_name: str,
    arguments: dict,
) -> JsonRpcRequest:
    """Create a tools/call request (client -> server)."""
    return JsonRpcRequest(
        method=METHOD_TOOLS_CALL,
        params={
            "name": tool_name,
            "arguments": arguments,
        },
        id=id_gen.next(),
    )


def make_tool_call_response(
    request_id: Union[str, int],
    content: list[dict],
    is_error: bool = False,
) -> JsonRpcResponse:
    """
    Create a tools/call response (server -> client).

    content: list of content blocks, each with 'type' and 'text'.
    Per MCP spec, tool results contain content array.
    """
    return JsonRpcResponse.success(
        id=request_id,
        result={
            "content": content,
            "isError": is_error,
        },
    )


def make_ping_request(id_gen: IdGenerator) -> JsonRpcRequest:
    """Create a ping request."""
    return JsonRpcRequest(
        method=METHOD_PING,
        params={},
        id=id_gen.next(),
    )


# ── Tool Definitions ──────────────────────────────────────────────

REFINE_CODE_TOOL = {
    "name": "refine_code",
    "description": (
        "Refine cryptographic Python code according to a specified strategy. "
        "Returns the refined code along with attention and vulnerability analysis."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python cryptographic code to refine.",
            },
            "iteration": {
                "type": "integer",
                "description": "Current iteration number (0-indexed).",
            },
            "strategy": {
                "type": "string",
                "enum": ["EF", "FF", "SF", "AI"],
                "description": (
                    "Refinement strategy: EF=Efficiency, FF=Feature, "
                    "SF=Security, AI=Ambiguous."
                ),
            },
        },
        "required": ["code", "iteration", "strategy"],
    },
}


# ── Message Interceptor ──────────────────────────────────────────

@dataclass
class InterceptedMessage:
    """A message captured by the interceptor with metadata."""
    timestamp: float
    direction: str  # "client_to_server" or "server_to_client"
    raw_json: str
    parsed: Optional[Union[JsonRpcRequest, JsonRpcResponse]] = None
    size_bytes: int = 0

    def __post_init__(self):
        self.size_bytes = len(self.raw_json.encode("utf-8"))


class MessageInterceptor:
    """
    Intercepts and logs all MCP messages for analysis.
    Records timestamps, direction, and raw content.
    """

    def __init__(self):
        self.messages: list[InterceptedMessage] = []
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def intercept(
        self,
        direction: str,
        raw_json: str,
        parsed: Optional[Union[JsonRpcRequest, JsonRpcResponse]] = None,
    ) -> InterceptedMessage:
        """Record an intercepted message."""
        msg = InterceptedMessage(
            timestamp=time.time(),
            direction=direction,
            raw_json=raw_json,
            parsed=parsed,
        )
        if self._enabled:
            self.messages.append(msg)
            logger.debug(
                "MCP [%s] %d bytes: %.100s...",
                direction,
                msg.size_bytes,
                raw_json,
            )
        return msg

    def get_messages(
        self, direction: Optional[str] = None
    ) -> list[InterceptedMessage]:
        """Get intercepted messages, optionally filtered by direction."""
        if direction is None:
            return list(self.messages)
        return [m for m in self.messages if m.direction == direction]

    def clear(self) -> None:
        """Clear all intercepted messages."""
        self.messages.clear()

    def total_bytes(self) -> int:
        """Total bytes of all intercepted messages."""
        return sum(m.size_bytes for m in self.messages)
