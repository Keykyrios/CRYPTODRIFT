"""
CryptoDrift MCP Server.

Implements a minimal MCP server per the 2025-03-26 spec.
Exposes one tool: refine_code.

INTENTIONAL SECURITY OMISSIONS (reproducing arXiv:2601.17549):
- No capability attestation
- No origin authentication
- No input sanitization beyond basic type checking

These are deliberate research choices to study how MCP protocol
vulnerabilities interact with cryptographic code degradation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Callable, Optional

from .jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    JsonRpcErrorCode,
)
from .messages import (
    METHOD_INITIALIZE,
    METHOD_INITIALIZED,
    METHOD_TOOLS_LIST,
    METHOD_TOOLS_CALL,
    METHOD_PING,
    REFINE_CODE_TOOL,
    make_initialize_response,
    make_tools_list_response,
    make_tool_call_response,
    MessageInterceptor,
)
from .transport import (
    StdioServerTransport,
    TransportError,
    TransportClosedError,
)

# All server logging goes to stderr per MCP spec
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[CryptoDrift-MCP] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# Type for the refinement callback that the experiment harness provides
RefinementCallback = Callable[
    [str, int, str],  # code, iteration, strategy
    tuple[str, dict, dict],  # refined_code, attention_data, vuln_data
]


class CryptoDriftMCPServer:
    """
    MCP server that exposes the refine_code tool.

    Protocol flow (per MCP spec):
    1. Client sends 'initialize' request
    2. Server responds with capabilities
    3. Client sends 'notifications/initialized' notification
    4. Client can now call tools via 'tools/call'
    """

    def __init__(
        self,
        refinement_callback: Optional[RefinementCallback] = None,
    ):
        self._transport = StdioServerTransport()
        self._refinement_callback = refinement_callback
        self._initialized = False
        self._running = False

        # Method dispatch table
        self._handlers: dict[str, Callable] = {
            METHOD_INITIALIZE: self._handle_initialize,
            METHOD_INITIALIZED: self._handle_initialized,
            METHOD_TOOLS_LIST: self._handle_tools_list,
            METHOD_TOOLS_CALL: self._handle_tools_call,
            METHOD_PING: self._handle_ping,
        }

    @property
    def interceptor(self) -> MessageInterceptor:
        return self._transport.interceptor

    def set_refinement_callback(self, callback: RefinementCallback) -> None:
        """Set the callback that performs actual code refinement."""
        self._refinement_callback = callback

    async def run(self) -> None:
        """Main server loop — reads requests and dispatches handlers."""
        await self._transport.start()
        self._running = True

        logger.info("CryptoDrift MCP server started")

        while self._running:
            try:
                message = await self._transport.receive(timeout=300.0)
            except TransportClosedError:
                logger.info("Client disconnected")
                break
            except TransportError as e:
                logger.error("Transport error: %s", e)
                continue

            if isinstance(message, JsonRpcRequest):
                await self._dispatch_request(message)
            elif isinstance(message, JsonRpcResponse):
                # Server shouldn't normally receive responses
                # but log it per our interception goals
                logger.warning(
                    "Received unexpected response: id=%s", message.id
                )
            else:
                logger.warning("Unknown message type: %s", type(message))

        logger.info("CryptoDrift MCP server stopped")

    async def stop(self) -> None:
        """Signal the server to stop."""
        self._running = False

    async def _dispatch_request(self, request: JsonRpcRequest) -> None:
        """Dispatch a request to the appropriate handler."""
        handler = self._handlers.get(request.method)

        if handler is None:
            if not request.is_notification:
                error_resp = JsonRpcResponse.error_response(
                    id=request.id,
                    error=JsonRpcError.method_not_found(
                        f"Unknown method: {request.method}"
                    ),
                )
                await self._transport.respond(error_resp)
            return

        try:
            response = await handler(request)
            if response is not None and not request.is_notification:
                await self._transport.respond(response)
        except Exception as e:
            logger.exception("Handler error for %s", request.method)
            if not request.is_notification:
                error_resp = JsonRpcResponse.error_response(
                    id=request.id,
                    error=JsonRpcError.internal_error(str(e)),
                )
                await self._transport.respond(error_resp)

    async def _handle_initialize(
        self, request: JsonRpcRequest
    ) -> JsonRpcResponse:
        """Handle the initialize request."""
        logger.info("Received initialize request")

        # NOTE: No capability attestation — intentional per research design
        # NOTE: No origin authentication — intentional per research design
        # See arXiv:2601.17549 for why this matters

        response = make_initialize_response(request.id)
        return response

    async def _handle_initialized(
        self, request: JsonRpcRequest
    ) -> Optional[JsonRpcResponse]:
        """Handle the initialized notification."""
        self._initialized = True
        logger.info("Client initialized — session ready")
        # Notification — no response
        return None

    async def _handle_tools_list(
        self, request: JsonRpcRequest
    ) -> JsonRpcResponse:
        """Handle tools/list — return available tools."""
        return make_tools_list_response(
            request.id,
            tools=[REFINE_CODE_TOOL],
        )

    async def _handle_tools_call(
        self, request: JsonRpcRequest
    ) -> JsonRpcResponse:
        """Handle tools/call — dispatch to the appropriate tool."""
        if not self._initialized:
            return JsonRpcResponse.error_response(
                id=request.id,
                error=JsonRpcError(
                    code=-32002,
                    message="Server not initialized",
                ),
            )

        params = request.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "refine_code":
            return await self._handle_refine_code(request.id, arguments)
        else:
            return JsonRpcResponse.error_response(
                id=request.id,
                error=JsonRpcError.method_not_found(
                    f"Unknown tool: {tool_name}"
                ),
            )

    async def _handle_refine_code(
        self,
        request_id: Any,
        arguments: dict,
    ) -> JsonRpcResponse:
        """
        Handle the refine_code tool call.

        Arguments:
            code: str — Python crypto code to refine
            iteration: int — Current iteration number
            strategy: str — Refinement strategy (EF/FF/SF/AI)

        Returns refined code + analysis data.
        """
        # Validate arguments
        code = arguments.get("code")
        iteration = arguments.get("iteration")
        strategy = arguments.get("strategy")

        if not isinstance(code, str) or not code.strip():
            return JsonRpcResponse.error_response(
                id=request_id,
                error=JsonRpcError.invalid_params("'code' must be a non-empty string"),
            )

        if not isinstance(iteration, int) or iteration < 0:
            return JsonRpcResponse.error_response(
                id=request_id,
                error=JsonRpcError.invalid_params(
                    "'iteration' must be a non-negative integer"
                ),
            )

        if strategy not in ("EF", "FF", "SF", "AI"):
            return JsonRpcResponse.error_response(
                id=request_id,
                error=JsonRpcError.invalid_params(
                    "'strategy' must be one of: EF, FF, SF, AI"
                ),
            )

        # Call the refinement callback
        if self._refinement_callback is None:
            return JsonRpcResponse.error_response(
                id=request_id,
                error=JsonRpcError.internal_error(
                    "No refinement callback configured"
                ),
            )

        try:
            refined_code, attention_data, vuln_data = (
                self._refinement_callback(code, iteration, strategy)
            )
        except Exception as e:
            logger.exception("Refinement callback failed")
            return JsonRpcResponse.error_response(
                id=request_id,
                error=JsonRpcError.internal_error(
                    f"Refinement failed: {e}"
                ),
            )

        # Build response content
        content = [
            {
                "type": "text",
                "text": json.dumps({
                    "refined_code": refined_code,
                    "iteration": iteration,
                    "strategy": strategy,
                    "attention_summary": attention_data,
                    "vulnerability_delta": vuln_data,
                }),
            },
        ]

        return make_tool_call_response(request_id, content)

    async def _handle_ping(
        self, request: JsonRpcRequest
    ) -> JsonRpcResponse:
        """Handle ping request."""
        return JsonRpcResponse.success(id=request.id, result={})


async def run_server(
    refinement_callback: Optional[RefinementCallback] = None,
) -> None:
    """Entry point for running the MCP server."""
    server = CryptoDriftMCPServer(refinement_callback)
    await server.run()
