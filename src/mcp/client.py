"""
CryptoDrift MCP Client.

Drives multi-turn refinement sessions by sending tool calls
to the MCP server and tracking conversation context.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Union

from .jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    IdGenerator,
)
from .messages import (
    make_initialize_request,
    make_initialized_notification,
    make_tools_list_request,
    make_tool_call_request,
    MessageInterceptor,
)
from .transport import (
    StdioClientTransport,
    TransportError,
    TransportClosedError,
    TransportTimeoutError,
)

logger = logging.getLogger(__name__)


class CryptoDriftMCPClient:
    """
    MCP client that drives iterative code refinement sessions.

    Usage:
        client = CryptoDriftMCPClient(["python", "-m", "src.mcp.server"])
        await client.connect()
        result = await client.refine_code(code, iteration=0, strategy="SF")
        await client.disconnect()
    """

    def __init__(
        self,
        server_command: list[str],
        interceptor: Optional[MessageInterceptor] = None,
    ):
        self._interceptor = interceptor or MessageInterceptor()
        self._transport = StdioClientTransport(
            server_command, self._interceptor
        )
        self._id_gen = IdGenerator()
        self._connected = False
        self._server_info: Optional[dict] = None
        self._available_tools: list[dict] = []

    @property
    def interceptor(self) -> MessageInterceptor:
        return self._interceptor

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def server_info(self) -> Optional[dict]:
        return self._server_info

    async def connect(self) -> None:
        """
        Connect to the MCP server.

        Performs the MCP initialization handshake:
        1. Send 'initialize' request
        2. Receive capabilities response
        3. Send 'notifications/initialized' notification
        4. Fetch available tools
        """
        await self._transport.start()

        # Step 1: Send initialize request
        init_req = make_initialize_request(self._id_gen)
        await self._transport.send_request(init_req)

        # Step 2: Receive initialize response
        init_resp = await self._transport.receive_response(timeout=30.0)
        if not isinstance(init_resp, JsonRpcResponse):
            raise MCPClientError(
                f"Expected initialize response, got {type(init_resp).__name__}"
            )
        if init_resp.is_error:
            raise MCPClientError(
                f"Initialize failed: {init_resp.error.message}"
            )

        self._server_info = init_resp.result.get("serverInfo", {})
        logger.info(
            "Connected to server: %s v%s",
            self._server_info.get("name", "unknown"),
            self._server_info.get("version", "unknown"),
        )

        # Step 3: Send initialized notification
        init_notif = make_initialized_notification()
        await self._transport.send_request(init_notif)

        # Step 4: Fetch available tools
        tools_req = make_tools_list_request(self._id_gen)
        await self._transport.send_request(tools_req)

        tools_resp = await self._transport.receive_response(timeout=10.0)
        if isinstance(tools_resp, JsonRpcResponse) and not tools_resp.is_error:
            self._available_tools = tools_resp.result.get("tools", [])
            logger.info(
                "Available tools: %s",
                [t["name"] for t in self._available_tools],
            )

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        await self._transport.stop()
        self._connected = False
        logger.info("Disconnected from server")

    async def refine_code(
        self,
        code: str,
        iteration: int,
        strategy: str,
    ) -> dict:
        """
        Call the refine_code tool on the server.

        Args:
            code: Python cryptographic code to refine.
            iteration: Current iteration number (0-indexed).
            strategy: Refinement strategy (EF/FF/SF/AI).

        Returns:
            Dict containing refined_code, attention_summary,
            vulnerability_delta.
        """
        if not self._connected:
            raise MCPClientError("Not connected to server")

        # Build and send the tool call request
        request = make_tool_call_request(
            self._id_gen,
            tool_name="refine_code",
            arguments={
                "code": code,
                "iteration": iteration,
                "strategy": strategy,
            },
        )
        await self._transport.send_request(request)

        # Receive the response
        response = await self._transport.receive_response(timeout=120.0)

        if not isinstance(response, JsonRpcResponse):
            raise MCPClientError(
                f"Expected response, got {type(response).__name__}"
            )

        if response.is_error:
            raise MCPClientError(
                f"Tool call failed: {response.error.message} "
                f"(code: {response.error.code})"
            )

        # Parse the tool result content
        result = response.result
        content_blocks = result.get("content", [])

        if not content_blocks:
            raise MCPClientError("Empty tool response")

        # Extract the text content (JSON payload)
        text_block = content_blocks[0]
        if text_block.get("type") != "text":
            raise MCPClientError(
                f"Expected text content, got {text_block.get('type')}"
            )

        try:
            payload = json.loads(text_block["text"])
        except (json.JSONDecodeError, KeyError) as e:
            raise MCPClientError(f"Failed to parse tool response: {e}") from e

        return payload

    async def ping(self) -> bool:
        """Send a ping to the server. Returns True if server responds."""
        try:
            from .messages import make_ping_request
            request = make_ping_request(self._id_gen)
            await self._transport.send_request(request)
            response = await self._transport.receive_response(timeout=5.0)
            return (
                isinstance(response, JsonRpcResponse)
                and not response.is_error
            )
        except (TransportError, MCPClientError):
            return False


class MCPClientError(Exception):
    """Client-side MCP error."""
    pass
