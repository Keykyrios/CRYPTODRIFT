"""
Stdio transport for MCP.

Verified against MCP spec (2025-03-26) Section "stdio":
- Messages are delimited by NEWLINES (NOT Content-Length headers)
- Messages MUST NOT contain embedded newlines
- Server reads from stdin, writes to stdout
- Logging goes to stderr ONLY
- Client launches server as subprocess

This is a critical divergence from LSP-style transports.
The MCP stdio spec explicitly says: "Messages are delimited by newlines"
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from typing import Optional, AsyncIterator, Union

from .jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    parse_message,
    JsonRpcValidationError,
)
from .messages import MessageInterceptor

logger = logging.getLogger(__name__)


class StdioTransport:
    """
    MCP stdio transport implementation.

    Per MCP spec (2025-03-26):
    - Newline-delimited JSON messages (one JSON object per line)
    - No Content-Length headers
    - No embedded newlines in messages
    """

    def __init__(self, interceptor: Optional[MessageInterceptor] = None):
        self._interceptor = interceptor or MessageInterceptor()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    @property
    def interceptor(self) -> MessageInterceptor:
        return self._interceptor

    async def send(
        self,
        message: Union[JsonRpcRequest, JsonRpcResponse],
        writer: asyncio.StreamWriter,
        direction: str = "outbound",
    ) -> None:
        """
        Send a JSON-RPC message over the transport.

        Per MCP stdio spec:
        - One JSON object per line
        - No embedded newlines
        - Terminated by newline
        """
        raw = message.to_json()

        # Ensure no embedded newlines (spec requirement)
        if "\n" in raw or "\r" in raw:
            raise TransportError(
                "MCP stdio messages MUST NOT contain embedded newlines"
            )

        self._interceptor.intercept(direction, raw, message)

        line = raw + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()

    async def receive_line(
        self,
        reader: asyncio.StreamReader,
        direction: str = "inbound",
        timeout: Optional[float] = None,
    ) -> Union[JsonRpcRequest, JsonRpcResponse]:
        """
        Receive a single JSON-RPC message from the transport.

        Reads one newline-delimited line and parses it.
        """
        try:
            if timeout is not None:
                raw_bytes = await asyncio.wait_for(
                    reader.readline(), timeout=timeout
                )
            else:
                raw_bytes = await reader.readline()
        except asyncio.TimeoutError:
            raise TransportTimeoutError("Read timed out")

        if not raw_bytes:
            raise TransportClosedError("Transport closed (EOF)")

        raw = raw_bytes.decode("utf-8").rstrip("\n").rstrip("\r")

        if not raw:
            raise TransportError("Received empty message")

        try:
            parsed = parse_message(raw)
        except JsonRpcValidationError as e:
            raise TransportError(f"Invalid JSON-RPC message: {e}") from e

        # parse_message can return a list (batch) but we handle single
        # messages for simplicity in the MCP harness
        if isinstance(parsed, list):
            # For batch, intercept the whole thing but return first item
            # Full batch support can be added later if needed
            self._interceptor.intercept(direction, raw, None)
            raise TransportError(
                "Batch messages not yet supported in CryptoDrift harness"
            )

        self._interceptor.intercept(direction, raw, parsed)
        return parsed


class StdioServerTransport(StdioTransport):
    """
    Server-side stdio transport.

    Reads from sys.stdin, writes to sys.stdout.
    All logging MUST go to stderr (per MCP spec).
    """

    def __init__(self, interceptor: Optional[MessageInterceptor] = None):
        super().__init__(interceptor)
        self._stdin_reader: Optional[asyncio.StreamReader] = None
        self._stdout_writer: Optional[asyncio.StreamWriter] = None

    async def start(self) -> None:
        """Set up async readers/writers for stdin/stdout."""
        loop = asyncio.get_event_loop()

        # Create async reader for stdin
        self._stdin_reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self._stdin_reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        # Create async writer for stdout
        transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer
        )
        self._stdout_writer = asyncio.StreamWriter(
            transport, protocol, None, loop
        )

    async def receive(
        self, timeout: Optional[float] = None
    ) -> Union[JsonRpcRequest, JsonRpcResponse]:
        """Receive a message from stdin."""
        if self._stdin_reader is None:
            raise TransportError("Transport not started")
        return await self.receive_line(
            self._stdin_reader, "client_to_server", timeout
        )

    async def respond(
        self, message: Union[JsonRpcRequest, JsonRpcResponse]
    ) -> None:
        """Send a message to stdout."""
        if self._stdout_writer is None:
            raise TransportError("Transport not started")
        await self.send(message, self._stdout_writer, "server_to_client")


class StdioClientTransport(StdioTransport):
    """
    Client-side stdio transport.

    Launches the server as a subprocess and communicates
    via stdin/stdout pipes.
    """

    def __init__(
        self,
        server_command: list[str],
        interceptor: Optional[MessageInterceptor] = None,
    ):
        super().__init__(interceptor)
        self._server_command = server_command
        self._process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> None:
        """Launch the server subprocess."""
        logger.info("Launching MCP server: %s", self._server_command)
        self._process = await asyncio.create_subprocess_exec(
            *self._server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("Server process started (PID: %d)", self._process.pid)

    async def send_request(
        self, message: Union[JsonRpcRequest, JsonRpcResponse]
    ) -> None:
        """Send a message to the server's stdin."""
        if self._process is None or self._process.stdin is None:
            raise TransportError("Transport not started")

        raw = message.to_json()
        if "\n" in raw or "\r" in raw:
            raise TransportError("Messages MUST NOT contain embedded newlines")

        self._interceptor.intercept("client_to_server", raw, message)

        line = raw + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def receive_response(
        self, timeout: Optional[float] = 30.0
    ) -> Union[JsonRpcRequest, JsonRpcResponse]:
        """Receive a message from the server's stdout."""
        if self._process is None or self._process.stdout is None:
            raise TransportError("Transport not started")

        return await self.receive_line(
            self._process.stdout, "server_to_client", timeout
        )

    async def stop(self) -> None:
        """Stop the server subprocess."""
        if self._process is not None:
            logger.info("Stopping MCP server (PID: %d)", self._process.pid)
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Server did not exit, killing")
                self._process.kill()
                await self._process.wait()
            logger.info("Server process stopped")

    async def read_stderr(self) -> str:
        """Read any available stderr output (server logs)."""
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(
                self._process.stderr.read(4096), timeout=0.1
            )
            return data.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            return ""


class TransportError(Exception):
    """Base transport error."""
    pass


class TransportClosedError(TransportError):
    """Transport connection was closed."""
    pass


class TransportTimeoutError(TransportError):
    """Transport operation timed out."""
    pass
