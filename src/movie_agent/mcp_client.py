"""MCP client: starts the email server over stdio and calls its `send_email` tool."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).with_name("mcp_email_server.py")


async def _call(to: str, subject: str, body: str, cc: str | None) -> dict:
    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_PATH)], env={**os.environ}  # pass SMTP settings through
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            if "send_email" not in tools:
                raise RuntimeError(f"MCP server does not expose send_email (found: {sorted(tools)})")
            args = {"to": to, "subject": subject, "body": body}
            if cc:
                args["cc"] = cc
            result = await session.call_tool("send_email", args)
            text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
            if result.isError:
                raise RuntimeError(text or "send_email failed")
            return json.loads(text)


def send_email_via_mcp(to: str, subject: str, body: str, cc: str | None = None) -> dict:
    """Synchronous helper used by the agent."""
    return asyncio.run(_call(to, subject, body, cc))
