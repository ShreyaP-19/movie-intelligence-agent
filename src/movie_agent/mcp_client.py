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


def _leaf_errors(exc: BaseException):
    """anyio wraps errors in ExceptionGroup ('unhandled errors in a TaskGroup');
    dig out the real underlying exceptions so the user sees the actual cause."""
    subs = getattr(exc, "exceptions", None)
    if subs:
        for e in subs:
            yield from _leaf_errors(e)
    else:
        yield exc


async def _call(to: str, subject: str, body: str, cc: str | None) -> dict:
    params = StdioServerParameters(
        command=sys.executable, args=[str(SERVER_PATH)], env={**os.environ}  # pass SMTP settings through
    )
    args = {"to": to, "subject": subject, "body": body}
    if cc:
        args["cc"] = cc
    # Do the MCP I/O inside the context managers, but raise *after* leaving them:
    # exceptions raised inside get re-wrapped by anyio and hide the real message.
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            result = await session.call_tool("send_email", args) if "send_email" in tools else None
    if result is None:
        raise RuntimeError(f"MCP server does not expose send_email (found: {sorted(tools)})")
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    if result.isError:
        raise RuntimeError(text or "send_email failed")
    return json.loads(text)


def send_email_via_mcp(to: str, subject: str, body: str, cc: str | None = None) -> dict:
    """Synchronous helper used by the agent."""
    try:
        return asyncio.run(_call(to, subject, body, cc))
    except Exception as exc:
        causes = list(_leaf_errors(exc))
        if len(causes) == 1 and causes[0] is exc:
            raise
        detail = "; ".join(f"{type(e).__name__}: {e}" for e in causes)
        raise RuntimeError(f"{detail} (see the terminal running Streamlit for the MCP server log)") from exc