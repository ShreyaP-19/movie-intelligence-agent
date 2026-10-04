"""MCP server exposing a single `send_email` tool (stdio transport).

Self-contained on purpose: it can be launched by any MCP client, e.g.
    python src/movie_agent/mcp_email_server.py
or registered in Claude Desktop / MCP Inspector as a stdio server.

Delivery modes
  * SMTP_HOST set        -> real delivery over SMTP (STARTTLS by default)
  * SMTP_HOST empty or
    EMAIL_DRY_RUN=true   -> the message is written as an .eml file under ./outbox
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
mcp = FastMCP("movie-email")


def _addresses(value: str | None) -> list[str]:
    return [a.strip() for a in re.split(r"[;,]", value or "") if a.strip()]


@mcp.tool()
def send_email(to: str, subject: str, body: str, cc: str | None = None) -> str:
    """Send a plain-text email.

    Args:
        to: Recipient email address (comma-separated for several).
        subject: Subject line.
        body: Plain-text message body.
        cc: Optional CC address(es), comma-separated.

    Returns a JSON string: {"status": "sent" | "dry_run", "to": [...], "detail": "..."}.
    """
    recipients, copies = _addresses(to), _addresses(cc)
    bad = [a for a in recipients + copies if not EMAIL_RE.match(a)]
    if not recipients or bad:
        raise ValueError(f"Invalid recipient address(es): {bad or to!r}")
    if not subject.strip() or not body.strip():
        raise ValueError("Subject and body must not be empty.")

    host = os.getenv("SMTP_HOST", "").strip()
    sender = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or "movie-assistant@localhost"

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, ", ".join(recipients), subject
    if copies:
        msg["Cc"] = ", ".join(copies)
    msg.set_content(body)

    if not host or os.getenv("EMAIL_DRY_RUN", "false").lower() in {"1", "true", "yes"}:
        outbox = Path(os.getenv("OUTBOX_DIR", ROOT / "outbox"))
        outbox.mkdir(parents=True, exist_ok=True)
        path = outbox / f"{int(time.time() * 1000)}.eml"
        path.write_bytes(bytes(msg))
        return json.dumps({"status": "dry_run", "to": recipients, "detail": f"saved to {path}"})

    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=30) as smtp:
        if os.getenv("SMTP_STARTTLS", "true").lower() in {"1", "true", "yes"}:
            smtp.starttls()
        if os.getenv("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.getenv("SMTP_PASSWORD", ""))
        smtp.send_message(msg, to_addrs=recipients + copies)
    return json.dumps({"status": "sent", "to": recipients, "detail": f"delivered via {host}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
