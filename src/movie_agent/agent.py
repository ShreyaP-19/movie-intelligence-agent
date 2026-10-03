"""The agent: routes each message, guards parameters, calls RAG and the MCP email tool."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .catalog import Catalog, resolve_title
from .models import Citation
from .rag import RAGEngine
from .router import RouteDecision, Router

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
FIND_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

SendEmail = Callable[..., dict]


@dataclass
class AgentResult:
    route: str  # informational | email_action | needs_clarification | error
    reply: str
    sources: list[Citation] = field(default_factory=list)
    email: dict | None = None
    trace: list[str] = field(default_factory=list)


class MovieAgent:
    def __init__(self, catalog: Catalog, rag: RAGEngine, router: Router, send_email: SendEmail):
        self.catalog = catalog
        self.rag = rag
        self.router = router
        self.send_email = send_email

    # ------------------------------------------------------------------ public API
    def handle(self, message: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        trace: list[str] = []
        movies = self.catalog.all()
        if not movies:
            return AgentResult("needs_clarification", "The library is empty. Ingest subtitles first "
                               "(`movie-ingest --subtitles data/subtitles`).", trace=trace)
        try:
            decision = self.router.route(message, history, [m.label for m in movies])
        except Exception as exc:  # network / API failure
            return AgentResult("error", f"I couldn't analyse that request ({exc}).", trace=trace)
        trace.append(f"Router → **{decision.intent}** — {decision.reasoning}")

        if decision.intent == "needs_clarification":
            return self._clarify(decision.clarification_question or "Could you give me a bit more detail?", trace)

        recipient = None
        if decision.intent == "email_action":
            recipient = self._recipient(decision, message)
            if not recipient:
                return self._clarify("Who should I send this to? Please give me the recipient's email address.", trace)

        movie_ids, problem = self._resolve_movies(decision.movie_titles)
        if problem:
            return self._clarify(problem, trace)
        if movie_ids:
            trace.append("Resolved movies: " + ", ".join(self.catalog.get(i).label for i in movie_ids))

        query = decision.rewritten_query or message
        hits = self.rag.retrieve(query, movie_ids or None)
        trace.append(f"RAG: retrieved {len(hits)} passage(s) for “{query}”")
        if not hits:
            msg = "I couldn't find anything relevant in the subtitles for that."
            if decision.intent == "email_action":
                msg += " I haven't sent any email."
            return AgentResult(decision.intent, msg, trace=trace)

        if decision.intent == "informational":
            reply, cites = self.rag.answer(query, hits)
            return AgentResult("informational", reply, cites, trace=trace)
        return self._email(decision, recipient, query, hits, trace)

    # ------------------------------------------------------------------ email branch
    def _email(self, decision: RouteDecision, recipient: str, query: str, hits, trace) -> AgentResult:
        instruction = decision.email_goal or "Summarise the relevant scenes and dialogue."
        subject, body, cites = self.rag.draft_email(
            instruction=instruction, recipient=recipient, question=query, hits=hits)
        trace.append(f"Drafted email “{subject}”")
        try:
            result = self.send_email(to=recipient, subject=subject, body=body)
        except Exception as exc:
            trace.append(f"MCP send_email failed: {exc}")
            email = {"to": recipient, "subject": subject, "body": body, "status": "failed", "detail": str(exc)}
            return AgentResult("email_action", f"I drafted the email but sending failed: {exc}", cites, email, trace)
        status = result.get("status", "unknown")
        trace.append(f"MCP send_email → {status}")
        email = {"to": recipient, "subject": subject, "body": body, **result}
        verb = {"sent": "Sent", "dry_run": "Saved (dry-run, SMTP not configured)"}.get(status, status)
        reply = f"✅ {verb}: email to **{recipient}** — *{subject}*\n\n{body}"
        return AgentResult("email_action", reply, cites, email, trace)

    # ------------------------------------------------------------------ guards
    @staticmethod
    def _clarify(question: str, trace: list[str]) -> AgentResult:
        trace.append("Asking for clarification instead of guessing")
        return AgentResult("needs_clarification", question, trace=trace)

    @staticmethod
    def _recipient(decision: RouteDecision, message: str) -> str | None:
        if decision.recipient_email and EMAIL_RE.match(decision.recipient_email.strip()):
            return decision.recipient_email.strip()
        found = FIND_EMAIL_RE.search(message)
        return found.group(0) if found else None

    def _resolve_movies(self, titles: list[str]) -> tuple[list[str], str | None]:
        ids: list[str] = []
        movies = self.catalog.all()
        for t in titles:
            res = resolve_title(t, movies)
            if res.status == "exact":
                if res.matches[0].movie_id not in ids:
                    ids.append(res.matches[0].movie_id)
            elif res.status == "ambiguous":
                opts = "; ".join(m.label for m in res.matches)
                return [], f"“{t}” could mean several movies in the library: {opts}. Which one do you mean?"
            else:
                hint = (" Did you mean: " + "; ".join(m.label for m in res.matches) + "?") if res.matches else ""
                return [], f"I couldn't find “{t}” in the library.{hint}"
        return ids, None


def build_agent() -> MovieAgent:
    """Wire the production components (imports are lazy so tests stay light)."""
    from .config import settings
    from .llm import LLM
    from .mcp_client import send_email_via_mcp
    from .vector_store import SubtitleIndex

    catalog = Catalog(settings.catalog_path)
    index = SubtitleIndex(settings.chroma_dir, settings.embedding_model, settings.min_similarity)
    llm = LLM.from_env()
    return MovieAgent(catalog, RAGEngine(index, llm, settings.top_k), Router(llm), send_email_via_mcp)
