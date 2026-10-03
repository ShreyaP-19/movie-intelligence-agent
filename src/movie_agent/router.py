"""Intent routing: informational vs email action vs clarification."""
from __future__ import annotations

from dataclasses import dataclass, field

ROUTER_SYSTEM = """You are the routing brain of a movie-subtitle assistant. The library contains subtitles for the movies listed below. Decide how to handle the user's latest message.

Intents:
- informational: a question about a movie's plot, scenes, characters, dialogue or quotes (including "who said ...", "which movie has ..."). Searching the whole library is fine when no movie is named.
- email_action: the user wants something sent/shared/emailed to someone (scene breakdown, dialogue summary, quote analysis, ...).
- needs_clarification: you cannot proceed without guessing. Use it ONLY when:
  * an email is requested but no recipient email address is given (a name alone is not an address),
  * the user refers to "it", "that", "the movie" and nothing in the conversation says which,
  * the title is vague or could match several library movies (e.g. "Batman" when several Batman films exist),
  * the quote/scene reference is too vague to search (e.g. "that famous line").
  Do not ask for clarification when a reasonable search over the library would work.

Fill the fields:
- rewritten_query: a standalone search query that resolves pronouns/follow-ups using the conversation (what to look up in the subtitles).
- movie_titles: titles the user named, exactly as they wrote them (empty list if none).
- recipient_email: the email address if the user gave one (also from earlier turns), else null.
- email_goal: for email_action, what the email should contain, in the user's words.
- clarification_question: for needs_clarification, ONE short, specific question (offer options if you can).
- reasoning: one short sentence explaining the routing.

Library: {library}"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["informational", "email_action", "needs_clarification"]},
        "rewritten_query": {"type": "string"},
        "movie_titles": {"type": "array", "items": {"type": "string"}},
        "recipient_email": {"type": "string"},
        "email_goal": {"type": "string"},
        "clarification_question": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["intent", "rewritten_query", "movie_titles", "reasoning"],
}


@dataclass
class RouteDecision:
    intent: str
    rewritten_query: str = ""
    movie_titles: list[str] = field(default_factory=list)
    recipient_email: str | None = None
    email_goal: str | None = None
    clarification_question: str | None = None
    reasoning: str = ""


class Router:
    def __init__(self, llm):
        self.llm = llm

    def route(self, message: str, history: list[dict], library: list[str]) -> RouteDecision:
        convo = "\n".join(f"{m['role'].upper()}: {m['content'][:1500]}" for m in history[-6:])
        user = f"Conversation so far:\n{convo or '(none)'}\n\nLatest user message:\n{message}"
        out = self.llm.structured(
            system=ROUTER_SYSTEM.format(library="; ".join(library) or "(empty)"),
            user=user, schema=ROUTER_SCHEMA, name="route",
            description="Classify the request and extract its parameters.",
        )
        intent = out.get("intent")
        if intent not in {"informational", "email_action", "needs_clarification"}:
            intent = "needs_clarification"
        return RouteDecision(
            intent=intent,
            rewritten_query=(out.get("rewritten_query") or message).strip(),
            movie_titles=[t for t in out.get("movie_titles", []) if t and t.strip()],
            recipient_email=(out.get("recipient_email") or None),
            email_goal=out.get("email_goal"),
            clarification_question=out.get("clarification_question"),
            reasoning=out.get("reasoning", ""),
        )
