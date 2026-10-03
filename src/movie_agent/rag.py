"""Retrieval + grounded generation with verifiable citations."""
from __future__ import annotations

import re

from .models import Citation, Hit

ANSWER_SYSTEM = """You answer questions about movies using ONLY the numbered subtitle excerpts supplied.
Rules:
- End every factual statement with citation markers such as [S2] or [S1][S4] that refer to the excerpts.
- Subtitles contain dialogue only: no speaker names and no scene descriptions. When you infer who is speaking or what is happening on screen, say it is an inference and stay cautious.
- Quote dialogue verbatim and keep each quote short (one or two lines).
- If the excerpts do not contain the answer, say you could not find it in the subtitles. Never use outside knowledge and never write timestamps yourself; the markers carry them."""

EMAIL_SYSTEM = """You write short, clear plain-text emails that share information about movies, using ONLY the numbered subtitle excerpts supplied.
Rules:
- Follow the sender's instruction about what the email should contain (scene breakdown, dialogue summary, quote analysis, ...).
- Put citation markers like [S1] after every fact or quote that comes from an excerpt.
- Do NOT add a sources list, signature or timestamps; they are appended automatically.
- Subtitles have no speaker names or scene descriptions; flag inferences as such.
- Address the recipient politely; keep the email under 250 words unless asked otherwise."""

EMAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "description": "Concise subject line"},
        "body": {"type": "string", "description": "Plain-text email body with [S#] markers"},
    },
    "required": ["subject", "body"],
}

_MARKER = re.compile(r"\[S(\d+)\]")


class RAGEngine:
    def __init__(self, index, llm, top_k: int = 8):
        self.index = index
        self.llm = llm
        self.top_k = top_k

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, movie_ids: list[str] | None = None, k: int | None = None) -> list[Hit]:
        k = k or (self.top_k + 4 if movie_ids and len(movie_ids) == 1 else self.top_k)
        hits = self.index.search(query, k=k * 2, movie_ids=movie_ids)
        hits = _drop_overlaps(hits)[:k]
        # present chronologically per movie so the model reads scenes in order
        return sorted(hits, key=lambda h: (h.movie_label, h.start_ms))

    @staticmethod
    def build_context(hits: list[Hit]) -> str:
        return "\n\n".join(f"[S{i}] {h.citation}\n{h.text}" for i, h in enumerate(hits, 1))

    # ------------------------------------------------------------------ generation
    def answer(self, question: str, hits: list[Hit]) -> tuple[str, list[Citation]]:
        user = f"Subtitle excerpts:\n\n{self.build_context(hits)}\n\nQuestion: {question}"
        text = self.llm.complete(system=ANSWER_SYSTEM, user=user)
        return finalize(text, hits)

    def draft_email(
        self, *, instruction: str, recipient: str, question: str, hits: list[Hit]
    ) -> tuple[str, str, list[Citation]]:
        user = (
            f"Subtitle excerpts:\n\n{self.build_context(hits)}\n\n"
            f"Recipient: {recipient}\nTopic / question: {question}\n"
            f"Sender's instruction: {instruction}"
        )
        out = self.llm.structured(
            system=EMAIL_SYSTEM, user=user, schema=EMAIL_SCHEMA, name="draft_email",
            description="Return the email subject and body.",
        )
        body, cites = finalize(out["body"], hits)
        body += "\n\n-- \nSent by the Movie Intelligence Assistant"
        return out["subject"].strip(), body, cites


# ---------------------------------------------------------------------- helpers
def finalize(text: str, hits: list[Hit]) -> tuple[str, list[Citation]]:
    """Validate [S#] markers against real hits and append a Sources section.

    Markers that point at non-existent excerpts are stripped, so a citation can
    never refer to a timestamp the retriever did not return.
    """
    used: list[int] = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(hits):
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""

    text = _MARKER.sub(keep, text).strip()
    if not used:  # model forgot to cite: fall back to listing everything retrieved
        used = list(range(1, len(hits) + 1))
    cites = [
        Citation(f"S{n}", hits[n - 1].citation, hits[n - 1].movie_id, hits[n - 1].start_ms,
                 hits[n - 1].end_ms, _excerpt(hits[n - 1].text))
        for n in sorted(used)
    ]
    lines = "\n".join(f"[{c.tag}] {c.citation}" for c in cites)
    return f"{text}\n\nSources:\n{lines}", cites


def _excerpt(text: str, n: int = 110) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _drop_overlaps(hits: list[Hit], threshold: float = 0.5) -> list[Hit]:
    """Overlapping windows of the same movie are near-duplicates; keep the best-ranked one."""
    kept: list[Hit] = []
    for h in hits:  # already sorted by score
        dup = False
        for k in kept:
            if k.movie_id != h.movie_id:
                continue
            inter = min(k.end_ms, h.end_ms) - max(k.start_ms, h.start_ms)
            if inter > 0 and inter / max(1, min(k.end_ms - k.start_ms, h.end_ms - h.start_ms)) > threshold:
                dup = True
                break
        if not dup:
            kept.append(h)
    return kept
