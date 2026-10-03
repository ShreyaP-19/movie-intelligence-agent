"""Movie catalog (JSON on disk) and fuzzy title resolution."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .models import MovieInfo


class Catalog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._movies: dict[str, MovieInfo] = {}
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._movies = {d["movie_id"]: MovieInfo(**d) for d in data}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(m) for m in self._movies.values()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert(self, info: MovieInfo) -> None:
        self._movies[info.movie_id] = info

    def remove(self, movie_id: str) -> None:
        self._movies.pop(movie_id, None)

    def clear(self) -> None:
        self._movies.clear()

    def get(self, movie_id: str) -> MovieInfo | None:
        return self._movies.get(movie_id)

    def by_source(self, source_file: str) -> MovieInfo | None:
        return next((m for m in self._movies.values() if m.source_file == source_file), None)

    def all(self) -> list[MovieInfo]:
        return sorted(self._movies.values(), key=lambda m: m.label.lower())


# --------------------------------------------------------------------------- resolution
@dataclass
class Resolution:
    status: str  # "exact" | "ambiguous" | "not_found"
    matches: list[MovieInfo] = field(default_factory=list)


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def _score(qn: str, tn: str) -> float:
    if not qn or not tn:
        return 0.0
    if qn == tn:
        return 1.0
    s = SequenceMatcher(None, qn, tn).ratio()
    if (len(qn) >= 4 and f" {qn} " in f" {tn} ") or (len(tn) >= 4 and f" {tn} " in f" {qn} "):
        s = max(s, 0.9)
    return s


def resolve_title(query: str, movies: list[MovieInfo]) -> Resolution:
    """Map a user-written title to catalog entries without guessing.

    exact      -> one clear match
    ambiguous  -> several plausible movies (remakes, franchises) or a weak single match
    not_found  -> nothing close enough (matches holds loose suggestions)
    """
    if not movies or not query.strip():
        return Resolution("not_found")
    ym = re.search(r"\b((?:19|20)\d{2})\b", query)
    year = int(ym.group(1)) if ym else None
    q_full = _norm(query)
    q_noyear = _norm(re.sub(r"\(?\b(?:19|20)\d{2}\b\)?", " ", query)) if ym else q_full

    scored = sorted(
        ((max(_score(q_full, _norm(m.title)), _score(q_noyear, _norm(m.title))), m) for m in movies),
        key=lambda x: -x[0],
    )
    best = scored[0][0]
    if best < 0.6:
        return Resolution("not_found", [m for s, m in scored[:3] if s >= 0.45])

    if best >= 0.95:
        top = [m for s, m in scored if s >= 0.95]
    else:
        top = [m for s, m in scored if s >= max(0.6, best - 0.1)]
    if year:
        same_year = [m for m in top if m.year == year]
        if same_year:
            top = same_year
    if len(top) == 1 and best >= 0.85:
        return Resolution("exact", top)
    return Resolution("ambiguous", top[:5])
