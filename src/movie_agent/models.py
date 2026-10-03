"""Plain dataclasses shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass


def ms_to_clock(ms: int) -> str:
    """Milliseconds -> HH:MM:SS (used in citations)."""
    s = max(0, int(ms)) // 1000
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def ms_to_srt(ms: int) -> str:
    """Milliseconds -> HH:MM:SS,mmm (SRT notation)."""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


@dataclass(frozen=True)
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    movie_id: str
    movie_title: str
    year: int | None
    chunk_index: int
    start_ms: int
    end_ms: int
    text: str
    source_file: str
    first_cue: int
    last_cue: int

    @property
    def movie_label(self) -> str:
        return f"{self.movie_title} ({self.year})" if self.year else self.movie_title


@dataclass
class MovieInfo:
    movie_id: str
    title: str
    year: int | None
    source_file: str
    file_hash: str
    num_cues: int
    num_chunks: int
    duration_ms: int

    @property
    def label(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


@dataclass
class Hit:
    """One retrieved subtitle passage with its provenance."""

    chunk_id: str
    movie_id: str
    movie_label: str
    start_ms: int
    end_ms: int
    text: str
    similarity: float | None  # cosine similarity; None if found only lexically
    score: float              # fused (hybrid) ranking score

    @property
    def time_range(self) -> str:
        return f"{ms_to_clock(self.start_ms)} – {ms_to_clock(self.end_ms)}"

    @property
    def citation(self) -> str:
        return f"{self.movie_label} | {self.time_range}"


@dataclass
class Citation:
    tag: str          # "S1"
    citation: str     # "Inception (2010) | 00:12:03 – 00:12:41"
    movie_id: str
    start_ms: int
    end_ms: int
    excerpt: str
