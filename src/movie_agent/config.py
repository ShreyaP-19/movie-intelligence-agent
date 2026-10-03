"""Central configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    subtitles_dir: Path
    chroma_dir: Path
    catalog_path: Path
    embedding_model: str
    llm_model: str
    top_k: int
    min_similarity: float
    max_movies: int
    chunk_max_chars: int
    chunk_max_seconds: int
    chunk_overlap_cues: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            subtitles_dir=Path(os.getenv("SUBTITLES_DIR", ROOT / "data" / "subtitles")),
            chroma_dir=Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db")),
            catalog_path=Path(os.getenv("CATALOG_PATH", ROOT / "data" / "catalog.json")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
            top_k=int(os.getenv("TOP_K", "8")),
            min_similarity=float(os.getenv("MIN_SIMILARITY", "0.25")),
            max_movies=int(os.getenv("MAX_MOVIES", "100")),
            chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "600")),
            chunk_max_seconds=int(os.getenv("CHUNK_MAX_SECONDS", "75")),
            chunk_overlap_cues=int(os.getenv("CHUNK_OVERLAP_CUES", "2")),
        )


settings = Settings.from_env()
