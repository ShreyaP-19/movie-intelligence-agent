"""Chroma-backed hybrid (dense + BM25) retrieval over subtitle chunks."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .models import Chunk, Hit, ms_to_clock

COLLECTION = "subtitles"
_STOP = set(
    "a an the and or but if of to in on at by for with from is are was were be been am do does did "
    "i you he she it we they me him her us them my your his its our their this that these those "
    "what who whom which when where why how say said says tell about movie film scene".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP and len(t) > 1]


class SubtitleIndex:
    def __init__(self, persist_dir: str | Path, model_name: str, min_similarity: float = 0.25):
        import chromadb

        self.model_name = model_name
        self.min_similarity = min_similarity
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self._model = None
        self._lex = None

    # ------------------------------------------------------------------ embeddings
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()

    # ------------------------------------------------------------------ writes
    def add_chunks(self, chunks: list[Chunk], batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            # embed with the title prefix so title-flavoured queries land on the right movie
            vectors = self._embed([f"{c.movie_label}: {c.text}" for c in part])
            self._col.add(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=vectors,
                metadatas=[
                    {
                        "movie_id": c.movie_id,
                        "movie_title": c.movie_title,
                        "movie_label": c.movie_label,
                        "year": c.year or 0,
                        "chunk_index": c.chunk_index,
                        "start_ms": c.start_ms,
                        "end_ms": c.end_ms,
                        "start_ts": ms_to_clock(c.start_ms),
                        "end_ts": ms_to_clock(c.end_ms),
                        "first_cue": c.first_cue,
                        "last_cue": c.last_cue,
                        "source_file": c.source_file,
                    }
                    for c in part
                ],
            )
        self._lex = None

    def delete_movie(self, movie_id: str) -> None:
        self._col.delete(where={"movie_id": movie_id})
        self._lex = None

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._col = self._client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self._lex = None

    def count(self) -> int:
        return self._col.count()

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _where(movie_ids: list[str] | None):
        if not movie_ids:
            return None
        return {"movie_id": movie_ids[0]} if len(movie_ids) == 1 else {"movie_id": {"$in": list(movie_ids)}}

    def _build_lexical(self):
        from rank_bm25 import BM25Okapi

        data = self._col.get(include=["documents", "metadatas"])
        docs, metas = data["documents"], data["metadatas"]
        self._lex = {
            "ids": data["ids"],
            "docs": docs,
            "metas": metas,
            "movie": np.array([m["movie_id"] for m in metas]),
            "bm25": BM25Okapi([tokenize(d) or ["_"] for d in docs]),
        }

    def _lexical(self, query: str, movie_ids: list[str] | None, n: int):
        if self._lex is None:
            self._build_lexical()
        toks = tokenize(query)
        if not toks:
            return []
        scores = np.asarray(self._lex["bm25"].get_scores(toks), dtype=float)
        if movie_ids:
            scores = np.where(np.isin(self._lex["movie"], movie_ids), scores, -1.0)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 8, movie_ids: list[str] | None = None) -> list[Hit]:
        """Hybrid retrieval: cosine search + BM25, fused with reciprocal rank fusion."""
        total = self._col.count()
        if total == 0:
            return []
        n = min(max(k * 4, 20), total)

        qv = self._embed([query])[0]
        res = self._col.query(
            query_embeddings=[qv], n_results=n, where=self._where(movie_ids),
            include=["documents", "metadatas", "distances"],
        )
        cands: dict[str, dict] = {}
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            cands[cid] = {"doc": doc, "meta": meta, "sim": 1.0 - float(dist), "score": 1.0 / (60 + rank)}

        if self._lex is None:
            self._build_lexical()
        for rank, i in enumerate(self._lexical(query, movie_ids, n)):
            cid = self._lex["ids"][i]
            entry = cands.setdefault(
                cid, {"doc": self._lex["docs"][i], "meta": self._lex["metas"][i], "sim": None, "score": 0.0}
            )
            entry["score"] += 1.0 / (60 + rank)
            entry["lexical"] = True

        hits = []
        for cid, e in cands.items():
            # dense-only candidates must clear the similarity floor; lexical matches are kept
            if not e.get("lexical") and (e["sim"] is None or e["sim"] < self.min_similarity):
                continue
            m = e["meta"]
            hits.append(
                Hit(cid, m["movie_id"], m["movie_label"], int(m["start_ms"]), int(m["end_ms"]),
                    e["doc"], e["sim"], e["score"])
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
