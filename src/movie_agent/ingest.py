"""Ingestion CLI: parse .srt files -> clean -> chunk -> embed -> store."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .catalog import Catalog
from .config import settings
from .models import MovieInfo
from .srt_parser import chunk_cues, parse_srt, parse_title, read_srt_file, slugify


def ingest_file(path: Path, index, catalog: Catalog, *, force: bool = False) -> MovieInfo | None:
    data = path.read_bytes()
    file_hash = hashlib.sha1(data).hexdigest()
    source = path.name
    existing = catalog.by_source(source)
    if existing and existing.file_hash == file_hash and not force:
        print(f"  = unchanged, skipped: {source}")
        return existing
    if existing:  # file changed: replace its chunks
        index.delete_movie(existing.movie_id)
        catalog.remove(existing.movie_id)

    cues = parse_srt(read_srt_file(path))
    if not cues:
        print(f"  ! no usable cues, skipped: {source}")
        return None

    title, year = parse_title(source)
    movie_id = slugify(title, year)
    clash = catalog.get(movie_id)
    if clash and clash.source_file != source:  # two files, same title
        movie_id = f"{movie_id}-{file_hash[:6]}"

    chunks = chunk_cues(
        cues, movie_id=movie_id, movie_title=title, year=year, source_file=source,
        max_chars=settings.chunk_max_chars, max_seconds=settings.chunk_max_seconds,
        overlap_cues=settings.chunk_overlap_cues,
    )
    index.add_chunks(chunks)
    info = MovieInfo(movie_id, title, year, source, file_hash, len(cues), len(chunks), cues[-1].end_ms)
    catalog.upsert(info)
    catalog.save()
    print(f"  + {info.label}: {len(cues)} cues -> {len(chunks)} chunks")
    return info


def ingest_directory(root: Path, index, catalog: Catalog, *, limit: int, reset: bool, force: bool) -> int:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".srt")
    if not files:
        print(f"No .srt files found under {root}")
        return 0
    if len(files) > limit:
        print(f"Found {len(files)} files; ingesting the first {limit} (MAX_MOVIES).")
        files = files[:limit]
    if reset:
        index.reset()
        catalog.clear()
        catalog.save()
    done = sum(1 for f in files if ingest_file(f, index, catalog, force=force))
    print(f"\nIndexed {done}/{len(files)} files; {index.count()} chunks in the vector store.")
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest .srt subtitle files into the vector store.")
    ap.add_argument("--subtitles", type=Path, default=settings.subtitles_dir)
    ap.add_argument("--limit", type=int, default=settings.max_movies)
    ap.add_argument("--reset", action="store_true", help="wipe the index and catalog first")
    ap.add_argument("--force", action="store_true", help="re-ingest even if the file is unchanged")
    args = ap.parse_args(argv)

    from .vector_store import SubtitleIndex

    index = SubtitleIndex(settings.chroma_dir, settings.embedding_model, settings.min_similarity)
    ingest_directory(args.subtitles, index, Catalog(settings.catalog_path),
                     limit=args.limit, reset=args.reset, force=args.force)


if __name__ == "__main__":
    main()
