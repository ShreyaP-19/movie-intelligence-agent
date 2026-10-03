"""SRT parsing, text cleaning, title extraction and time-window chunking."""
from __future__ import annotations

import html
import re
from pathlib import Path

from .models import Chunk, Cue

_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")                 # <i>, <font ...>
_ASS_RE = re.compile(r"\{[^}]*\}")               # {\an8}
_SOUND_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [door slams] (sighs)
_MUSIC_RE = re.compile(r"[♪♫♬]+")
_AD_RE = re.compile(
    r"opensubtitles|subscene|yify|yts\.|addic7ed|sync(?:ed)?(?: and| &) correct"
    r"|downloaded from|advertise your product|www\.\S+|https?://",
    re.I,
)


# --------------------------------------------------------------------------- reading:handles messy real-world files
def read_srt_file(path: str | Path) -> str:
    """Decode an .srt file, tolerating BOMs and legacy encodings."""
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _to_ms(h: str, m: str, s: str, frac: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(frac.ljust(3, "0")[:3])


# --------------------------------------------------------------------------- cleaning: strips html tags, sound descriptions
def clean_text(raw: str, strip_sound_cues: bool = True) -> str:
    """Remove markup, sound descriptions, music notes and dialogue dashes."""
    t = _TAG_RE.sub("", raw)
    t = _ASS_RE.sub("", t)
    t = html.unescape(t)
    if strip_sound_cues:
        t = _SOUND_RE.sub(" ", t)
    t = _MUSIC_RE.sub(" ", t)
    lines = []
    for line in t.splitlines():
        line = re.sub(r"^\s*[-–—]+\s*", "", line).strip()
        if line:
            lines.append(line)
    t = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return t if re.search(r"[A-Za-z0-9]", t) else ""


#finds the timecode line with a regex
def parse_srt(raw: str, strip_sound_cues: bool = True) -> list[Cue]:
    """Parse SRT text into cleaned, time-ordered cues."""
    raw = raw.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"\n[ \t]+\n", "\n\n", raw)
    cues: list[Cue] = []
    counter = 0
    for block in re.split(r"\n{2,}", raw.strip()):
        lines = block.split("\n")
        k = next((i for i, ln in enumerate(lines) if _TIME_RE.search(ln)), None)
        if k is None:
            continue
        m = _TIME_RE.search(lines[k])
        start = _to_ms(*m.group(1, 2, 3, 4))
        end = max(start, _to_ms(*m.group(5, 6, 7, 8)))
        counter += 1
        index = int(lines[k - 1].strip()) if k > 0 and lines[k - 1].strip().isdigit() else counter
        text = clean_text("\n".join(lines[k + 1:]), strip_sound_cues)
        if not text or _AD_RE.search(text):
            continue
        if cues and cues[-1].text == text and start - cues[-1].end_ms <= 1000:
            prev = cues[-1]  # merge consecutive duplicates
            cues[-1] = Cue(prev.index, prev.start_ms, max(prev.end_ms, end), text)
            continue
        cues.append(Cue(index, start, end, text))
    cues.sort(key=lambda c: c.start_ms)
    return cues


# --------------------------------------------------------------------------- metadata
_RELEASE_TAGS = re.compile(
    r"\b(480p|576p|720p|1080p|2160p|4k|uhd|bluray|blu ray|bdrip|brrip|dvdrip|webrip|web dl|"
    r"webdl|hdrip|hdtv|hdcam|x264|x265|h264|h265|hevc|xvid|yify|yts|rarbg|ac3|aac|dts|"
    r"remastered|extended|unrated|directors cut|eng|english)\b",
    re.I,
)


#turns a filename into a title and year
def parse_title(filename: str) -> tuple[str, int | None]:
    """'The.Dark.Knight.2008.1080p.BluRay.srt' / 'Inception (2010).srt' -> (title, year)."""
    stem = re.sub(r"[._]+", " ", Path(filename).stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    year: int | None = None
    m = re.search(r"[(\[]((?:19|20)\d{2})[)\]]", stem)
    if m:
        year, stem = int(m.group(1)), stem[: m.start()]
    else:
        tag = _RELEASE_TAGS.search(stem)
        if tag and tag.start() > 0:
            stem = stem[: tag.start()]
        m = re.search(r"\s((?:19|20)\d{2})\s*$", stem)
        if m and m.start() > 0:
            year, stem = int(m.group(1)), stem[: m.start()]
    title = re.sub(r"\s+", " ", re.sub(r"[-\s]+$", "", stem)).strip()
    if title.islower() or title.isupper():
        title = title.title()
    return title or Path(filename).stem, year


def slugify(title: str, year: int | None = None) -> str:
    base = f"{title} {year}" if year else title
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


# --------------------------------------------------------------------------- chunking: Single cues are too short to embed meaningfully, so chunk_cues groups consecutive cues into windows. A window stops growing at about 600 characters or 75 seconds. The next window starts 2 cues back, so a line near a boundary appears in two chunks and isn't cut off from its context

def chunk_cues(
    cues: list[Cue],
    *,
    movie_id: str,
    movie_title: str,
    year: int | None,
    source_file: str,
    max_chars: int = 600,
    max_seconds: int = 75,
    overlap_cues: int = 2,
) -> list[Chunk]:
    """Group consecutive cues into overlapping windows bounded by size and duration.

    Each chunk keeps the exact start/end timecode of its first/last cue, so every
    retrieved passage can be cited with a precise range.
    """
    chunks: list[Chunk] = []
    i, n, prev_j = 0, len(cues), -1
    while i < n:
        j, chars = i, 0
        while j < n:
            add = len(cues[j].text) + 1
            if j > i and (
                chars + add > max_chars or cues[j].end_ms - cues[i].start_ms > max_seconds * 1000
            ):
                break
            chars += add
            j += 1
        if j == prev_j:  # overlap window adds no new cues (e.g. after a long silence)
            i = j
            continue
        prev_j = j
        group = cues[i:j]
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{movie_id}:{idx:05d}",
                movie_id=movie_id,
                movie_title=movie_title,
                year=year,
                chunk_index=idx,
                start_ms=group[0].start_ms,
                end_ms=group[-1].end_ms,
                text=" ".join(c.text for c in group),
                source_file=source_file,
                first_cue=group[0].index,
                last_cue=group[-1].index,
            )
        )
        if j >= n:
            break
        i = max(j - overlap_cues, i + 1)
    return chunks
