from pathlib import Path

from movie_agent.models import ms_to_clock
from movie_agent.srt_parser import chunk_cues, clean_text, parse_srt, parse_title, read_srt_file

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "The Lighthouse Keeper (2021).srt"


def test_clean_text_removes_markup_and_dashes():
    assert clean_text("<i>Hello</i> {\\an8}\n- Hi there\n[door slams]") == "Hello Hi there"
    assert clean_text("♪ ♪") == ""


def test_parse_keeps_exact_timecodes_and_drops_junk():
    cues = parse_srt(read_srt_file(SAMPLE))
    assert len(cues) == 12  # music-only and ad cues removed
    assert (cues[0].start_ms, cues[0].end_ms) == (2000, 5200)
    assert cues[1].text == "Then we light the lamp early. Marta, the generator is failing."
    assert all("opensubtitles" not in c.text.lower() for c in cues)


def test_parse_handles_dot_separator_and_missing_index():
    cues = parse_srt("00:01:02.5 --> 00:01:04.25\nHello\n")
    assert (cues[0].start_ms, cues[0].end_ms) == (62500, 64250)


def test_title_parsing():
    assert parse_title("Inception (2010).srt") == ("Inception", 2010)
    assert parse_title("The.Dark.Knight.2008.1080p.BluRay.x264.srt") == ("The Dark Knight", 2008)
    assert parse_title("inception.srt") == ("Inception", None)


def test_chunks_cover_all_cues_with_precise_ranges():
    cues = parse_srt(read_srt_file(SAMPLE))
    chunks = chunk_cues(cues, movie_id="m", movie_title="M", year=2021, source_file="f.srt")
    assert chunks[0].start_ms == cues[0].start_ms and chunks[-1].end_ms == cues[-1].end_ms
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.end_ms > c.start_ms for c in chunks)
    # silence gaps must split chunks: first chunk ends before the 00:02:10 scene
    assert chunks[0].end_ms < 130000
    assert ms_to_clock(chunks[1].start_ms) == "00:02:10"
