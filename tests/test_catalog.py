from movie_agent.catalog import resolve_title
from movie_agent.models import MovieInfo


def mk(title, year=None):
    return MovieInfo(f"{title}-{year}", title, year, f"{title}.srt", "h", 1, 1, 1)


LIB = [mk("Batman Begins", 2005), mk("Batman Returns", 1992), mk("The Dark Knight", 2008),
       mk("The Dark Knight Rises", 2012), mk("Inception", 2010), mk("Blade Runner", 1982),
       mk("Blade Runner", 2017)]


def test_exact_and_article_insensitive():
    assert resolve_title("inception", LIB).status == "exact"
    r = resolve_title("Dark Knight", LIB)
    assert r.status == "exact" and r.matches[0].year == 2008


def test_franchise_is_ambiguous():
    r = resolve_title("Batman", LIB)
    assert r.status == "ambiguous" and len(r.matches) >= 2


def test_year_disambiguates_remakes():
    assert resolve_title("Blade Runner", LIB).status == "ambiguous"
    r = resolve_title("Blade Runner 2017", LIB)
    assert r.status == "exact" and r.matches[0].year == 2017


def test_unknown_title_not_found():
    assert resolve_title("Casablanca", LIB).status == "not_found"
