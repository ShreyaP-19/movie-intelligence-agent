"""Routing/guard tests with a fake LLM, fake index and fake MCP mailer."""
from movie_agent.agent import MovieAgent
from movie_agent.catalog import Catalog
from movie_agent.models import Hit, MovieInfo
from movie_agent.rag import RAGEngine, finalize
from movie_agent.router import Router


class FakeLLM:
    def __init__(self, decision):
        self.decision = decision

    def structured(self, *, name, **kw):
        if name == "route":
            return self.decision
        return {"subject": "Lighthouse scene", "body": "The lamp must keep turning [S1]. Bogus [S9]."}

    def complete(self, **kw):
        return "Marta tells Tomas to keep the lamp turning [S1]. Invented [S7]."


class FakeIndex:
    def search(self, query, k=8, movie_ids=None):
        return [Hit("c1", "lh-2021", "The Lighthouse Keeper (2021)", 344000, 348000,
                    "Keep the lamp turning, Tomas.", 0.7, 0.03)]


def make_agent(decision, mails):
    cat = Catalog()
    for t, y in [("The Lighthouse Keeper", 2021), ("Batman Begins", 2005), ("Batman Returns", 1992)]:
        cat.upsert(MovieInfo(f"{t}-{y}", t, y, f"{t}.srt", "h", 1, 1, 1))
    llm = FakeLLM(decision)

    def mailer(**kw):
        mails.append(kw)
        return {"status": "dry_run", "to": [kw["to"]], "detail": "x"}

    return MovieAgent(cat, RAGEngine(FakeIndex(), llm), Router(llm), mailer)


def test_informational_routes_to_rag_with_valid_citations_only():
    agent = make_agent({"intent": "informational", "rewritten_query": "lamp", "movie_titles": [],
                        "reasoning": "q"}, [])
    r = agent.handle("What does Marta say about the lamp?")
    assert r.route == "informational" and "[S1]" in r.reply and "[S7]" not in r.reply
    assert "The Lighthouse Keeper (2021) | 00:05:44 – 00:05:48" in r.reply


def test_email_without_recipient_asks_and_sends_nothing():
    mails = []
    agent = make_agent({"intent": "email_action", "rewritten_query": "lamp", "movie_titles": [],
                        "reasoning": "r"}, mails)
    r = agent.handle("Email me the lighthouse scene")
    assert r.route == "needs_clarification" and "email address" in r.reply and not mails


def test_email_with_recipient_calls_mcp_tool():
    mails = []
    agent = make_agent({"intent": "email_action", "rewritten_query": "lamp", "movie_titles": ["Lighthouse Keeper"],
                        "recipient_email": "bob@example.com", "email_goal": "scene breakdown",
                        "reasoning": "r"}, mails)
    r = agent.handle("Send a scene breakdown to bob@example.com")
    assert r.route == "email_action" and mails[0]["to"] == "bob@example.com"
    assert "[S9]" not in mails[0]["body"] and "Sources:" in mails[0]["body"]


def test_recipient_regex_fallback_when_router_misses_it():
    mails = []
    agent = make_agent({"intent": "email_action", "rewritten_query": "lamp", "movie_titles": [],
                        "reasoning": "r"}, mails)
    agent.handle("email the lamp scene to amy@site.org")
    assert mails and mails[0]["to"] == "amy@site.org"


def test_ambiguous_title_triggers_clarification():
    agent = make_agent({"intent": "informational", "rewritten_query": "plot", "movie_titles": ["Batman"],
                        "reasoning": "r"}, [])
    r = agent.handle("Summarise Batman")
    assert r.route == "needs_clarification" and "Batman Begins" in r.reply


def test_unknown_movie_triggers_clarification():
    agent = make_agent({"intent": "informational", "rewritten_query": "plot", "movie_titles": ["Casablanca"],
                        "reasoning": "r"}, [])
    assert "couldn't find" in agent.handle("plot of Casablanca").reply


def test_finalize_falls_back_to_all_hits_when_uncited():
    hit = Hit("c", "m", "M (2000)", 0, 1000, "hi", 0.5, 0.1)
    text, cites = finalize("No markers here.", [hit])
    assert cites[0].tag == "S1" and "M (2000) | 00:00:00 – 00:00:01" in text
