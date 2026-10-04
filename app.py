"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from movie_agent.agent import build_agent

st.set_page_config(page_title="Movie Intelligence Assistant", page_icon="🎬")
st.title("🎬 Movie Intelligence & Follow-up Assistant")


@st.cache_resource(show_spinner="Loading index and models…")
def load_agent():
    return build_agent()


try:
    agent = load_agent()
except Exception as exc:
    st.error(f"Could not start the agent: {exc}")
    st.stop()

movies = agent.catalog.all()
with st.sidebar:
    st.metric("Movies indexed", len(movies))
    with st.expander("Library"):
        st.write("\n".join(f"- {m.label}" for m in movies) or "Empty — run `movie-ingest`.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Try: “Who says ‘come home before dark’?” · “Summarise the storm scene and "
               "email it to a@b.com”")

if "messages" not in st.session_state:
    st.session_state.messages = []

ROUTE_BADGE = {"informational": "🔎 RAG answer", "email_action": "✉️ RAG + MCP email",
               "needs_clarification": "❓ Clarification", "error": "⚠️ Error"}


def render_meta(meta: dict):
    st.caption(ROUTE_BADGE.get(meta["route"], meta["route"]))
    with st.expander("Agent trace"):
        for step in meta["trace"]:
            st.markdown(f"- {step}")
    if meta["sources"]:
        with st.expander(f"Sources ({len(meta['sources'])})"):
            for c in meta["sources"]:
                st.markdown(f"**[{c.tag}] {c.citation}**  \n> {c.excerpt}")
    if meta["email"]:
        with st.expander("MCP email result"):
            st.json({k: v for k, v in meta["email"].items() if k != "body"})


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            render_meta(m["meta"])

if prompt := st.chat_input("Ask about a movie, or ask me to email an analysis…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = agent.handle(prompt, history)
        st.markdown(result.reply)
        meta = {"route": result.route, "trace": result.trace, "sources": result.sources, "email": result.email}
        render_meta(meta)
    st.session_state.messages.append({"role": "assistant", "content": result.reply, "meta": meta})
