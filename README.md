# 🎬 Movie Intelligence & Follow-up Assistant

An agent that answers questions about movies from their `.srt` subtitles (RAG, with
timestamped citations) and can email scene breakdowns / quote analyses through an
**MCP email tool**. Ambiguous requests trigger a clarifying question instead of a guess.

## Architecture

```
 .srt files ──► srt_parser ──► chunks (title, start/end timecode, text)
                                   │  embed (MiniLM) + metadata
                                   ▼
                              ChromaDB  +  BM25   ◄── hybrid search (RRF)
                                                          ▲
 user ─► Streamlit ─► Agent ─► Router (LLM, forced JSON)  │
                       │         ├─ informational ────────┤──► answer + [S#] citations
                       │         ├─ email_action ─────────┘──► draft ─► MCP client ─► MCP server `send_email`
                       │         └─ needs_clarification ───────► ask the user
                       └─ guards: recipient e-mail check · fuzzy movie-title resolution
```

| Module | Responsibility |
|---|---|
| `srt_parser.py` | tolerant SRT parsing (BOM/encodings, `,`/`.` ms), cleaning (HTML/ASS tags, sound cues, music, ads), title/year from filename, time-window chunking with overlap |
| `ingest.py` | CLI: incremental ingestion (file hash), up to 100 movies, `--reset` / `--force` |
| `vector_store.py` | Chroma (cosine) + BM25, reciprocal-rank fusion, similarity floor |
| `rag.py` | context building, grounded generation, **citation validation** |
| `llm.py` | provider-agnostic LLM client: Gemini (default), Groq, OpenRouter, Ollama, Anthropic; JSON-mode fallback; automatic fallback provider |
| `router.py` | LLM intent classification (informational / email_action / needs_clarification) |
| `agent.py` | orchestration + deterministic guards |
| `catalog.py` | movie catalog + fuzzy title resolution (exact / ambiguous / not found) |
| `mcp_email_server.py` / `mcp_client.py` | standard MCP stdio server exposing `send_email`, and the client the agent uses |
| `app.py` | Streamlit chat UI (shows route, trace, sources, MCP result) |

### Design decisions
* **Chunking**: windows of consecutive cues (≤600 chars / ≤75 s, 2-cue overlap). Each chunk keeps the
  exact start of its first cue and end of its last cue → precise citation ranges. Long silences split chunks.
* **Hybrid retrieval**: dense vectors catch paraphrase/plot questions; BM25 catches exact quotes and names.
* **Citations can't be invented**: the model cites `[S1]`, `[S2]`…; `finalize()` drops markers that don't
  map to a retrieved passage and appends `Movie (Year) | HH:MM:SS – HH:MM:SS` for each one used.
* **Routing**: the router LLM returns schema-constrained JSON (intent, standalone query, titles,
  recipient, goal). Code then validates it: invalid/missing recipient → ask; unknown or ambiguous title → ask.
* **MCP**: `send_email` is a real MCP tool (FastMCP, stdio). With no `SMTP_HOST` it writes `.eml` files to
  `./outbox` (dry-run) so it can be demoed without credentials.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GEMINI_API_KEY (free) and optionally SMTP_*

# put up to 100 .srt files in data/subtitles/  (naming: "Title (Year).srt")
movie-ingest --subtitles data/subtitles                # try: --subtitles data/sample  for the demo file
streamlit run app.py
```

Inspect the MCP server on its own: `npx @modelcontextprotocol/inspector python src/movie_agent/mcp_email_server.py`

## Choosing a (free) LLM

Set `LLM_PROVIDER` in `.env`:

| Provider | Key | Notes |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` from Google AI Studio | free tier; limits are per project, check AI Studio. Free-tier prompts may be used by Google to improve products |
| `groq` | `GROQ_API_KEY` | fast; tight token-per-minute caps on free tier |
| `openrouter` | `OPENROUTER_API_KEY` | `:free` models; low daily request cap |
| `ollama` | none | local/unlimited; use a 7–8B+ model, e.g. `ollama pull qwen2.5:7b` |
| `anthropic` | `ANTHROPIC_API_KEY` | paid; `pip install anthropic` |

Add `LLM_FALLBACK_PROVIDER=groq` (plus its key) so a rate limit on the primary doesn't break a demo.
Free-tier model ids change; if you see "model not found", set `LLM_MODEL`.
Structured output uses a forced function call and falls back to JSON mode for models without tool support.

## Example conversations

| User | Route | Behaviour |
|---|---|---|
| “What does Marta tell Tomas about the lamp?” | informational | answer with `[S1]` + `Sources: The Lighthouse Keeper (2021) \| 00:05:44 – 00:05:48` |
| “Summarise the storm scene and email it to amy@example.com” | email_action | RAG → draft → MCP `send_email` |
| “Email me that scene” | needs_clarification | “Who should I send this to? …” |
| “Summarise Batman” (several Batman films) | needs_clarification | lists the candidates |

## Tests

```bash
pytest -q        # parser, chunker, title resolver, routing/guards (fake LLM + fake mailer; no network)
```

## Config (`.env`)
`LLM_PROVIDER`, `LLM_MODEL`, `LLM_FALLBACK_PROVIDER`, `LLM_BASE_URL`, `EMBEDDING_MODEL`, `TOP_K`, `MIN_SIMILARITY`, `MAX_MOVIES`, `CHUNK_*`, `SMTP_*`, `EMAIL_DRY_RUN`.