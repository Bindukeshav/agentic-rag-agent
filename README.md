# Agentic RAG Assistant

**An agent that checks its own homework before answering.**

Most RAG chatbots retrieve documents and answer blindly — even when the retrieval is bad. This one doesn't. It grades its own retrieved context for relevance before trusting it, and falls back to a live web search the moment its own knowledge base comes up short — instead of confidently hallucinating an answer from irrelevant chunks.

## Why this isn't just another RAG demo

- **Self-correcting retrieval** — an LLM-based grading node checks every retrieved document before it's allowed anywhere near the final answer. Bad retrieval doesn't get to fail silently.
- **Graceful fallback, not a guess** — if local documents fail the relevance check, the agent automatically pivots to a live web search rather than answering from junk context.
- **Follow-ups that actually follow** — a query-rewriting step turns vague, conversational questions ("what was the score in that final?") into standalone ones using recent chat history, before they ever hit search.
- **Real persistence** — conversations survive restarts via SQLite-backed checkpointing, not an in-memory list that vanishes when the process stops.

## How it works

```
User question
      │
      ▼
Contextualize ──► rewrites vague follow-ups using chat history
      │
      ▼
 Retrieve ──► vector search over ingested documents (ChromaDB)
      │
      ▼
  Grade ──► LLM checks: are these documents actually relevant?
      │
   ┌──┴───┐
 relevant  not relevant
   │         │
   │         ▼
   │    Web search ──► live fallback via Tavily
   │         │
   └────┬────┘
        ▼
    Generate ──► answers using whatever context earned its place
        │
        ▼
   Answer delivered
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** | State machine with conditional routing — the only way to cleanly express "check the work, then decide" |
| LLM | **Groq (Llama 3.3 70B)** | Fast inference, used for grading, rewriting, and generation |
| Vector store | **ChromaDB** | Local, zero-setup similarity search |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Free, local — no API cost for indexing |
| Web fallback | **Tavily** | Live search when local knowledge isn't enough |
| Backend | **FastAPI** | REST API with interactive docs out of the box |
| Persistence | **SQLite** (via LangGraph checkpointing) | Conversations survive restarts |

## Setup

```bash
git clone https://github.com/Bindukeshav/agentic-rag-agent.git
cd agentic-rag-agent
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

## Usage

```bash
# Index a document into the vector store
python test_ingestion.py

# Start the API
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## Endpoints

| Method | Path | Does |
|---|---|---|
| `POST` | `/ingest` | Upload and index a new document |
| `POST` | `/chat` | Ask a question, with thread-based memory |
| `GET` | `/history/{thread_id}` | Retrieve a conversation's last state |

## Known limitations

- Only the most recent 6 turns are used when rewriting follow-up questions
- Documents are graded one at a time, not batched — retrieval quality checks add latency as document count grows

## What this project demonstrates

Agent architecture with real decision-making (not just chained prompts), self-correction as a design pattern, conversational context handling, and a production-shaped backend — checkpointed state, a documented REST API, and a clear separation between orchestration (LangGraph) and tooling (LangChain components).
