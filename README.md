<div align="center">

# 📚 simple pdf rag — Simple PDF/Docx/Txt RAG

**Turn a folder of documents into a chat assistant that only answers from what you gave it.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-6A4C93)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## ✨ What is this?

**DocuMind** is a lightweight Retrieval-Augmented Generation (RAG) app. Upload your
PDFs, Word documents, or text/markdown files, and ask questions about them in a clean
chat interface. Every answer is **grounded in your documents** — if the answer isn't
in there, DocuMind says so instead of making something up.

| | |
|---|---|
| 📄 **Ingests** | PDF · DOCX · TXT · Markdown |
| ✂️ **Chunking** | Recursive character splitting, tuned for semantic coherence |
| 🧠 **Embeddings** | Local, free — `BAAI/bge-small-en-v1.5` (via Hugging Face) |
| 🗄️ **Vector store** | [ChromaDB](https://www.trychroma.com/), persisted to disk |
| 🤖 **LLM** | Provider-agnostic — **Gemini** (primary) or **OpenRouter** (fallback) |
| 💬 **UI** | A polished Streamlit chat app with live source citations |
| 🐳 **Deployment** | One command with Docker Compose |

---

## 🖼️ How it works

```
 ┌──────────────┐     ┌───────────┐     ┌────────────┐     ┌───────────────┐
 │  Your files   │ ──▶ │  Chunker  │ ──▶ │  Embedder  │ ──▶ │  Chroma (DB)  │
 │ pdf/docx/txt  │     │ recursive │     │  bge-small │     │   on disk     │
 └──────────────┘     └───────────┘     └────────────┘     └───────┬───────┘
                                                                     │
 ┌──────────────┐     ┌────────────────┐     ┌───────────┐          │
 │   Your chat   │ ◀── │ LLM (Gemini /  │ ◀── │ Retrieval │ ◀────────┘
 │   question    │     │  OpenRouter)   │     │  (top-k)  │
 └──────────────┘     └────────────────┘     └───────────┘
```

The LLM only ever receives the question **plus the retrieved chunks** — never the
whole document set — which is what keeps answers grounded and fast.

---

## 🚀 Quick start (Docker — recommended)

This is the easiest way to run DocuMind; you don't need Python installed locally.

```bash
# 1. Clone the project and step into it
git clone <your-repo-url> documind && cd documind

# 2. Set up your API key
cp .env.example .env
# then open .env and paste your GOOGLE_API_KEY (or OPENROUTER_API_KEY)

# 3. Build & run
docker compose up --build
```

Open **http://localhost:8501** in your browser — you're live. 🎉

Your indexed documents persist across restarts because `vector_database/` is
mounted as a volume, so `docker compose up` (without `down -v`) will remember
everything you already ingested.

To stop the app:

```bash
docker compose down
```

<details>
<summary>Prefer plain <code>docker</code> instead of Compose?</summary>

```bash
docker build -t documind-rag .
docker run -p 8501:8501 --env-file .env -v $(pwd)/vector_database:/app/vector_database documind-rag
```

</details>

---

## 🧑‍💻 Quick start (without Docker)

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and add your key

# 4. Launch the app
streamlit run app/streamlit_app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 🔑 Getting an API key

DocuMind talks to the LLM through a provider-agnostic interface, so you can use
either of these (Gemini is the default):

| Provider | Where to get a key | Env variable |
|---|---|---|
| **Gemini** (default) | [Google AI Studio](https://aistudio.google.com/apikey) — has a free tier | `GOOGLE_API_KEY` |
| **OpenRouter** (fallback) | [openrouter.ai/keys](https://openrouter.ai/keys) — access to many models with one key | `OPENROUTER_API_KEY` |

You can either put the key in `.env` before starting the app, **or** paste it
directly into the sidebar once the app is running — whichever is easier for a
live demo.

---

## 📖 How to use the app

1. **Open the app** in your browser (`http://localhost:8501`).
2. In the **sidebar**, pick your LLM provider and make sure your API key is filled in.
3. **Upload one or more files** (PDF, DOCX, TXT, or Markdown) using the file picker.
4. Click **"➕ Add to KB"** — DocuMind will chunk, embed, and store them. A progress
   bar shows the ingestion status, and a badge appears for every indexed file.
5. Type a question in the **chat box** at the bottom, or click one of the suggested
   example questions to get started.
6. Read the answer, then expand **"📎 sources used"** under it to see exactly which
   passages (and which file/page) the answer came from.
7. Use **"🗑️ Clear KB"** to wipe the knowledge base and start over, or
   **"🧽 Clear chat"** to reset just the conversation.

If a question can't be answered from your documents, DocuMind will tell you it
doesn't have enough information instead of guessing — that's by design.

---

## 📂 Project structure

```
.
├── app/
│   └── streamlit_app.py     # Chat UI
├── src/
│   ├── loaders/              # PDF / DOCX / TXT / MD loaders
│   ├── chunk/                # Recursive text chunker
│   ├── embedding/             # HuggingFace (bge-small) embeddings
│   ├── vector_db/            # Chroma vector store wrapper
│   ├── llm/                   # Provider-agnostic LLM interface (Gemini / OpenRouter)
│   └── rag_pipeline.py       # Orchestrates ingest + retrieval + generation
├── vector_database/           # Persisted Chroma DB (created at runtime)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
└── requirements.txt
```

---

## 🛠️ Tech stack

- **Python 3.12**
- **[Streamlit](https://streamlit.io/)** for the UI
- **[LangChain](https://www.langchain.com/)** for document loading & chunking
- **[ChromaDB](https://www.trychroma.com/)** for vector storage
- **`sentence-transformers` / `BAAI/bge-small-en-v1.5`** for local, free embeddings
- **Google Gemini API** and **OpenRouter** for generation, behind a common interface

---

## 📝 License

MIT — do whatever you'd like with it.
