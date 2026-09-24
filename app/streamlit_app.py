import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.llm import get_provider  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="DocuMind — Chat with your documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #0f172a 0%, #111827 100%); }
        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        .doc-badge {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.35rem;
        }
        .hero {
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            background: linear-gradient(120deg, rgba(99,102,241,0.15), rgba(56,189,248,0.10));
            border: 1px solid rgba(148,163,184,0.15);
            margin-bottom: 1rem;
        }
        .source-card {
            border-radius: 10px;
            border: 1px solid rgba(148,163,184,0.18);
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.5rem;
            background: rgba(148,163,184,0.05);
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
EXAMPLE_QUESTIONS = [
    "Summarize the main points of this document",
    "What are the key definitions I should know?",
    "Compare the first two topics covered",
]

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


def get_pipeline(provider_name: str, api_key: str) -> RAGPipeline | None:
    """(Re)build the pipeline only when the provider/key actually changes."""
    key_signature = (provider_name, api_key)
    if st.session_state.get("_pipeline_signature") != key_signature:
        if not api_key:
            return None
        env_var = "GOOGLE_API_KEY" if provider_name == "gemini" else "OPENROUTER_API_KEY"
        os.environ[env_var] = api_key
        try:
            llm = get_provider(provider_name)
            st.session_state.pipeline = RAGPipeline(llm_provider=llm)
            st.session_state._pipeline_signature = key_signature
        except Exception as exc:  # noqa: BLE001
            st.session_state.pipeline = None
            st.session_state._pipeline_error = str(exc)
    return st.session_state.pipeline


# --------------------------------------------------------------------------- #
# Sidebar — setup, knowledge base
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📚 DocuMind")
    st.caption("A local RAG assistant that answers **only** from your documents.")

    st.markdown("### ⚙️ Model")
    provider_name = st.selectbox(
        "LLM provider",
        options=["gemini", "openrouter"],
        format_func=lambda p: "Gemini (Google AI)" if p == "gemini" else "OpenRouter",
        help="Gemini is the primary provider; OpenRouter is the fallback and "
        "gives access to many other models through one API key.",
    )
    default_key = os.getenv("GOOGLE_API_KEY" if provider_name == "gemini" else "OPENROUTER_API_KEY", "")
    api_key = st.text_input(
        "API key",
        value=default_key,
        type="password",
        placeholder="Paste your API key…",
        help="Your key is only kept in this browser session — it is never stored.",
    )

    st.divider()
    st.markdown("### 📥 Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    ingest_col, clear_col = st.columns(2)
    ingest_clicked = ingest_col.button("➕ Add to KB", use_container_width=True, disabled=not uploaded_files)
    clear_clicked = clear_col.button("🗑️ Clear KB", use_container_width=True)

    pipeline = get_pipeline(provider_name, api_key)

    if ingest_clicked:
        if pipeline is None:
            st.error("Add a valid API key above before ingesting documents.")
        else:
            progress = st.progress(0.0, text="Reading documents…")
            total_chunks = 0
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, uploaded_file in enumerate(uploaded_files, start=1):
                    tmp_path = Path(tmp_dir) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getbuffer())
                    total_chunks += pipeline.ingest(tmp_path)
                    st.session_state.ingested_files.append(uploaded_file.name)
                    progress.progress(i / len(uploaded_files), text=f"Indexed {uploaded_file.name}")
            progress.empty()
            st.toast(f"Added {total_chunks} chunks from {len(uploaded_files)} file(s) ✅", icon="✅")

    if clear_clicked and pipeline is not None:
        pipeline.reset()
        st.session_state.ingested_files = []
        st.toast("Knowledge base cleared", icon="🧹")

    if st.session_state.ingested_files:
        st.markdown("**Indexed files:**")
        for name in st.session_state.ingested_files:
            st.markdown(f'<span class="doc-badge">📄 {name}</span>', unsafe_allow_html=True)

    if pipeline is not None:
        try:
            st.caption(f"🔎 {pipeline.document_count()} chunks in the vector store")
        except Exception:  # noqa: BLE001
            pass

    st.divider()
    if st.button("🧽 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main — chat
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h2 style="margin-bottom:0.2rem;">💬 Ask your documents anything</h2>
        <p style="margin-bottom:0; opacity:0.8;">
        Upload files on the left, then ask a question below. Every answer is
        grounded in your documents — if it's not in there, I'll say so instead
        of guessing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source(s) used"):
                for src in message["sources"]:
                    page_info = f" · page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(
                        f'<div class="source-card"><b>[{src["index"]}] {src["source"]}{page_info}</b>'
                        f"<br>{src['text'][:400]}{'…' if len(src['text']) > 400 else ''}</div>",
                        unsafe_allow_html=True,
                    )

user_question = st.chat_input("Ask a question about your documents…")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("user"):
        st.markdown(st.session_state.messages[-1]["content"])

    with st.chat_message("assistant"):
        if pipeline is None:
            answer = "Please add a valid API key in the sidebar first."
            sources = []
            st.warning(answer)
        elif not st.session_state.ingested_files:
            answer = "Please upload at least one document in the sidebar before asking questions."
            sources = []
            st.info(answer)
        else:
            with st.spinner("Thinking…"):
                result = pipeline.query(st.session_state.messages[-1]["content"])
            answer, sources = result["answer"], result["sources"]
            st.markdown(answer)
            if sources:
                with st.expander(f"📎 {len(sources)} source(s) used"):
                    for src in sources:
                        page_info = f" · page {src['page']}" if src.get("page") is not None else ""
                        st.markdown(
                            f'<div class="source-card"><b>[{src["index"]}] {src["source"]}{page_info}</b>'
                            f"<br>{src['text'][:400]}{'…' if len(src['text']) > 400 else ''}</div>",
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
