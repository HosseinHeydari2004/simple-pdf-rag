from pathlib import Path

from langchain_core.documents import Document

from .chunk.recursive import RecursiveChunker
from .loaders.pipeline import PipelineLoader
from .llm import LLMProvider, get_provider
from .vector_db.chroma import VectorDB

_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the provided context.
If the answer is not in the context, say exactly: "I don't have enough information to answer."
Keep the answer concise and, where useful, mention which source number(s) you relied on, e.g. [1].

Context:
{context}

Question:
{question}

Answer:
"""

_NO_CONTEXT_ANSWER = "I don't have enough information to answer."


class RAGPipeline:
    """End-to-end pipeline: load -> chunk -> embed/store -> retrieve -> answer."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._loader = PipelineLoader()
        self._chunker = RecursiveChunker()
        self._vectordb = VectorDB()
        self._llm = llm_provider or get_provider()

    # ---- Ingestion -----------------------------------------------------

    def ingest(self, file_path: str | Path) -> int:
        """Load a single file, chunk it, and store it in the vector DB.
        Returns the number of chunks that were added."""
        docs: list[Document] = self._loader.load(file_path=file_path)
        chunks = self._chunker.chunk(documents=docs)
        self._vectordb.add_documents(documents=chunks)
        return len(chunks)

    def ingest_many(self, file_paths: list[str | Path]) -> int:
        """Ingest several files. Returns the total number of chunks added."""
        total = 0
        for file_path in file_paths:
            total += self.ingest(file_path)
        return total

    def document_count(self) -> int:
        return self._vectordb.count()

    def reset(self) -> None:
        """Wipe the vector store (used by the UI's 'clear knowledge base')."""
        self._vectordb.clear()

    # ---- Querying --------------------------------------------------------

    def query(self, question: str, k: int = 4) -> dict:
        """Retrieve relevant chunks and ask the LLM to answer, grounded only
        in that context. Returns {"answer": str, "sources": list[dict]}."""
        results = self._vectordb.search(query=question, k=k)

        if not results:
            return {"answer": _NO_CONTEXT_ANSWER, "sources": []}

        context = "\n\n".join(
            f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(results)
        )
        prompt = _SYSTEM_PROMPT.format(context=context, question=question)
        answer = self._llm.generate(prompt)

        sources = [
            {
                "index": i + 1,
                "text": doc.page_content,
                "source": Path(doc.metadata.get("source", "unknown")).name,
                "page": doc.metadata.get("page"),
            }
            for i, doc in enumerate(results)
        ]
        return {"answer": answer, "sources": sources}
