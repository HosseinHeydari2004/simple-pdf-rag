from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embedding.embedding import Embedding

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_VECTOR_DB_PATH = (
    _PROJECT_ROOT
    / "vector_database"
    / "chroma_db"
)


class VectorDB:
    def __init__(self):
        _VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        embedding = Embedding()
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embedding.embedding,
            persist_directory=str(_VECTOR_DB_PATH),
        )

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self.vector_store.add_documents(documents=documents)

    def search(self, query: str, k: int = 3) -> list[Document]:
        return self.vector_store.similarity_search(query=query, k=k)

    def count(self) -> int:
        return self.vector_store._collection.count()

    def clear(self) -> None:
        ids = self.vector_store.get()["ids"]
        if ids:
            self.vector_store.delete(ids=ids)
