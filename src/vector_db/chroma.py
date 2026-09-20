from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.embedding.embedding import Embedding


class VectorDB:
    def __init__(self):
        embedding = Embedding()
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embedding.embedding,
            persist_directory = "vector_database/vector_store"
        )

    def add_documents(self, documents: list[Document]):
        return self.vector_store.add_documents(documents=documents)
    def
