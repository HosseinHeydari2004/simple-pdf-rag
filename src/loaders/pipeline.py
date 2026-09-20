from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document


class PipelineLoader:
    def __init__(self):
        self.__loaders: dict = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }

    def load(self, file_path: str | Path) -> list[Document]:
        file = Path(file_path)
        extension = file.suffix.lower()
        loader_class = self.__loaders.get(extension)
        if loader_class is None:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )
        return loader_class(
            file_path=str(file)
        ).load()
