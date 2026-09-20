from langchain_huggingface import HuggingFaceEmbeddings


class Embedding:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
