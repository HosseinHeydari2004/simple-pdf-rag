from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Provider-agnostic interface for text generation.

    Any concrete LLM backend (Gemini, OpenRouter, ...) must implement
    `generate`. The RAG service only ever talks to this interface, never
    to a specific vendor SDK, so swapping providers never touches
    `rag_pipeline.py`.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the model and return the raw text answer."""
        raise NotImplementedError
