import os

from .base import LLMProvider
from langchain.chat_models import init_chat_model

class GeminiProvider(LLMProvider):
    """Primary provider — Google Gemini API."""

    def __init__(self, model: str = "gemini-3.5-flash", api_key: str | None = None):
        from google import genai

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file "
                "or pass api_key explicitly."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return (response.text or "").strip()


class OpenRouterProvider(LLMProvider):
    """Fallback / alternative provider — any model available on OpenRouter,
    accessed through the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str = "auto",
        api_key: str | None = None,
    ):

        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file "
                "or pass api_key explicitly."
            )
        self._client = init_chat_model(
            model=model,
            model_provider="openrouter",
        )
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()


_PROVIDERS = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """Factory: build an `LLMProvider` from a name (or the LLM_PROVIDER
    env var, defaulting to Gemini)."""
    name = (name or os.getenv("LLM_PROVIDER", "gemini")).lower()
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown LLM provider '{name}'. Choose one of: {list(_PROVIDERS)}"
        )
    return provider_cls(**kwargs)
