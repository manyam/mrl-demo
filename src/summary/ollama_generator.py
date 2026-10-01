"""Ollama client for local Gemma model search summarization."""

import json
import logging
from typing import Iterator

import httpx

from src.summary.base import BaseSearchSummarizer
from src.summary.prompt import build_summary_prompt

logger = logging.getLogger("mrl.summary.ollama")


class OllamaGemmaSummarizer(BaseSearchSummarizer):
    """Generative summarizer using local Ollama daemon hosting Gemma models."""

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        timeout: float = 30.0,
    ):
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check if requested model or any variant is pulled
                    has_model = any(self._model_name in m for m in models)
                    if not has_model:
                        logger.warning(
                            f"Ollama is running, but model '{self._model_name}' is not in installed list: {models}. "
                            f"Run `ollama run {self._model_name}` to pull it."
                        )
                    return True
        except Exception as e:
            logger.debug(f"Ollama daemon not reachable at {self.base_url}: {e}")
        return False

    def generate(self, query: str, documents: list[dict]) -> str:
        """Generate complete summary via Ollama non-streaming endpoint."""
        prompt = build_summary_prompt(query, documents)
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.base_url}/api/generate", json=payload)
            res.raise_for_status()
            data = res.json()
            return data.get("response", "").strip()

    def stream_generate(self, query: str, documents: list[dict]) -> Iterator[str]:
        """Stream summary response token-by-token from Ollama."""
        prompt = build_summary_prompt(query, documents)
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
