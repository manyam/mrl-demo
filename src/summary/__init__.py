"""Generative search summarization package with multi-backend fallback support."""

import logging
from typing import Iterator

from src.summary.base import BaseSearchSummarizer
from src.summary.extractive_generator import ExtractiveSummarizer
from src.summary.google_generator import GoogleGeminiSummarizer
from src.summary.local_generator import LocalGemmaSummarizer
from src.summary.ollama_generator import OllamaGemmaSummarizer
from src.utils.config import ConfigDict, load_config

logger = logging.getLogger("mrl.summary")


class FallbackChainSummarizer(BaseSearchSummarizer):
    """Wraps an ordered chain of summarizers with automatic graceful failover."""

    def __init__(self, backends: list[BaseSearchSummarizer]):
        self.backends = [b for b in backends if b is not None]
        if not self.backends:
            self.backends = [ExtractiveSummarizer()]
        self._last_active_backend: BaseSearchSummarizer | None = None

    @property
    def active_backend(self) -> BaseSearchSummarizer:
        """Return the currently active or primary available backend in the chain."""
        if self._last_active_backend is not None:
            return self._last_active_backend
        for b in self.backends:
            if b.is_available():
                return b
        return self.backends[-1]

    @property
    def name(self) -> str:
        return f"chain({self.active_backend.name})"

    @property
    def model_name(self) -> str:
        return self.active_backend.model_name

    def is_available(self) -> bool:
        return any(b.is_available() for b in self.backends)

    def generate(self, query: str, documents: list[dict]) -> str:
        for idx, backend in enumerate(self.backends):
            try:
                if backend.is_available():
                    logger.info(f"Generating summary using backend '{backend.name}' ({backend.model_name})...")
                    res = backend.generate(query, documents)
                    self._last_active_backend = backend
                    return res
            except Exception as e:
                logger.warning(f"Backend '{backend.name}' failed during generation: {e}. Trying fallback...")
        fallback = ExtractiveSummarizer()
        self._last_active_backend = fallback
        return fallback.generate(query, documents)

    def stream_generate(self, query: str, documents: list[dict]) -> Iterator[str]:
        for idx, backend in enumerate(self.backends):
            try:
                if backend.is_available():
                    logger.info(f"Streaming summary using backend '{backend.name}' ({backend.model_name})...")
                    self._last_active_backend = backend
                    yield from backend.stream_generate(query, documents)
                    return
            except Exception as e:
                logger.warning(f"Backend '{backend.name}' failed during streaming: {e}. Trying fallback...")
        fallback = ExtractiveSummarizer()
        self._last_active_backend = fallback
        yield from fallback.stream_generate(query, documents)


def create_summarizer(
    config: ConfigDict | None = None,
    backend_override: str | None = None,
) -> BaseSearchSummarizer:
    """Factory creating summarizer according to YAML configuration and fallback hierarchy."""
    if config is None:
        config = load_config()

    summary_cfg = getattr(config, "summary", {})
    backend = (backend_override or summary_cfg.get("default_backend", "auto")).lower()

    # Extract configs
    local_cfg = summary_cfg.get("local", {})
    ollama_cfg = summary_cfg.get("ollama", {})
    google_cfg = summary_cfg.get("google", {})

    temp = summary_cfg.get("temperature", 0.3)
    max_tokens = summary_cfg.get("max_new_tokens", 384)
    top_p = summary_cfg.get("top_p", 0.9)

    # Instantiate individual backends
    def get_ollama():
        return OllamaGemmaSummarizer(
            model_name=ollama_cfg.get("model_name", "gemma4:26b-mlx"),
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            temperature=temp,
            timeout=ollama_cfg.get("timeout", 30.0),
        )

    def get_google():
        return GoogleGeminiSummarizer(
            model_name=google_cfg.get("model_name", "gemini-2.0-flash"),
            temperature=temp,
            max_output_tokens=max_tokens,
            timeout=google_cfg.get("timeout", 20.0),
        )

    def get_local():
        return LocalGemmaSummarizer(
            model_name=local_cfg.get("model_name", "google/gemma-2-2b-it"),
            device=local_cfg.get("device", "mps"),
            torch_dtype=local_cfg.get("torch_dtype", "float16"),
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
        )

    def get_extractive():
        return ExtractiveSummarizer()

    if backend == "ollama":
        return get_ollama()
    elif backend == "google":
        return get_google()
    elif backend == "local":
        return get_local()
    elif backend == "extractive":
        return get_extractive()
    elif backend == "auto":
        # Fallback hierarchy: 1. Ollama -> 2. Google API -> 3. Local MPS Gemma -> 4. Extractive
        chain = [
            get_ollama(),
            get_google(),
            get_local(),
            get_extractive(),
        ]
        return FallbackChainSummarizer(chain)
    else:
        logger.warning(f"Unknown backend '{backend}'; falling back to auto chain.")
        return FallbackChainSummarizer([get_ollama(), get_google(), get_local(), get_extractive()])


__all__ = [
    "BaseSearchSummarizer",
    "ExtractiveSummarizer",
    "OllamaGemmaSummarizer",
    "GoogleGeminiSummarizer",
    "LocalGemmaSummarizer",
    "FallbackChainSummarizer",
    "create_summarizer",
]
