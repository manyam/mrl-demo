"""Google Generative AI (Gemini) client for search summarization."""

import json
import logging
import os
from typing import Iterator

import httpx

from src.summary.base import BaseSearchSummarizer
from src.summary.prompt import build_summary_prompt

logger = logging.getLogger("mrl.summary.google")


class GoogleGeminiSummarizer(BaseSearchSummarizer):
    """Generative summarizer using Google Generative AI REST API with streaming SSE."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 512,
        timeout: float = 25.0,
    ):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def name(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Check if an API key is provided."""
        return bool(self.api_key and self.api_key.strip())

    def _build_payload(self, query: str, documents: list[dict]) -> dict:
        prompt = build_summary_prompt(query, documents)
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

    def generate(self, query: str, documents: list[dict]) -> str:
        """Generate complete summary via standard Google generateContent endpoint."""
        if not self.is_available():
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

        url = f"{self.base_url}/{self._model_name}:generateContent?key={self.api_key}"
        payload = self._build_payload(query, documents)

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts).strip()
            return ""

    def stream_generate(self, query: str, documents: list[dict]) -> Iterator[str]:
        """Stream summary tokens via streamGenerateContent with Server-Sent Events."""
        if not self.is_available():
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

        url = f"{self.base_url}/{self._model_name}:streamGenerateContent?alt=sse&key={self.api_key}"
        payload = self._build_payload(query, documents)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        raw_json = line[6:].strip()
                        try:
                            chunk = json.loads(raw_json)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    text = p.get("text", "")
                                    if text:
                                        yield text
                        except json.JSONDecodeError:
                            continue
