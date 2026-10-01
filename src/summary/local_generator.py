"""Local HuggingFace causal LM summarizer running on Apple Silicon MPS."""

import logging
from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.summary.base import BaseSearchSummarizer
from src.summary.prompt import build_chat_messages, build_summary_prompt

logger = logging.getLogger("mrl.summary.local")


class LocalGemmaSummarizer(BaseSearchSummarizer):
    """Local HuggingFace causal LM running on Apple Silicon (MPS) with streaming output."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "mps",
        torch_dtype: str = "float16",
        max_new_tokens: int = 384,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        self._model_name = model_name
        self.device_str = device
        self.dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._tokenizer = None
        self._model = None
        self._load_attempted = False
        self._load_successful = False

    @property
    def name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> bool:
        """Lazy load weights and tokenizer to avoid unnecessary overhead."""
        if self._load_attempted:
            return self._load_successful

        self._load_attempted = True
        logger.info(f"Initializing local causal LM '{self._model_name}' on {self.device_str}...")

        # Resolve device and dtype
        if self.device_str == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        dtype = torch.float16 if self.dtype_str == "float16" else torch.bfloat16

        # 1. Try loading from local HuggingFace cache first
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
                device_map=device,
                local_files_only=True,
            )
            self._model.eval()
            self._load_successful = True
            logger.info(f"Successfully loaded '{self._model_name}' from local cache on {device}.")
            return True
        except Exception:
            # Model not cached locally
            pass

        # 2. Check if network is reachable before attempting download
        try:
            import socket
            socket.create_connection(("huggingface.co", 443), timeout=0.8).close()
            has_net = True
        except Exception:
            has_net = False

        if not has_net:
            logger.info(
                f"Local model '{self._model_name}' is not in local cache and HuggingFace Hub is unreachable. "
                "Skipping to fallback backend."
            )
            self._load_successful = False
            return False

        # 3. Attempt download if network is available
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
                device_map=device,
            )
            self._model.eval()
            self._load_successful = True
            logger.info(f"Successfully downloaded and loaded '{self._model_name}' on {device}.")
            return True
        except Exception as e:
            logger.warning(f"Could not load local model '{self._model_name}': {e}")
            self._load_successful = False
            return False

    def is_available(self) -> bool:
        """Check if local model can be loaded successfully."""
        return self._ensure_loaded()

    def generate(self, query: str, documents: list[dict]) -> str:
        """Generate complete summary."""
        chunks = list(self.stream_generate(query, documents))
        return "".join(chunks).strip()

    def stream_generate(self, query: str, documents: list[dict]) -> Iterator[str]:
        """Stream generated text chunks using TextIteratorStreamer on a background thread."""
        if not self._ensure_loaded():
            raise RuntimeError(f"Local model '{self._model_name}' could not be loaded.")

        # Try chat template formatting if supported by tokenizer
        try:
            messages = build_chat_messages(query, documents)
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = build_summary_prompt(query, documents)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        do_sample = self.temperature > 0.0
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if do_sample else None,
            do_sample=do_sample,
            top_p=self.top_p if do_sample else None,
        )

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()
