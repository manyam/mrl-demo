"""Abstract base class for generative search summarizers."""

from abc import ABC, abstractmethod
from typing import Iterator


class BaseSearchSummarizer(ABC):
    """Interface for search summarization backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this summarizer."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or path of the underlying model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is ready and reachable."""
        pass

    @abstractmethod
    def generate(self, query: str, documents: list[dict]) -> str:
        """Generate a complete search summary for query and top documents."""
        pass

    @abstractmethod
    def stream_generate(self, query: str, documents: list[dict]) -> Iterator[str]:
        """Stream chunks/tokens of the search summary as they are produced."""
        pass
