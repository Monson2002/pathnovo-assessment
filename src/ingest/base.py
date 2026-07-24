from abc import ABC, abstractmethod
from src.canonical.model import CanonicalDocument


class FormatAdapter(ABC):
    """Abstract Base Class for document format adapters."""

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Return True if this adapter can process the given file."""
        pass

    @abstractmethod
    def ingest(self, file_path: str, pid: str) -> CanonicalDocument:
        """Convert a file into the canonical representation."""
        pass
