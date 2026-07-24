from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter
from src.ingest.dwg import DWGAdapter
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter

# Order matters: NativePDFAdapter is attempted first.
# ScannedPDFAdapter acts as fallback for PDFs without digital text layers.
ADAPTERS: list[FormatAdapter] = [
    NativePDFAdapter(),
    DWGAdapter(),
    ScannedPDFAdapter(),
]


def detect_and_ingest(file_path: str, pid: str) -> CanonicalDocument:
    """Auto-detect format and ingest file into canonical representation."""
    for adapter in ADAPTERS:
        if adapter.can_handle(file_path):
            return adapter.ingest(file_path, pid)

    raise ValueError(
        f"No suitable adapter found to handle document at path: '{file_path}'"
    )
