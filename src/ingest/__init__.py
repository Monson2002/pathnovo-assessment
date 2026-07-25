"""Format adapters and ingestion registry."""

from src.ingest.base import FormatAdapter
from src.ingest.dwg import DWGAdapter
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.ingest.registry import detect_and_ingest

__all__ = [
    "FormatAdapter",
    "NativePDFAdapter",
    "ScannedPDFAdapter",
    "DWGAdapter",
    "detect_and_ingest",
]
