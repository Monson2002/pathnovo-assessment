import hashlib
import json
import os
from typing import Optional
from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter
from src.ingest.dwg import DWGAdapter
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.observability.logging import get_logger

logger = get_logger(__name__)

# Order matters: NativePDFAdapter is attempted first.
# ScannedPDFAdapter acts as fallback for PDFs without digital text layers.
ADAPTERS: list[FormatAdapter] = [
    NativePDFAdapter(),
    DWGAdapter(),
    ScannedPDFAdapter(),
]


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_and_ingest(
    file_path: str,
    pid: str,
    cache_dir: Optional[str] = "output/.cache/ingest",
    use_cache: bool = True,
) -> CanonicalDocument:
    """Auto-detect format and ingest file into canonical representation with SHA-256 caching."""
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        file_hash = compute_file_hash(file_path)
        cache_file = os.path.join(cache_dir, f"{file_hash}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                doc = CanonicalDocument.model_validate(data)
                doc.metadata.pid = pid
                doc.metadata.is_cached = True
                logger.info(
                    f"Loaded cached CanonicalDocument for {file_path} (hash: {file_hash[:8]})"
                )
                return doc
            except Exception as e:
                logger.warning(
                    f"Failed to read cache file {cache_file}, re-ingesting: {e}"
                )

    for adapter in ADAPTERS:
        if adapter.can_handle(file_path):
            doc = adapter.ingest(file_path, pid)
            doc.metadata.is_cached = False
            if use_cache and cache_dir:
                file_hash = compute_file_hash(file_path)
                cache_file = os.path.join(cache_dir, f"{file_hash}.json")
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(doc.model_dump_json(indent=2))
                    logger.info(
                        f"Saved ingested CanonicalDocument to cache {cache_file}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to write cache file {cache_file}: {e}")
            return doc

    raise ValueError(
        f"No suitable adapter found to handle document at path: '{file_path}'"
    )
