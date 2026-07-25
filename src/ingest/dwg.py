import os
from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter


class DWGAdapter(FormatAdapter):
    """Stub adapter for DWG / DXF CAD files.

    The seam is real and fully defined — plug in ezdxf or ODA File Converter
    behind this interface when ready.
    """

    def can_handle(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        return file_path.lower().endswith((".dwg", ".dxf"))

    def ingest(self, file_path: str, pid: str) -> CanonicalDocument:
        raise NotImplementedError(
            f"DWG/DXF ingestion is stubbed for '{file_path}'. "
            "To enable DWG support, integrate ezdxf or ODA File Converter "
            "behind the DWGAdapter seam."
        )
