import os
import fitz
import numpy as np
from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Page,
    TextBlock,
)
from src.config import settings
from src.ingest.base import FormatAdapter
from src.observability.logging import get_logger

logger = get_logger(__name__)


class ScannedPDFAdapter(FormatAdapter):
    """Adapter for scanned / raster PDFs using OCR (Tesseract via pdf2image)."""

    def can_handle(self, file_path: str) -> bool:
        if not file_path.lower().endswith(".pdf"):
            return False
        return os.path.exists(file_path)

    def ingest(self, file_path: str, pid: str) -> CanonicalDocument:
        logger.info(
            f"Ingesting scanned PDF with streaming page OCR: {file_path} (PID: {pid})"
        )
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from pytesseract import Output
        except ImportError as e:
            logger.error("pdf2image or pytesseract missing for scanned PDF ingestion")
            raise ImportError(
                "pdf2image and pytesseract are required for ScannedPDFAdapter. "
                "Ensure they are installed."
            ) from e

        dpi = getattr(settings, "ocr_dpi", 300)

        # Get total page count cleanly
        try:
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
        except Exception:
            total_pages = 1

        pages = []

        # Process page-by-page streaming to save RAM memory
        for i in range(1, total_pages + 1):
            try:
                images = convert_from_path(
                    file_path, dpi=dpi, first_page=i, last_page=i
                )
                if not images:
                    continue
                img = images[0]
            except Exception as e:
                logger.error(f"Failed to convert page {i} of PDF to image: {e}")
                continue

            w, h = float(img.size[0]), float(img.size[1])
            w = max(w, 1.0)
            h = max(h, 1.0)

            img_np = np.array(img)
            ocr_data = pytesseract.image_to_data(img_np, output_type=Output.DICT)

            text_blocks = []
            n_boxes = len(ocr_data.get("text", []))
            for j in range(n_boxes):
                text = ocr_data["text"][j].strip()
                conf_val = ocr_data["conf"][j]
                try:
                    conf = float(conf_val)
                except (ValueError, TypeError):
                    conf = -1.0

                if not text or conf < 0:
                    continue

                left = float(ocr_data["left"][j])
                top = float(ocr_data["top"][j])
                width = float(ocr_data["width"][j])
                height = float(ocr_data["height"][j])

                x0 = max(0.0, min(1.0, left / w))
                y0 = max(0.0, min(1.0, top / h))
                x1 = max(x0 + 0.0001, min(1.0, (left + width) / w))
                y1 = max(y0 + 0.0001, min(1.0, (top + height) / h))

                bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                normalized_conf = max(0.0, min(1.0, round(conf / 100.0, 2)))

                text_blocks.append(
                    TextBlock(
                        content=text,
                        bbox=bbox,
                        confidence=normalized_conf,
                    )
                )

            pages.append(
                Page(
                    page_number=i,
                    width=w,
                    height=h,
                    text_blocks=text_blocks,
                )
            )

        return CanonicalDocument(
            metadata=DocumentMetadata(
                pid=pid,
                filename=os.path.basename(file_path),
                format="scanned_pdf",
                page_count=len(pages),
            ),
            pages=pages,
        )
