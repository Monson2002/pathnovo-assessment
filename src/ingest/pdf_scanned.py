import os
import numpy as np
from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    Page,
    TextBlock,
)
from src.ingest.base import FormatAdapter


class ScannedPDFAdapter(FormatAdapter):
    """Adapter for scanned / raster PDFs using OCR (Tesseract via pdf2image)."""

    def can_handle(self, file_path: str) -> bool:
        if not file_path.lower().endswith(".pdf"):
            return False
        return os.path.exists(file_path)

    def ingest(self, file_path: str, pid: str) -> CanonicalDocument:
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from pytesseract import Output
        except ImportError as e:
            raise ImportError(
                "pdf2image and pytesseract are required for ScannedPDFAdapter. "
                "Ensure they are installed."
            ) from e

        images = convert_from_path(file_path, dpi=300)
        pages = []

        for i, img in enumerate(images):
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

                bbox = BoundingBox(
                    x0=max(0.0, min(1.0, left / w)),
                    y0=max(0.0, min(1.0, top / h)),
                    x1=max(0.0, min(1.0, (left + width) / w)),
                    y1=max(0.0, min(1.0, (top + height) / h)),
                )

                text_blocks.append(
                    TextBlock(
                        content=text,
                        bbox=bbox,
                        confidence=round(conf / 100.0, 2),
                    )
                )

            pages.append(
                Page(
                    page_number=i + 1,
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
