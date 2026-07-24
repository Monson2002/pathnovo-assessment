import os
import fitz  # PyMuPDF
from src.canonical.model import (
    BoundingBox,
    CanonicalDocument,
    DocumentMetadata,
    DrawingElement,
    Page,
    TextBlock,
)
from src.ingest.base import FormatAdapter


class NativePDFAdapter(FormatAdapter):
    """Adapter for native digital PDFs with extractable text and vector graphics."""

    def can_handle(self, file_path: str) -> bool:
        if not file_path.lower().endswith(".pdf"):
            return False
        if not os.path.exists(file_path):
            return False
        try:
            doc = fitz.open(file_path)
            has_text = any(page.get_text().strip() for page in doc)
            doc.close()
            return has_text
        except Exception:
            return False

    def ingest(self, file_path: str, pid: str) -> CanonicalDocument:
        doc = fitz.open(file_path)
        pages = []
        for i, fitz_page in enumerate(doc):
            w, h = float(fitz_page.rect.width), float(fitz_page.rect.height)
            w = max(w, 1.0)
            h = max(h, 1.0)

            # Extract text blocks with position & formatting metadata
            text_blocks = []
            page_dict = fitz_page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            sb = span.get("bbox", (0, 0, 0, 0))
                            bbox = BoundingBox(
                                x0=max(0.0, min(1.0, sb[0] / w)),
                                y0=max(0.0, min(1.0, sb[1] / h)),
                                x1=max(0.0, min(1.0, sb[2] / w)),
                                y1=max(0.0, min(1.0, sb[3] / h)),
                            )
                            text_blocks.append(
                                TextBlock(
                                    content=text,
                                    bbox=bbox,
                                    confidence=1.0,
                                    font_size=span.get("size"),
                                    font_name=span.get("font"),
                                )
                            )

            # Extract vector graphics / drawing paths
            drawing_elements = []
            try:
                for drawing in fitz_page.get_drawings():
                    rect = drawing.get("rect")
                    if rect:
                        bbox = BoundingBox(
                            x0=max(0.0, min(1.0, rect[0] / w)),
                            y0=max(0.0, min(1.0, rect[1] / h)),
                            x1=max(0.0, min(1.0, rect[2] / w)),
                            y1=max(0.0, min(1.0, rect[3] / h)),
                        )
                        drawing_elements.append(
                            DrawingElement(
                                element_type=drawing.get("type", "path"),
                                bbox=bbox,
                                properties={
                                    "color": drawing.get("color"),
                                    "fill": drawing.get("fill"),
                                    "width": drawing.get("width"),
                                },
                            )
                        )
            except Exception:
                pass

            pages.append(
                Page(
                    page_number=i + 1,
                    width=w,
                    height=h,
                    text_blocks=text_blocks,
                    drawing_elements=drawing_elements,
                )
            )

        doc.close()
        return CanonicalDocument(
            metadata=DocumentMetadata(
                pid=pid,
                filename=os.path.basename(file_path),
                format="native_pdf",
                page_count=len(pages),
            ),
            pages=pages,
        )
