import os
from collections import defaultdict
import fitz  # PyMuPDF
from src.delta.engine import ChangeType, DeltaResult
from src.observability.logging import get_logger

logger = get_logger(__name__)


def generate_delta_markup(
    pdf_path: str,
    delta_result: DeltaResult,
    output_path: str = "output/annotated_delta.pdf",
    force_rebuild: bool = False,
) -> str:
    """Overlay visual bounding box annotations and redline highlights onto document PDF."""
    if not os.path.exists(pdf_path):
        logger.error(f"Cannot generate markup: PDF file not found: {pdf_path}")
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Group items by page for batch drawing
    items_by_page = defaultdict(list)
    for idx, item in enumerate(delta_result.items, start=1):
        items_by_page[item.page - 1].append((idx, item))

    logger.info(f"Generating visual delta markup overlay for {pdf_path}")

    with fitz.open(pdf_path) as doc:
        for page_num, page_items in items_by_page.items():
            if page_num < 0 or page_num >= len(doc):
                continue

            page = doc[page_num]
            w, h = page.rect.width, page.rect.height
            shape = page.new_shape()

            for idx, item in page_items:
                # Convert normalized bounding box (0-1) to page coordinates
                bbox = item.location
                rect = fitz.Rect(
                    bbox.x0 * w,
                    bbox.y0 * h,
                    bbox.x1 * w,
                    bbox.y1 * h,
                )

                # Color scheme: Red for REMOVED, Green for ADDED, Yellow/Orange for MODIFIED
                if item.change_type == ChangeType.REMOVED:
                    color = (1.0, 0.0, 0.0)  # Red
                    fill = (1.0, 0.8, 0.8)
                elif item.change_type == ChangeType.ADDED:
                    color = (0.0, 0.6, 0.0)  # Green
                    fill = (0.8, 1.0, 0.8)
                else:
                    color = (0.9, 0.5, 0.0)  # Orange
                    fill = (1.0, 0.9, 0.7)

                # Draw rectangle border and subtle fill highlight
                shape.draw_rect(rect)
                shape.finish(color=color, fill=fill, width=2.0)

                # Insert annotation text badge
                label_text = f"Item #{idx} ({item.change_type.value.upper()})"
                shape.insert_text(
                    fitz.Point(rect.x0, max(12, rect.y0 - 3)),
                    label_text,
                    fontsize=8,
                    color=color,
                )

            shape.commit()

        doc.save(output_path)

    logger.info(f"Saved visual delta markup to {output_path}")
    return output_path
