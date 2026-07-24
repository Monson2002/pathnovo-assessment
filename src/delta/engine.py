import re
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.canonical.model import BoundingBox, CanonicalDocument, Page
from src.config import settings
from src.delta.align import align_blocks, bbox_iou
from src.observability.logging import get_logger

logger = get_logger(__name__)


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class DeltaItem(BaseModel):
    """A single change detected between revision A and revision B."""

    change_type: ChangeType
    item_type: str  # text | dimension | note | geometry | label
    page: int
    location: BoundingBox
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    confidence: float


class DeltaResult(BaseModel):
    """The structured result of computing deltas between two documents."""

    pid_a: str
    pid_b: str
    items: List[DeltaItem] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)


def classify_item_type(text: str) -> str:
    """Classify the domain role of extracted text in an engineering drawing."""
    if not text:
        return "text"
    text_clean = text.strip()

    # Dimension pattern e.g., 2", 100mm, DN50, 1'-6", 3/4", 150#, 150 LB, Ø12, NPS 4
    if re.search(
        r'^\d+(\.\d+)?\s*(mm|cm|m|"|in|ft|DN|ANSI|LB|#|psi|bar|kPa)?$',
        text_clean,
        re.IGNORECASE,
    ) or re.search(
        r'^\d+[\'-]\d+"?|\d+/\d+"?|Ø\d+|NPS\s*\d+', text_clean, re.IGNORECASE
    ):
        return "dimension"

    # Equipment/Tag/Piping Line Label pattern e.g., P-101A, V-202, FT-1001, HV-001, FIC-1001A, PSV-100A/B, 10"-P-1001-CS
    if re.search(
        r"^[A-Z0-9]{1,6}-?[A-Z0-9]{1,6}(-[A-Z0-9]{1,6})*(/[A-Z0-9]+)?$", text_clean
    ) and any(c.isdigit() for c in text_clean):
        return "label"

    # Note pattern e.g., NOTE: ..., REMARKS:, REVISION:
    if re.search(
        r"^(NOTE|REMARK|REF|DWG|DRAWING|REVISION)\b", text_clean, re.IGNORECASE
    ):
        return "note"

    return "text"


def combine_bboxes(b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
    """Return a bounding box that encompasses both b1 and b2."""
    return BoundingBox(
        x0=min(b1.x0, b2.x0),
        y0=min(b1.y0, b2.y0),
        x1=max(b1.x1, b2.x1),
        y1=max(b1.y1, b2.y1),
    )


class DeltaEngine:
    """Engine for comparing two CanonicalDocuments and producing a DeltaResult."""

    def __init__(
        self,
        text_similarity_threshold: Optional[float] = None,
        spatial_weight: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.text_threshold = (
            text_similarity_threshold
            if text_similarity_threshold is not None
            else settings.text_similarity_threshold
        )
        self.spatial_weight = (
            spatial_weight if spatial_weight is not None else settings.spatial_weight
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )

    def _compare_drawings(
        self, page_num: int, page_a: Page, page_b: Page
    ) -> List[DeltaItem]:
        """Compare significant vector drawing elements between revision A and B for a page."""
        items: List[DeltaItem] = []
        # Filter for significant geometric elements (avoid micro-line segments)
        drawings_a = [
            d
            for d in page_a.drawing_elements
            if d.bbox.width >= 0.02 or d.bbox.height >= 0.02
        ]
        drawings_b = [
            d
            for d in page_b.drawing_elements
            if d.bbox.width >= 0.02 or d.bbox.height >= 0.02
        ]

        if not drawings_a and not drawings_b:
            return items

        used_b = set()
        for idx_a, elem_a in enumerate(drawings_a):
            best_match = None
            best_iou = 0.0
            for idx_b, elem_b in enumerate(drawings_b):
                if idx_b in used_b:
                    continue
                iou = bbox_iou(elem_a.bbox, elem_b.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = idx_b

            if best_match is not None and best_iou >= 0.6:
                used_b.add(best_match)
            else:
                items.append(
                    DeltaItem(
                        change_type=ChangeType.REMOVED,
                        item_type="geometry",
                        page=page_num,
                        location=elem_a.bbox,
                        description=f"Removed {elem_a.element_type} geometric shape on page {page_num}",
                        old_value=elem_a.element_type,
                        new_value=None,
                        confidence=0.80,
                    )
                )

        for idx_b, elem_b in enumerate(drawings_b):
            if idx_b not in used_b:
                items.append(
                    DeltaItem(
                        change_type=ChangeType.ADDED,
                        item_type="geometry",
                        page=page_num,
                        location=elem_b.bbox,
                        description=f"Added {elem_b.element_type} geometric shape on page {page_num}",
                        old_value=None,
                        new_value=elem_b.element_type,
                        confidence=0.80,
                    )
                )

        return items

    def compute_delta(
        self, doc_a: CanonicalDocument, doc_b: CanonicalDocument
    ) -> DeltaResult:
        """Compute structured changes between canonical document A and B."""
        logger.info(
            f"Computing delta between PID A ({doc_a.metadata.pid}) and PID B ({doc_b.metadata.pid})"
        )
        items: List[DeltaItem] = []
        max_pages = max(len(doc_a.pages), len(doc_b.pages))

        for page_idx in range(max_pages):
            page_num = page_idx + 1
            page_a = doc_a.pages[page_idx] if page_idx < len(doc_a.pages) else None
            page_b = doc_b.pages[page_idx] if page_idx < len(doc_b.pages) else None

            if page_a and not page_b:
                # Entire page removed
                for block in page_a.text_blocks:
                    items.append(
                        DeltaItem(
                            change_type=ChangeType.REMOVED,
                            item_type=classify_item_type(block.content),
                            page=page_num,
                            location=block.bbox,
                            description=f"Removed text on page {page_num}: '{block.content}'",
                            old_value=block.content,
                            new_value=None,
                            confidence=block.confidence,
                        )
                    )
                continue

            if page_b and not page_a:
                # Entire page added
                for block in page_b.text_blocks:
                    items.append(
                        DeltaItem(
                            change_type=ChangeType.ADDED,
                            item_type=classify_item_type(block.content),
                            page=page_num,
                            location=block.bbox,
                            description=f"Added text on page {page_num}: '{block.content}'",
                            old_value=None,
                            new_value=block.content,
                            confidence=block.confidence,
                        )
                    )
                continue

            if not page_a or not page_b:
                continue

            # 1. Compare text blocks
            matched, unmatched_a, unmatched_b = align_blocks(
                page_a.text_blocks,
                page_b.text_blocks,
                text_threshold=self.text_threshold,
                spatial_weight=self.spatial_weight,
            )

            # Process matched blocks
            for block_a, block_b, score in matched:
                if block_a.content != block_b.content:
                    # Content modified
                    item_type = classify_item_type(block_b.content)
                    items.append(
                        DeltaItem(
                            change_type=ChangeType.MODIFIED,
                            item_type=item_type,
                            page=page_num,
                            location=combine_bboxes(block_a.bbox, block_b.bbox),
                            description=(
                                f"Modified {item_type} on page {page_num}: "
                                f"'{block_a.content}' -> '{block_b.content}'"
                            ),
                            old_value=block_a.content,
                            new_value=block_b.content,
                            confidence=round(
                                min(block_a.confidence, block_b.confidence, score), 2
                            ),
                        )
                    )

            # Process removals (unmatched in A)
            for block_a in unmatched_a:
                item_type = classify_item_type(block_a.content)
                items.append(
                    DeltaItem(
                        change_type=ChangeType.REMOVED,
                        item_type=item_type,
                        page=page_num,
                        location=block_a.bbox,
                        description=f"Removed {item_type} on page {page_num}: '{block_a.content}'",
                        old_value=block_a.content,
                        new_value=None,
                        confidence=round(block_a.confidence, 2),
                    )
                )

            # Process additions (unmatched in B)
            for block_b in unmatched_b:
                item_type = classify_item_type(block_b.content)
                items.append(
                    DeltaItem(
                        change_type=ChangeType.ADDED,
                        item_type=item_type,
                        page=page_num,
                        location=block_b.bbox,
                        description=f"Added {item_type} on page {page_num}: '{block_b.content}'",
                        old_value=None,
                        new_value=block_b.content,
                        confidence=round(block_b.confidence, 2),
                    )
                )

            # 2. Compare drawing elements (geometry)
            drawing_deltas = self._compare_drawings(page_num, page_a, page_b)
            items.extend(drawing_deltas)

        # Filter items below confidence threshold if configured
        filtered_items = [
            item for item in items if item.confidence >= self.confidence_threshold
        ]

        summary = {
            "added": sum(
                1 for item in filtered_items if item.change_type == ChangeType.ADDED
            ),
            "removed": sum(
                1 for item in filtered_items if item.change_type == ChangeType.REMOVED
            ),
            "modified": sum(
                1 for item in filtered_items if item.change_type == ChangeType.MODIFIED
            ),
            "total_changes": len(filtered_items),
        }

        logger.info(
            f"Delta computation complete: {summary['total_changes']} changes detected "
            f"({summary['added']} added, {summary['removed']} removed, {summary['modified']} modified)"
        )

        return DeltaResult(
            pid_a=doc_a.metadata.pid,
            pid_b=doc_b.metadata.pid,
            items=filtered_items,
            summary=summary,
        )
