"""Delta Engine module for document revision comparison."""

from src.delta.align import align_blocks, bbox_iou, text_similarity
from src.delta.engine import ChangeType, DeltaEngine, DeltaItem, DeltaResult
from src.delta.report import (
    generate_delta_report,
    generate_json_report,
    generate_markdown_report,
)

__all__ = [
    "bbox_iou",
    "text_similarity",
    "align_blocks",
    "ChangeType",
    "DeltaItem",
    "DeltaResult",
    "DeltaEngine",
    "generate_markdown_report",
    "generate_json_report",
    "generate_delta_report",
]
