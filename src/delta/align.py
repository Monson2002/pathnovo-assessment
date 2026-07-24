import math
from difflib import SequenceMatcher
from typing import List, Tuple
from src.canonical.model import BoundingBox, TextBlock

MIN_TEXT_SIMILARITY: float = 0.25
HIGH_IOU_THRESHOLD: float = 0.5


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over Union of two bounding boxes in normalized coordinates."""
    xi0, yi0 = max(a.x0, b.x0), max(a.y0, b.y0)
    xi1, yi1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter_w = max(0.0, xi1 - xi0)
    inter_h = max(0.0, yi1 - yi0)
    inter_area = inter_w * inter_h

    area_a = a.width * a.height
    area_b = b.width * b.height
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def bbox_center_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Normalized Euclidean distance between centers of two bounding boxes (0 to ~1.414)."""
    ca_x, ca_y = a.center
    cb_x, cb_y = b.center
    return math.sqrt((ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2)


def text_similarity(a: str, b: str) -> float:
    """Sequence similarity ratio between two strings (case-insensitive, stripped)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def calculate_similarity_score(
    block_a: TextBlock,
    block_b: TextBlock,
    spatial_weight: float = 0.3,
) -> float:
    """Calculate composite similarity score balancing text similarity and spatial proximity."""
    text_sim = text_similarity(block_a.content, block_b.content)

    iou = bbox_iou(block_a.bbox, block_b.bbox)
    center_dist = bbox_center_distance(block_a.bbox, block_b.bbox)
    # Convert center distance to proximity score [0, 1] (max distance in unit box is sqrt(2) ~ 1.414)
    center_proximity = max(0.0, 1.0 - (center_dist / 1.414))
    spatial_sim = max(iou, center_proximity)

    text_w = max(0.0, 1.0 - spatial_weight)
    return (text_w * text_sim) + (spatial_weight * spatial_sim)


def align_blocks(
    page_a_blocks: List[TextBlock],
    page_b_blocks: List[TextBlock],
    text_threshold: float = 0.4,
    spatial_weight: float = 0.3,
) -> Tuple[List[Tuple[TextBlock, TextBlock, float]], List[TextBlock], List[TextBlock]]:
    """Align text blocks between revision A and revision B on a single page.

    Returns:
        matched: List of (block_a, block_b, similarity_score)
        unmatched_a: Blocks in page A with no match (removals)
        unmatched_b: Blocks in page B with no match (additions)
    """
    candidates = []
    for idx_a, block_a in enumerate(page_a_blocks):
        for idx_b, block_b in enumerate(page_b_blocks):
            score = calculate_similarity_score(
                block_a, block_b, spatial_weight=spatial_weight
            )
            # Require minimum text similarity or exact spatial overlap to consider matching
            t_sim = text_similarity(block_a.content, block_b.content)
            if score >= text_threshold and (
                t_sim >= MIN_TEXT_SIMILARITY
                or bbox_iou(block_a.bbox, block_b.bbox) > HIGH_IOU_THRESHOLD
            ):
                candidates.append((score, idx_a, idx_b))

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_a = set()
    used_b = set()
    matched = []

    for score, idx_a, idx_b in candidates:
        if idx_a not in used_a and idx_b not in used_b:
            used_a.add(idx_a)
            used_b.add(idx_b)
            matched.append((page_a_blocks[idx_a], page_b_blocks[idx_b], score))

    unmatched_a = [block for i, block in enumerate(page_a_blocks) if i not in used_a]
    unmatched_b = [block for i, block in enumerate(page_b_blocks) if i not in used_b]

    return matched, unmatched_a, unmatched_b
