from typing import Any, Dict, List
from src.delta.align import text_similarity
from src.delta.engine import DeltaItem


def delta_precision_recall_f1(
    predicted: List[DeltaItem], expected: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute Precision, Recall, and F1 score for detected deltas vs ground truth."""
    if not expected:
        return {"precision": 1.0 if not predicted else 0.0, "recall": 1.0, "f1": 1.0}

    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = 0
    matched_expected = set()

    for item in predicted:
        for idx, exp in enumerate(expected):
            if idx in matched_expected:
                continue

            # Check change_type match
            type_match = (
                item.change_type.value.lower() == exp.get("change_type", "").lower()
            )
            page_match = item.page == exp.get("page", 1)

            old_sim = text_similarity(item.old_value or "", exp.get("old_value") or "")
            new_sim = text_similarity(item.new_value or "", exp.get("new_value") or "")
            desc_sim = text_similarity(item.description, exp.get("description") or "")

            if (
                type_match
                and page_match
                and (old_sim >= 0.5 or new_sim >= 0.5 or desc_sim >= 0.5)
            ):
                tp += 1
                matched_expected.add(idx)
                break

    fp = len(predicted) - tp
    fn = len(expected) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
    }


def groundedness_score(answer: str, retrieved_chunks: List[str]) -> float:
    """Evaluate if the answer content is grounded in the retrieved chunks (0.0 to 1.0)."""
    if not answer or "cannot answer" in answer.lower():
        return 1.0

    if not retrieved_chunks:
        return 0.0

    combined_context = " ".join(retrieved_chunks).lower()

    # Split answer into sentences/phrases
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    grounded_count = 0
    stop_words = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "were",
        "where",
        "what",
    }

    for sentence in sentences:
        # Ignore citation tags in sentence when checking grounding
        cleaned = (
            sentence.replace("[PID A", "")
            .replace("[PID B", "")
            .replace("[Delta Report", "")
        )
        words = [
            w.strip(".,!?:;\"'()[]")
            for w in cleaned.lower().split()
            if len(w) > 3 and w not in stop_words
        ]
        if not words:
            grounded_count += 1
            continue

        matches = sum(1 for w in words if w in combined_context)
        if (matches / len(words)) >= 0.3:
            grounded_count += 1

    return round(grounded_count / len(sentences), 2)


def citation_accuracy(
    predicted_citations: List[Any], expected_sources: List[str]
) -> float:
    """Evaluate citation accuracy against expected sources (0.0 to 1.0)."""
    if not expected_sources:
        return 1.0

    if not predicted_citations:
        return 0.0

    predicted_sources = set()
    for cit in predicted_citations:
        if isinstance(cit, dict):
            predicted_sources.add(cit.get("source", "").lower())
        elif hasattr(cit, "source"):
            predicted_sources.add(cit.source.lower())

    matched = sum(1 for exp in expected_sources if exp.lower() in predicted_sources)
    return round(matched / len(expected_sources), 2)
