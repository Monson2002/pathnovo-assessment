from eval.metrics import (
    citation_accuracy,
    delta_precision_recall_f1,
    groundedness_score,
)
from src.canonical.model import BoundingBox
from src.delta.engine import ChangeType, DeltaItem


def test_delta_precision_recall_f1():
    predicted = [
        DeltaItem(
            change_type=ChangeType.MODIFIED,
            item_type="label",
            page=1,
            location=BoundingBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2),
            description="Modified title",
            old_value="EXPORT GAS COMPRESSOR",
            new_value="LIFT GAS COMPRESSOR",
            confidence=1.0,
        )
    ]
    expected = [
        {
            "change_type": "modified",
            "page": 1,
            "old_value": "EXPORT GAS COMPRESSOR",
            "new_value": "LIFT GAS COMPRESSOR",
            "description": "Title change",
        }
    ]

    metrics = delta_precision_recall_f1(predicted, expected)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_groundedness_score():
    answer = "The title was modified from Export Gas Compressor to Lift Gas Compressor. [PID A, Page 1]"
    retrieved_chunks = [
        "PID A Page 1: Export Gas Compressor C-101 title",
        "PID B Page 1: Lift Gas Compressor C-101 title",
    ]
    score = groundedness_score(answer, retrieved_chunks)
    assert score >= 0.5


def test_citation_accuracy():
    class DummyCitation:
        def __init__(self, source):
            self.source = source

    predicted = [DummyCitation(source="pid_a"), DummyCitation(source="delta_report")]
    expected = ["pid_a", "delta_report"]

    score = citation_accuracy(predicted, expected)
    assert score == 1.0
