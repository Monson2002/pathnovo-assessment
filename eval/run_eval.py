import json
import os
from eval.metrics import (
    citation_accuracy,
    delta_precision_recall_f1,
    groundedness_score,
)
from src.chat import AnswerEngine, DocumentIndex
from src.delta import DeltaEngine, generate_markdown_report
from src.ingest import detect_and_ingest


def run_evaluation(dataset_path: str = "eval/datasets/pair_01_expected.json"):
    """Run full evaluation suite on a document pair dataset and output scorecard."""
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_a_path = data["pid_a"]
    pdf_b_path = data["pid_b"]

    if not os.path.exists(pdf_a_path) or not os.path.exists(pdf_b_path):
        print(f"Sample PDF files missing: {pdf_a_path} or {pdf_b_path}")
        return

    print("1. Ingesting documents...")
    doc_a = detect_and_ingest(pdf_a_path, pid="PID_A")
    doc_b = detect_and_ingest(pdf_b_path, pid="PID_B")

    print("2. Computing deltas...")
    engine = DeltaEngine()
    delta_res = engine.compute_delta(doc_a, doc_b)
    md_report = generate_markdown_report(delta_res)

    print("3. Evaluating Delta Engine...")
    delta_metrics = delta_precision_recall_f1(
        delta_res.items, data.get("expected_deltas", [])
    )

    print("4. Indexing for Grounded Chat...")
    index = DocumentIndex(persist_dir=None)
    index.build_index(doc_a, doc_b, md_report)

    chat_engine = AnswerEngine(index=index)
    groundedness_scores = []
    citation_scores = []

    qa_pairs = data.get("qa_pairs", [])
    for qa in qa_pairs:
        question = qa["question"]
        res = chat_engine.answer_question(question)
        g_score = groundedness_score(res.answer, res.retrieved_chunks)
        c_score = citation_accuracy(res.citations, qa.get("expected_sources", []))
        groundedness_scores.append(g_score)
        citation_scores.append(c_score)

    avg_groundedness = (
        sum(groundedness_scores) / len(groundedness_scores)
        if groundedness_scores
        else 1.0
    )
    avg_citation = (
        sum(citation_scores) / len(citation_scores) if citation_scores else 1.0
    )

    print("\n")
    print("╔══════════════════════════════════════════════════╗")
    print("║           EVALUATION SCORECARD                   ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║ Delta Detection                                  ║")
    print(f"║   Precision:  {delta_metrics['precision']:<35.2f}║")
    print(f"║   Recall:     {delta_metrics['recall']:<35.2f}║")
    print(f"║   F1:         {delta_metrics['f1']:<35.2f}║")
    print("║                                                  ║")
    print("║ Chat Quality                                     ║")
    print(f"║   Groundedness:         {avg_groundedness:<25.2f}║")
    print(f"║   Citation Accuracy:    {avg_citation:<25.2f}║")
    print("║                                                  ║")
    print("║ Known Limitations                                ║")
    print("║   - OCR confidence on scanned title blocks       ║")
    print("║   - Vector graphics bounding overlap heuristic   ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n")


if __name__ == "__main__":
    run_evaluation()
