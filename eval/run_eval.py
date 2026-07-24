import json
import os
import time
from typing import Any, Dict
from eval.metrics import (
    citation_accuracy,
    delta_precision_recall_f1,
    groundedness_score,
)
from src.chat import AnswerEngine, DocumentIndex
from src.delta import DeltaEngine, generate_markdown_report
from src.ingest import detect_and_ingest
from src.observability.logging import get_logger

logger = get_logger(__name__)


def run_evaluation(
    dataset_path: str = "eval/datasets/pair_01_expected.json",
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Run full evaluation suite on a document pair dataset, save JSON results, and print scorecard."""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        print(f"Dataset file not found: {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pdf_a_path = data["pid_a"]
    pdf_b_path = data["pid_b"]

    if not os.path.exists(pdf_a_path) or not os.path.exists(pdf_b_path):
        logger.error(f"Sample PDF files missing: {pdf_a_path} or {pdf_b_path}")
        print(f"Sample PDF files missing: {pdf_a_path} or {pdf_b_path}")
        return {}

    logger.info(f"Running evaluation harness for dataset: {dataset_path}")
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
    qa_results = []
    for qa in qa_pairs:
        question = qa["question"]
        res = chat_engine.answer_question(question)
        g_score = groundedness_score(res.answer, res.retrieved_chunks)
        c_score = citation_accuracy(res.citations, qa.get("expected_sources", []))
        groundedness_scores.append(g_score)
        citation_scores.append(c_score)
        qa_results.append(
            {
                "question": question,
                "answer": res.answer,
                "groundedness": g_score,
                "citation_accuracy": c_score,
                "citations": [c.model_dump() for c in res.citations],
            }
        )

    avg_groundedness = (
        sum(groundedness_scores) / len(groundedness_scores)
        if groundedness_scores
        else 1.0
    )
    avg_citation = (
        sum(citation_scores) / len(citation_scores) if citation_scores else 1.0
    )

    results = {
        "timestamp": time.time(),
        "dataset": dataset_path,
        "pid_a": pdf_a_path,
        "pid_b": pdf_b_path,
        "delta_metrics": delta_metrics,
        "chat_metrics": {
            "groundedness": round(avg_groundedness, 2),
            "citation_accuracy": round(avg_citation, 2),
        },
        "qa_details": qa_results,
        "known_limitations": [
            "OCR confidence variations on scanned title blocks",
            "Geometric vector path overlap heuristic thresholds",
            "Multi-line label bounding box fragmentation",
        ],
    }

    # Save JSON scorecard output
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "eval_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation scorecard written to {out_file}")

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
    print("║   - Bounding box fragmentation on long text      ║")
    print(f"║ Saved to: {out_file:<39}║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n")

    return results


if __name__ == "__main__":
    run_evaluation()
