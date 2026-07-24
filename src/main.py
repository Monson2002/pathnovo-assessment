import argparse
import os
from src.chat import AnswerEngine, DocumentIndex
from src.delta import DeltaEngine, generate_delta_report
from src.ingest import detect_and_ingest
from src.observability import Tracer, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Document Delta Computation & Grounded Chat CLI"
    )
    parser.add_argument(
        "--pid-a",
        type=str,
        required=True,
        help="Path to first P&ID PDF document (Revision A)",
    )
    parser.add_argument(
        "--pid-b",
        type=str,
        required=True,
        help="Path to second P&ID PDF document (Revision B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save delta reports and execution trace files",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch interactive Grounded Chat CLI after computing delta report",
    )

    args = parser.parse_args()
    setup_logging()

    tracer = Tracer(trace_name="document_delta_pipeline")

    print("\n--- Document Delta & Grounded Chat ---")
    print(f"Document A: {args.pid_a}")
    print(f"Document B: {args.pid_b}\n")

    with tracer.span("ingest_documents"):
        print("Ingesting Document A...")
        doc_a = detect_and_ingest(args.pid_a, pid=os.path.basename(args.pid_a))
        print(
            f"  Ingested {len(doc_a.pages)} page(s) via adapter '{doc_a.metadata.format}'"
        )

        print("Ingesting Document B...")
        doc_b = detect_and_ingest(args.pid_b, pid=os.path.basename(args.pid_b))
        print(
            f"  Ingested {len(doc_b.pages)} page(s) via adapter '{doc_b.metadata.format}'"
        )

    with tracer.span("compute_delta"):
        print("\nComputing delta engine analysis...")
        engine = DeltaEngine()
        delta_res = engine.compute_delta(doc_a, doc_b)
        print(f"  Delta Summary: {delta_res.summary}")

    with tracer.span("generate_report"):
        md_content, json_content = generate_delta_report(
            delta_res, output_dir=args.output_dir
        )
        print(f"  Saved Delta Reports to: {args.output_dir}/")

    trace_path = tracer.finish(output_dir=os.path.join(args.output_dir, "traces"))
    print(f"  Saved Observability Trace: {trace_path}\n")

    if args.chat:
        print("\n=== Building Grounded Chat Index ===")
        index = DocumentIndex(persist_dir=os.path.join(args.output_dir, ".chroma"))
        index.build_index(doc_a, doc_b, md_content)
        chat_engine = AnswerEngine(index=index)

        print("\n--- Interactive Grounded Chat (type 'exit' or 'quit' to stop) ---")
        while True:
            try:
                question = input(
                    "\nAsk a question about the documents/deltas: "
                ).strip()
                if not question or question.lower() in ["exit", "quit"]:
                    print("Exiting chat mode. Goodbye!")
                    break

                res = chat_engine.answer_question(question)
                print(f"\nAnswer:\n{res.answer}\n")
                if res.citations:
                    print("Citations:")
                    for cit in res.citations:
                        print(f"  - {cit.snippet}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat mode. Goodbye!")
                break


if __name__ == "__main__":
    main()
