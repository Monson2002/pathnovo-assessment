import argparse
import os
import sys
from src.chat import AnswerEngine, DocumentIndex
from src.delta import DeltaEngine, generate_delta_report
from src.ingest import detect_and_ingest
from src.markup import generate_delta_markup
from src.observability import Tracer, get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
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

    # Input file validation
    if not os.path.exists(args.pid_a):
        logger.error(f"Input file for PID A not found: {args.pid_a}")
        print(f"Error: File for PID A does not exist: {args.pid_a}", file=sys.stderr)
        return 1
    if not os.path.exists(args.pid_b):
        logger.error(f"Input file for PID B not found: {args.pid_b}")
        print(f"Error: File for PID B does not exist: {args.pid_b}", file=sys.stderr)
        return 1

    tracer = Tracer(trace_name="document_delta_pipeline")

    logger.info("Starting Document Delta & Grounded Chat Pipeline")
    print("\n--- Document Delta & Grounded Chat ---")
    print(f"Document A: {args.pid_a}")
    print(f"Document B: {args.pid_b}\n")

    try:
        with tracer.span("ingest_documents"):
            logger.info("Stage 1: Ingesting documents")
            print("Ingesting Document A...")
            doc_a = detect_and_ingest(args.pid_a, pid=os.path.basename(args.pid_a))
            logger.info(
                f"Ingested PID A: {doc_a.metadata.page_count} page(s) via {doc_a.metadata.format}"
            )
            print(
                f"  Ingested {len(doc_a.pages)} page(s) via adapter '{doc_a.metadata.format}'"
            )

            print("Ingesting Document B...")
            doc_b = detect_and_ingest(args.pid_b, pid=os.path.basename(args.pid_b))
            logger.info(
                f"Ingested PID B: {doc_b.metadata.page_count} page(s) via {doc_b.metadata.format}"
            )
            print(
                f"  Ingested {len(doc_b.pages)} page(s) via adapter '{doc_b.metadata.format}'"
            )

        with tracer.span("compute_delta"):
            logger.info("Stage 2: Computing delta")
            print("\nComputing delta engine analysis...")
            engine = DeltaEngine()
            delta_res = engine.compute_delta(doc_a, doc_b)
            logger.info(f"Delta computation complete: {delta_res.summary}")
            print(f"  Delta Summary: {delta_res.summary}")

        with tracer.span("generate_report"):
            logger.info("Stage 3: Generating delta report & visual redline overlay")
            md_content, json_content = generate_delta_report(
                delta_res, output_dir=args.output_dir
            )
            markup_path = generate_delta_markup(
                args.pid_b,
                delta_res,
                output_path=os.path.join(args.output_dir, "annotated_delta.pdf"),
            )
            print(f"  Saved Delta Reports to: {args.output_dir}/")
            if markup_path:
                print(f"  Saved Visual Redline Markup to: {markup_path}")

        trace_path = tracer.finish(output_dir=os.path.join(args.output_dir, "traces"))
        logger.info(f"Observability trace written to {trace_path}")
        print(f"  Saved Observability Trace: {trace_path}\n")

        if args.chat:
            with tracer.span("chat_session"):
                logger.info("Stage 4: Launching Grounded Chat")
                print("\n=== Building Grounded Chat Index ===")
                index = DocumentIndex(
                    persist_dir=os.path.join(args.output_dir, ".chroma")
                )
                index.build_index(doc_a, doc_b, md_content)
                chat_engine = AnswerEngine(index=index)

                print(
                    "\n--- Interactive Grounded Chat (type 'exit' or 'quit' to stop) ---"
                )
                while True:
                    try:
                        question = input(
                            "\nAsk a question about the documents/deltas: "
                        ).strip()
                        if not question or question.lower() in ["exit", "quit"]:
                            print("Exiting chat mode. Goodbye!")
                            break

                        res = chat_engine.answer_question(question, tracer=tracer.trace)
                        print(f"\nAnswer:\n{res.answer}\n")
                        if res.citations:
                            print("Citations:")
                            for cit in res.citations:
                                print(f"  - {cit.snippet}")
                    except (KeyboardInterrupt, EOFError):
                        print("\nExiting chat mode. Goodbye!")
                        break

        return 0

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        print(f"\nPipeline Execution Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
