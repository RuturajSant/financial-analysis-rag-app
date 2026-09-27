"""
===============================================================================
Financial RAG Agent - Evaluation & Benchmarking Harness
===============================================================================
Purpose:
    Automated evaluation framework to test Financial RAG Agents against 
    ground-truth financial evaluation datasets (JSON).

Metrics Evaluated:
    1. Correctness: Accurate numeric extraction, calculations, and answers.
    2. Citation Accuracy: Verifies if the agent cited the correct source page.
    3. Hallucination Defense: Verifies that unanswerable questions trigger a
       proper "not found" refusal instead of a hallucinated answer.

Quick Start:
    python benchmark_financial_rag.py \
      --dataset acme_synthetic_document_benchmark.json \
      --document ../../data/document_corpus/tier_1_synthetic_FS/statement_1_clean_single.pdf
===============================================================================
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage
from backend.agents.graph import build_graph
from backend.ingestion.pipeline import ingest_document
from backend.memory.checkpointer import get_checkpointer


# =============================================================================
# STEP 1: RAG AGENT INVOCATION
# =============================================================================
def call_rag_agent(
    question: str,
    graph,
    doc_id: str,
    thread_id: str = "benchmark_thread",
) -> Tuple[str, List[int]]:
    """
    Invoke the LangGraph RAG pipeline and return the answer text and cited pages.

    Args:
        question: The prompt/question to ask the RAG agent.
        graph: Compiled LangGraph graph instance.
        doc_id: Document ID returned by the ingestion pipeline.
        thread_id: Thread ID for checkpointer state (keeps follow-up context).

    Returns:
        Tuple of (response_text, list_of_cited_page_numbers).
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "doc_id": doc_id,
            "intent": None,
            "retrieved_context": None,
            "extracted_kpis": {},
            "kpi_result": None,
            "summary_result": None,
            "error": None,
        },
        config=config,
    )

    last_msg = result["messages"][-1]
    response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    cited_pages: set[int] = set()
    for chunk in result.get("retrieved_context") or []:
        if isinstance(chunk, dict) and "page_number" in chunk:
            cited_pages.add(chunk["page_number"])

    return response_text, list(cited_pages)


# =============================================================================
# STEP 2: EVALUATION HARNESS CLASS
# =============================================================================
class FinancialRAGEvaluator:
    """Evaluates RAG agent outputs against ground-truth benchmarks."""

    UNANSWERABLE_REFUSAL_TERMS = [
        "not found",
        "information not found",
        "not provided",
        "does not state",
        "absent",
        "cannot be found",
    ]

    def __init__(self, benchmark_filepath: str):
        """Loads the evaluation dataset from a JSON file."""
        with open(benchmark_filepath, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate_item(
        self, item: Dict, agent_response: str, retrieved_pages: List[int]
    ) -> Dict:
        """Evaluates a single question-answer pair against the ground truth."""
        q_type = item["type"]
        expected = item["expected_answer"]
        target_page = item["source_page"]

        metrics = {
            "id": item["id"],
            "type": q_type,
            "question": item["question"],
            "correctness": False,
            "citation_accuracy": False,
            "hallucination_detected": False,
        }

        # 1. Unanswerable Guardrail Check (Must refuse to answer)
        if q_type == "unanswerable":
            is_refused = any(
                term in agent_response.lower()
                for term in self.UNANSWERABLE_REFUSAL_TERMS
            )
            metrics["correctness"] = is_refused
            metrics["hallucination_detected"] = not is_refused
            metrics["citation_accuracy"] = (
                len(retrieved_pages) == 0 or target_page == "N/A"
            )
            return metrics

        # 2. Citation Accuracy Check
        if isinstance(target_page, int):
            metrics["citation_accuracy"] = target_page in retrieved_pages

        # 3. Fact and Number Extraction Accuracy Check
        if q_type in ["direct_lookup", "table_lookup", "derived_metric"]:
            expected_nums = re.findall(r"\d+(?:\.\d+)?", expected.replace(",", ""))
            actual_nums = re.findall(r"\d+(?:\.\d+)?", agent_response.replace(",", ""))
            metrics["correctness"] = any(num in actual_nums for num in expected_nums)
        else:
            # Overlap check for narrative/summary items
            metrics["correctness"] = (
                len(set(expected.lower().split()) & set(agent_response.lower().split())) > 3
            )

        return metrics

    def run_benchmark(self, graph, doc_id: str) -> List[Dict]:
        """Runs the entire dataset through the RAG agent and prints results."""
        results = []
        print(f"\n--- Starting Benchmark: {len(self.dataset)} Items ---")

        for item in self.dataset:
            response_text, cited_pages = call_rag_agent(
                item["question"], graph, doc_id
            )
            eval_result = self.evaluate_item(item, response_text, cited_pages)
            results.append(eval_result)

            status = "PASS" if eval_result["correctness"] else "FAIL"
            print(f"[{status}] ID: {item['id']} | Type: {item['type']}")

        return results


# =============================================================================
# STEP 3: CLI EXECUTION & SUMMARY REPORT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial RAG Evaluation Harness")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to benchmark JSON dataset",
    )
    parser.add_argument(
        "--document",
        type=str,
        required=True,
        help="Path to the PDF document to ingest and evaluate against",
    )
    args = parser.parse_args()

    # 1. Ingest the document
    pdf_path = Path(args.document)
    if not pdf_path.exists():
        print(f"Error: Document not found: {pdf_path}")
        sys.exit(1)

    print(f"Ingesting document: {pdf_path.name} ...")
    doc_id = ingest_document(str(pdf_path))
    print(f"Document ingested. doc_id={doc_id}")

    # 2. Build graph with in-memory checkpointer (preserves follow-up context)
    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    print("LangGraph compiled.")

    # 3. Run the benchmark
    evaluator = FinancialRAGEvaluator(args.dataset)
    benchmark_results = evaluator.run_benchmark(graph, doc_id)

    # 4. Calculate overall metrics
    total = len(benchmark_results)
    correct = sum(1 for r in benchmark_results if r["correctness"])
    accurate_citations = sum(1 for r in benchmark_results if r["citation_accuracy"])
    hallucinations = sum(1 for r in benchmark_results if r["hallucination_detected"])

    print("\n" + "=" * 50)
    print(" BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Evaluated Questions : {total}")
    print(f"Overall Accuracy          : {(correct / total) * 100:.2f}%")
    print(f"Citation Precision        : {(accurate_citations / total) * 100:.2f}%")
    print(f"Hallucination Failures    : {hallucinations}")
    print("=" * 50 + "\n")