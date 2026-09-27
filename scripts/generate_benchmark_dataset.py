"""
Benchmark Dataset Generator for Financial RAG

Iterates over Tier-2 PDFs (Dixon, Infosys, Titan) in data/document_corpus/tier_2_real_annual_reports
and generates structured ground-truth evaluation JSON files matching the benchmark schema.

Output Files:
  - data/ground_truth_data_evaluation/dixon_real_report_benchmark.json
  - data/ground_truth_data_evaluation/infosys_real_report_benchmark.json
  - data/ground_truth_data_evaluation/titan_real_report_benchmark.json
"""

import sys
import os
import json
import re
from pathlib import Path
import pdfplumber

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try importing LLM provider from backend
try:
    from backend.core.llm_provider import get_chat_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except Exception as e:
    LLM_AVAILABLE = False
    print(f"Notice: LLM provider not initialized ({e}). Using rule-based fallback generator.")

DOC_CORPUS_DIR = Path("data/document_corpus/tier_2_real_annual_reports")
OUTPUT_DIR = Path("data/ground_truth_data_evaluation")

COMPANY_MAPPING = {
    "Tier2_Dixon_Financials_MDA.pdf": ("dixon", "dixon_real_report_benchmark.json"),
    "Tier2_Infosys_Financials_MDA.pdf": ("infosys", "infosys_real_report_benchmark.json"),
    "Tier2_Titan_Financials_MDA.pdf": ("titan", "titan_real_report_benchmark.json"),
}


def extract_pdf_summary(pdf_path: Path, max_pages: int = 15) -> list[dict]:
    """Extract page text and basic tables from a PDF using pdfplumber."""
    pages_data = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        num_pages = len(pdf.pages)
        step = max(1, num_pages // max_pages)
        target_indices = list(range(0, num_pages, step))[:max_pages]

        for p_idx in target_indices:
            page = pdf.pages[p_idx]
            page_num = p_idx + 1
            text = page.extract_text() or ""
            tables = page.extract_tables()
            
            clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            
            pages_data.append({
                "page_number": page_num,
                "text": clean_text[:2000],  # Truncate for prompt efficiency
                "tables_count": len(tables),
                "has_numbers": bool(re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", clean_text))
            })
    return pages_data


def generate_with_llm(company_code: str, pages_data: list[dict]) -> list[dict]:
    """Use LLM to generate high-quality benchmark items adhering strictly to schema."""
    llm = get_chat_llm(temperature=0.1)
    
    context_blocks = []
    for p in pages_data:
        if p["text"]:
            context_blocks.append(f"--- PAGE {p['page_number']} ---\n{p['text']}")
    
    full_context = "\n\n".join(context_blocks[:8])
    
    system_prompt = (
        "You are an expert financial benchmark creator. Analyze the provided financial report excerpts "
        "and generate a JSON array of 8 diverse, factual, non-hallucinated question-answer pairs.\n"
        "Categories to include:\n"
        "1. direct_lookup (Direct metric from text)\n"
        "2. table_lookup (Metric from tabular data)\n"
        "3. derived_metric (Calculated metric like margin/ratio with formula shown)\n"
        "4. yoy_comparison (Comparison across years)\n"
        "5. narrative (Qualitative MD&A risk or strategy insight)\n"
        "6. unanswerable (Question about future/unrelated metrics not in doc, set source_page to 'N/A' and expected_answer to 'Information not found in the document.')\n"
        "7. follow_up (Short pronoun-based follow-up question)\n"
        "8. summary (High-level summary of financial position or segment)\n\n"
        "STRICT JSON SCHEMA:\n"
        "[\n"
        "  {\n"
        f'    "id": "{company_code}_001",\n'
        '    "type": "direct_lookup",\n'
        '    "question": "Exact question text",\n'
        '    "expected_answer": "Exact answer text with numeric values if applicable",\n'
        '    "source_page": 1, // integer or "N/A"\n'
        '    "context_snippet": "Exact quote from context_blocks"\n'
        "  }\n"
        "]\n"
        "Ensure valid raw JSON output without markdown block syntax if possible, or inside ```json code block."
    )
    
    user_prompt = f"Company Code: {company_code}\nDocument Context:\n{full_context}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    raw_content = response.content.strip()
    if raw_content.startswith("```"):
        raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
        raw_content = re.sub(r"\n?```$", "", raw_content)
        
    items = json.loads(raw_content)
    return validate_and_fix_schema(company_code, items)


def validate_and_fix_schema(company_code: str, items: list[dict]) -> list[dict]:
    """Validate and enforce standard schema across generated benchmark items."""
    valid_types = {
        "direct_lookup", "table_lookup", "derived_metric",
        "yoy_comparison", "narrative", "unanswerable",
        "follow_up", "summary"
    }
    
    cleaned = []
    for idx, item in enumerate(items, 1):
        q_type = item.get("type", "direct_lookup")
        if q_type not in valid_types:
            q_type = "direct_lookup"
            
        page = item.get("source_page")
        if q_type == "unanswerable":
            page = "N/A"
            expected = "Information not found in the document."
            snippet = None
        else:
            if isinstance(page, str) and page.isdigit():
                page = int(page)
            elif not isinstance(page, int):
                page = 1
            expected = str(item.get("expected_answer", ""))
            snippet = item.get("context_snippet")
            
        cleaned.append({
            "id": f"{company_code}_{idx:03d}",
            "type": q_type,
            "question": str(item.get("question", "")),
            "expected_answer": expected,
            "source_page": page,
            "context_snippet": snippet
        })
    return cleaned


def rule_based_fallback(company_code: str, pdf_path: Path) -> list[dict]:
    """Fallback generator using rule-based extraction from PDF pages."""
    items = []
    pages_data = extract_pdf_summary(pdf_path, max_pages=20)
    
    num_found = 0
    for p in pages_data:
        text = p["text"]
        lines = text.splitlines()
        page_num = p["page_number"]
        
        for line in lines:
            if any(k in line.lower() for k in ["revenue", "turnover", "income from operations", "profit after tax"]):
                match = re.search(r"(?:₹|\$|INR|USD)?\s*[\d,]+(?:\.\d+)?\s*(?:crore|million|billion|lakh)?", line, re.IGNORECASE)
                if match and num_found < 3:
                    num_found += 1
                    items.append({
                        "id": f"{company_code}_{num_found:03d}",
                        "type": "direct_lookup",
                        "question": f"What was the reported metric value in: '{line[:50]}...'?",
                        "expected_answer": line.strip(),
                        "source_page": page_num,
                        "context_snippet": line.strip()
                    })
                    break

    default_items = [
        {
            "id": f"{company_code}_004",
            "type": "derived_metric",
            "question": f"What is the net profit margin for {company_code.capitalize()}?",
            "expected_answer": "Refer to statement of profit and loss for calculations.",
            "source_page": 1,
            "context_snippet": pages_data[0]["text"][:200] if pages_data else "Excerpt"
        },
        {
            "id": f"{company_code}_005",
            "type": "narrative",
            "question": f"What operational risks or business highlights are discussed for {company_code.capitalize()}?",
            "expected_answer": "Management discusses operational strategy, market positioning, and revenue growth in MD&A.",
            "source_page": 1,
            "context_snippet": pages_data[0]["text"][:200] if pages_data else "Excerpt"
        },
        {
            "id": f"{company_code}_006",
            "type": "unanswerable",
            "question": "What is the projected revenue target for FY2035?",
            "expected_answer": "Information not found in the document.",
            "source_page": "N/A",
            "context_snippet": None
        },
        {
            "id": f"{company_code}_007",
            "type": "summary",
            "question": f"Summarize the overall performance of {company_code.capitalize()} from the MD&A report.",
            "expected_answer": f"{company_code.capitalize()} reported operational metrics in line with MD&A disclosures.",
            "source_page": 1,
            "context_snippet": pages_data[0]["text"][:200] if pages_data else "Excerpt"
        }
    ]
    
    items.extend(default_items)
    return validate_and_fix_schema(company_code, items)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, (company_code, json_filename) in COMPANY_MAPPING.items():
        pdf_path = DOC_CORPUS_DIR / filename
        out_json_path = OUTPUT_DIR / json_filename
        
        if not pdf_path.exists():
            print(f"Warning: PDF file {pdf_path} does not exist. Skipping.")
            continue
            
        print(f"\nProcessing {filename} ({company_code.upper()})...")
        pages_data = extract_pdf_summary(pdf_path)
        
        benchmark_items = []
        if LLM_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                print(f"Generating ground truth via LLM for {company_code}...")
                benchmark_items = generate_with_llm(company_code, pages_data)
            except Exception as exc:
                print(f"LLM generation failed ({exc}). Falling back to rule-based generation...")
                benchmark_items = rule_based_fallback(company_code, pdf_path)
        else:
            print(f"Generating ground truth via rule-based parser for {company_code}...")
            benchmark_items = rule_based_fallback(company_code, pdf_path)
            
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_items, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully generated benchmark JSON: {out_json_path} ({len(benchmark_items)} items)")


if __name__ == "__main__":
    main()
