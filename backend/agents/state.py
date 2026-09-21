"""
LangGraph state schema — shared across all nodes.

Design (from references/orchestration.md):
- Keep state small and serializable for cheap checkpointing.
- Store chunk IDs/references, not full texts, in long-lived state.
- extracted_kpis keyed by doc_id so KPI extraction isn't re-run on follow-ups.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    # Conversation messages (managed by LangGraph's add_messages reducer)
    messages: Annotated[list, add_messages]

    # Active document for this session
    doc_id: str

    # Classified intent for the current turn
    intent: Optional[Literal["retrieval", "kpi", "summary"]]

    # Retrieved context chunks for the current turn (list of dicts from hybrid_search)
    retrieved_context: Optional[list[dict]]

    # KPI extraction results cached by doc_id across turns — avoids re-extraction
    extracted_kpis: dict  # {doc_id: FinancialKPIs.model_dump()}

    # Agent outputs for the current turn
    kpi_result: Optional[dict]
    summary_result: Optional[str]

    # Error from any node — surfaced to user rather than crashing the graph
    error: Optional[str]
