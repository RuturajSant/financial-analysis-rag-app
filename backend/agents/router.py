"""
Intent router — classifies each incoming user message into one of three intents.

Design:
- Uses a cheap LLM call (not the full chat model) to classify intent.
- Keeps the router fast since it runs on every turn.
- Falls back to "retrieval" for ambiguous inputs.
"""
from __future__ import annotations

from backend.core.llm_provider import get_chat_llm
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import GraphState
from backend.core.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a routing assistant for a financial document Q&A system.
Classify the user's latest message into exactly one of these intents:

- "kpi": User wants specific numeric financial metrics (revenue, net income, EPS, assets, liabilities, equity, margins, growth rates).
- "summary": User wants a narrative overview or summary of the document or a section.
- "retrieval": User wants to find information, ask a question, or anything else.

Respond with ONLY one word: kpi, summary, or retrieval.
"""


def route_intent(state: GraphState) -> GraphState:
    """Classify the latest message and set state['intent']."""
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    router_llm = get_chat_llm(temperature=0, max_tokens=5)

    try:
        response = router_llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ])
        raw = response.content.strip().lower()
        intent = raw if raw in {"kpi", "summary", "retrieval"} else "retrieval"
    except Exception as exc:
        logger.warning("router_fallback", extra={"error": str(exc)})
        intent = "retrieval"

    logger.info("intent_routed", extra={"intent": intent, "query_preview": content[:80]})
    return {**state, "intent": intent}


def route_edge(state: GraphState) -> str:
    """Conditional edge function — returns the next node name based on intent."""
    return state.get("intent", "retrieval")
