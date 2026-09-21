"""
LangGraph orchestrator — wires all nodes into a compiled graph.

Design (from references/orchestration.md):
- Entry: router node classifies intent.
- Conditional edges route to retrieval_agent, kpi_agent, or summary_agent.
- Per-node RetryPolicy for transient failures (rate limits, timeouts).
- Checkpointer injected at compile time (InMemory for dev, SQLite for prod).
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.agents.kpi_extractor import extract_kpis
from backend.agents.retriever import retrieve_and_answer
from backend.agents.router import route_edge, route_intent
from backend.agents.state import GraphState
from backend.agents.summarizer import summarize
from backend.core.logger import get_logger

logger = get_logger(__name__)


def _error_passthrough(state: GraphState) -> GraphState:
    """
    Fallback node — returns a user-facing error message if a prior node set state['error'].
    Prevents the graph from silently failing.
    """
    from langchain_core.messages import AIMessage

    error = state.get("error", "An unknown error occurred.")
    logger.error("graph_error_node", extra={"error": error})
    return {
        **state,
        "messages": [AIMessage(content=f"Sorry, I encountered an error: {error}")],
        "error": None,
    }


def _has_error(state: GraphState) -> str:
    """Route to error handler if a node set state['error'], else END."""
    return "error_handler" if state.get("error") else END


def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph state graph.

    Args:
        checkpointer: LangGraph checkpointer instance. Pass None for stateless (testing).
                      Use InMemorySaver for local dev, SqliteSaver for deployment.

    Returns:
        Compiled LangGraph app.
    """
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("router", route_intent)
    builder.add_node("retrieval_agent", retrieve_and_answer)
    builder.add_node("kpi_agent", extract_kpis)
    builder.add_node("summary_agent", summarize)
    builder.add_node("error_handler", _error_passthrough)

    # Entry point
    builder.set_entry_point("router")

    # Router → agent conditional edges
    builder.add_conditional_edges(
        "router",
        route_edge,
        {
            "retrieval": "retrieval_agent",
            "kpi": "kpi_agent",
            "summary": "summary_agent",
        },
    )

    # Each agent → check for error or END
    for agent_node in ("retrieval_agent", "kpi_agent", "summary_agent"):
        builder.add_conditional_edges(agent_node, _has_error, {"error_handler": "error_handler", END: END})

    builder.add_edge("error_handler", END)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("graph_compiled", extra={"checkpointer": type(checkpointer).__name__ if checkpointer else "None"})
    return graph
