"""
Test session persistence:
1. App session metadata (JSON file save & load).
2. LangGraph SQLite checkpointer (checkpoint saving & restoring across instances).
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMMessage
import operator


class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def _dummy_node(state: State):
    return {"messages": [AIMMessage(content="Hello! How can I help with your financial report?")]}


def test_app_session_metadata(tmp_path):
    session_file = tmp_path / "last_session.json"
    payload = {
        "thread_id": "test-thread-123",
        "doc_id": "doc-abc-456",
        "doc_name": "sample_report.pdf",
    }
    session_file.write_text(json.dumps(payload), encoding="utf-8")

    loaded = json.loads(session_file.read_text(encoding="utf-8"))
    assert loaded["thread_id"] == "test-thread-123"
    assert loaded["doc_id"] == "doc-abc-456"
    assert loaded["doc_name"] == "sample_report.pdf"


def test_sqlite_checkpointer_persistence(tmp_path):
    db_path = tmp_path / "test_checkpoints.db"
    
    # Session 1: Create graph with sqlite checkpointer and send message
    conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer1 = SqliteSaver(conn1)

    builder = StateGraph(State)
    builder.add_node("agent", _dummy_node)
    builder.add_edge(START, "agent")
    graph1 = builder.compile(checkpointer=checkpointer1)

    config = {"configurable": {"thread_id": "session-xyz"}}
    res1 = graph1.invoke({"messages": [HumanMessage(content="What is the revenue?")]}, config=config)
    assert len(res1["messages"]) == 2
    conn1.close()

    # Session 2 (Simulating server restart): Reconnect SQLite checkpointer and query state
    conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer2 = SqliteSaver(conn2)
    graph2 = builder.compile(checkpointer=checkpointer2)

    saved_state = graph2.get_state(config)
    assert saved_state is not None
    assert len(saved_state.values["messages"]) == 2
    assert saved_state.values["messages"][0].content == "What is the revenue?"
    conn2.close()
