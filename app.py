"""
AI Financial Document Q&A Assistant — Streamlit Frontend

Layout:
  - Sidebar: file upload + session info
  - Main: tab 1 = Chat, tab 2 = KPI Dashboard
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoid slow imports on cold start) ───────────────────────────
@st.cache_resource
def get_graph(persistence_enabled: bool = True):
    """Build graph. The persistence_enabled arg also acts as a cache key."""
    from backend.agents.graph import build_graph
    from backend.memory.checkpointer import get_checkpointer
    from backend.core.config import settings
    settings.enable_session_persistence = persistence_enabled
    checkpointer = get_checkpointer()
    return build_graph(checkpointer=checkpointer)


@st.cache_resource
def get_ingest_fn():
    from backend.ingestion.pipeline import ingest_document
    return ingest_document


# ── Session persistence helpers ───────────────────────────────────────────────
_SESSION_FILE = Path("./metadata/last_session.json")


def _save_session() -> None:
    """Write doc_id, doc_name, thread_id to disk."""
    _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thread_id": st.session_state.thread_id,
        "doc_id": st.session_state.doc_id,
        "doc_name": st.session_state.doc_name,
    }
    _SESSION_FILE.write_text(json.dumps(payload), encoding="utf-8")


def _load_session() -> dict | None:
    """Return saved session dict or None if file is missing / corrupt."""
    try:
        return json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_session_file() -> None:
    """Delete the persisted session file."""
    try:
        _SESSION_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Session state init ─────────────────────────────────────────────────────────
def init_session():
    if "_session_initialised" in st.session_state:
        return
    st.session_state["_session_initialised"] = True

    # Defaults
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("doc_id", None)
    st.session_state.setdefault("doc_name", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("kpi_result", None)
    st.session_state.setdefault("ingested", False)
    st.session_state.setdefault("persistence_enabled", True)

    # Restore last session from disk (if any)
    saved = _load_session()
    if saved and saved.get("doc_id"):
        st.session_state.thread_id = saved["thread_id"]
        st.session_state.doc_id    = saved["doc_id"]
        st.session_state.doc_name  = saved["doc_name"]
        st.session_state.ingested  = True

        # Restore chat history & KPI results from LangGraph checkpointer into Streamlit UI
        try:
            graph = get_graph(st.session_state.persistence_enabled)
            state_snap = graph.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
            if state_snap and state_snap.values:
                msgs = state_snap.values.get("messages", [])
                restored_msgs = []
                for msg in msgs:
                    msg_type = getattr(msg, "type", "")
                    content = getattr(msg, "content", str(msg))
                    if msg_type == "human" or type(msg).__name__ == "HumanMessage":
                        restored_msgs.append({"role": "user", "content": content})
                    elif msg_type == "ai" or type(msg).__name__ == "AIMessage":
                        restored_msgs.append({"role": "assistant", "content": content})
                if restored_msgs:
                    st.session_state.messages = restored_msgs

                if state_snap.values.get("kpi_result"):
                    st.session_state.kpi_result = state_snap.values["kpi_result"]
        except Exception:
            pass


init_session()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    color: #e2e8f0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3f 0%, #0f172a 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.3);
}

/* Chat message bubbles */
.chat-message-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px 60px;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    line-height: 1.5;
}

.chat-message-assistant {
    background: rgba(30, 41, 59, 0.8);
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px 60px 8px 0;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    line-height: 1.5;
}

.chat-label-user {
    text-align: right;
    font-size: 11px;
    color: #a78bfa;
    margin-bottom: 2px;
    margin-right: 4px;
}

.chat-label-assistant {
    text-align: left;
    font-size: 11px;
    color: #60a5fa;
    margin-bottom: 2px;
    margin-left: 4px;
}

/* KPI cards */
.kpi-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #a78bfa;
}
.kpi-confidence-high   { color: #34d399; font-size: 10px; margin-top: 4px; }
.kpi-confidence-medium { color: #fbbf24; font-size: 10px; margin-top: 4px; }
.kpi-confidence-low    { color: #f87171; font-size: 10px; margin-top: 4px; }
.kpi-confidence-not_found { color: #475569; font-size: 10px; margin-top: 4px; }

/* Upload zone */
.upload-hint {
    color: #64748b;
    font-size: 13px;
    text-align: center;
    padding: 8px;
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.status-ready   { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.status-pending { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* Input box */
.stTextInput > div > div > input {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 10px;
    color: #e2e8f0;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
}

/* Streamlit tab overrides */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.25);
    color: #a78bfa;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: format KPI value for display ──────────────────────────────────────
def _fmt_kpi(kpi: dict | None) -> tuple[str, str]:
    """Return (display_value, confidence_level)."""
    if not kpi or kpi.get("confidence") == "not_found" or kpi.get("value") is None:
        return "—", "not_found"
    val = kpi["value"]
    unit = kpi.get("unit", "")
    conf = kpi.get("confidence", "not_found")
    if unit in ("USD_millions",):
        display = f"${val:,.0f}M"
    elif unit in ("USD_thousands",):
        display = f"${val:,.0f}K"
    elif unit == "percent":
        display = f"{val:.1f}%"
    elif unit == "USD":
        display = f"${val:,.0f}"
    elif unit == "shares":
        display = f"{val:.2f}"
    else:
        display = f"{val:,.2f}"
    return display, conf


def _kpi_card(label: str, kpi: dict | None) -> str:
    val, conf = _fmt_kpi(kpi)
    page = f"p.{kpi['source_page']}" if kpi and kpi.get("source_page") else ""
    return f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{val}</div>
    <div class="kpi-confidence-{conf}">{'●' if conf != 'not_found' else '○'} {conf.replace('_', ' ')}{' · ' + page if page else ''}</div>
</div>"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Q&A")
    st.markdown("---")

    # API Key input (if not set via env)
    if not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.markdown('<div class="status-badge status-ready">✓ API Key Loaded</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a financial PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="upload-hint">Annual reports, balance sheets, or any financial PDF</div>', unsafe_allow_html=True)

    if uploaded_file and not st.session_state.ingested:
        with st.spinner("Parsing and indexing document…"):
            try:
                # Save to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                ingest_fn = get_ingest_fn()
                doc_id = ingest_fn(tmp_path)
                st.session_state.doc_id = doc_id
                st.session_state.doc_name = uploaded_file.name
                st.session_state.ingested = True
                st.session_state.messages = []
                st.session_state.kpi_result = None
                os.unlink(tmp_path)
                _save_session()  # persist doc_id + thread_id to disk
                st.success("Document indexed!")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.markdown("---")

    # Session persistence toggle
    st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)
    persistence_on = st.toggle(
        "Session Persistence",
        value=st.session_state.persistence_enabled,
        help="When enabled, conversation history is preserved across reruns via a checkpointer. "
             "Disable for stateless mode (no memory between turns).",
    )
    if persistence_on != st.session_state.persistence_enabled:
        st.session_state.persistence_enabled = persistence_on
        get_graph.clear()  # invalidate cached graph so it recompiles
        st.rerun()

    st.markdown("---")

    # Document status
    if st.session_state.ingested:
        st.markdown('<div class="status-badge status-ready">✓ Document Ready</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='color:#64748b; font-size:12px; margin-top:6px;'>📄 {st.session_state.doc_name}</div>", unsafe_allow_html=True)

        if st.button("Clear Session", use_container_width=True):
            _clear_session_file()  # remove persisted session
            for key in ["doc_id", "doc_name", "messages", "kpi_result", "ingested",
                        "thread_id", "_session_initialised"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.markdown('<div class="status-badge status-pending">○ No Document</div>', unsafe_allow_html=True)

    st.markdown("---")
    from backend.core.config import settings as _cfg
    _provider_label = _cfg.llm_provider.capitalize()
    st.markdown(f"<div style='color:#475569; font-size:11px;'>Powered by LangGraph · ChromaDB · {_provider_label}</div>", unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# AI Financial Document Q&A")
st.markdown("<div style='color:#64748b; margin-bottom:20px;'>Ask questions, extract KPIs, or get summaries from your financial reports.</div>", unsafe_allow_html=True)

if not st.session_state.ingested:
    st.info("👈 Upload a financial PDF in the sidebar to get started.")
    st.stop()

tab_chat, tab_kpi = st.tabs(["💬 Chat", "📈 KPI Dashboard"])

# ── Tab 1: Chat ────────────────────────────────────────────────────────────────
with tab_chat:
    # Render message history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f'<div class="chat-label-user">You</div><div class="chat-message-user">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label-assistant">Assistant</div><div class="chat-message-assistant">{content}</div>', unsafe_allow_html=True)

    # Quick-action buttons
    st.markdown("---")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        if st.button("📋 Executive Summary", use_container_width=True):
            st.session_state["_pending_query"] = "Give me an executive summary of this annual report"
    with col_q2:
        if st.button("💰 Extract KPIs", use_container_width=True):
            st.session_state["_pending_query"] = "Extract all key financial KPIs from this document"
    with col_q3:
        if st.button("📉 Revenue & Margins", use_container_width=True):
            st.session_state["_pending_query"] = "What was the revenue and gross margin?"

    # Chat input
    user_input = st.chat_input("Ask a question about the document…")

    # Handle pending quick-action or user input
    query = user_input or st.session_state.pop("_pending_query", None)

    if query:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking…"):
            try:
                graph = get_graph(st.session_state.persistence_enabled)
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                from langchain_core.messages import HumanMessage
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "doc_id": st.session_state.doc_id,
                        "intent": None,
                        "retrieved_context": None,
                        "extracted_kpis": {},
                        "kpi_result": None,
                        "summary_result": None,
                        "error": None,
                    },
                    config=config,
                )
                # Extract the last AI message
                last_msg = result["messages"][-1]
                answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                # Cache KPI result if present
                if result.get("kpi_result"):
                    st.session_state.kpi_result = result["kpi_result"]

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {exc}"})

        st.rerun()


# ── Tab 2: KPI Dashboard ───────────────────────────────────────────────────────
with tab_kpi:
    kpis = st.session_state.kpi_result

    if not kpis:
        st.info("KPI data not yet extracted. Click **Extract KPIs** in the chat tab or ask a KPI question.")

        if st.button("🔍 Extract KPIs Now", use_container_width=False):
            with st.spinner("Extracting financial KPIs…"):
                try:
                    graph = get_graph(st.session_state.persistence_enabled)
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    from langchain_core.messages import HumanMessage
                    result = graph.invoke(
                        {
                            "messages": [HumanMessage(content="Extract all key financial KPIs from this document")],
                            "doc_id": st.session_state.doc_id,
                            "intent": "kpi",
                            "retrieved_context": None,
                            "extracted_kpis": {},
                            "kpi_result": None,
                            "summary_result": None,
                            "error": None,
                        },
                        config=config,
                    )
                    if result.get("kpi_result"):
                        st.session_state.kpi_result = result["kpi_result"]
                        st.rerun()
                    else:
                        st.error("KPI extraction returned no results.")
                except Exception as exc:
                    st.error(f"Extraction failed: {exc}")
    else:
        st.markdown("### Financial KPIs")
        st.markdown(f"<div style='color:#475569; font-size:12px; margin-bottom:16px;'>📄 {st.session_state.doc_name}</div>", unsafe_allow_html=True)

        # Row 1 — Revenue side
        cols = st.columns(4)
        kpi_rows = [
            [("Revenue", "revenue"), ("COGS", "cogs"), ("Gross Profit Margin", "gross_margin"), ("Net Income", "net_income")],
            [("EPS (Basic)", "eps"), ("Total Assets", "total_assets"), ("Total Liabilities", "total_liabilities"), ("Stockholders' Equity", "total_equity")],
        ]
        for row in kpi_rows:
            cols = st.columns(4)
            for col, (label, key) in zip(cols, row):
                with col:
                    st.markdown(_kpi_card(label, kpis.get(key)), unsafe_allow_html=True)
            st.markdown("")

        # YoY growth if present
        if kpis.get("yoy_revenue_growth") and kpis["yoy_revenue_growth"].get("confidence") != "not_found":
            st.markdown("---")
            col_g, _ = st.columns([1, 3])
            with col_g:
                st.markdown(_kpi_card("YoY Revenue Growth", kpis["yoy_revenue_growth"]), unsafe_allow_html=True)

        # Raw data expander
        with st.expander("Raw KPI Data (with source pages)"):
            st.json(kpis)
