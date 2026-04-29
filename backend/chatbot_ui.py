"""
Physiotherapy Cost-Effectiveness — Chat Interface
Single-page conversational UI. No tabs. Just ask questions.
"""

import os
from typing import Optional
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PhysioAI · Cost-Effectiveness",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Header ── */
.app-header {
    padding: 1.2rem 1.6rem 0.8rem;
    border-bottom: 1px solid #e8ecf0;
    margin-bottom: 1.2rem;
}
.app-header h1 {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1a2332;
    margin: 0 0 0.15rem 0;
    letter-spacing: -0.3px;
}
.app-header p {
    font-size: 0.82rem;
    color: #7a8a9a;
    margin: 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.2rem 0;
}

/* ── Source pills ── */
.source-pill {
    display: inline-block;
    background: #f0f4ff;
    border: 1px solid #d0d8ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.73rem;
    color: #3a5acd;
    margin: 2px 3px 2px 0;
    font-weight: 500;
}

/* ── Mode badge ── */
.mode-badge {
    display: inline-block;
    background: #f0faf4;
    border: 1px solid #b0ddc0;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.72rem;
    color: #2d7a4f;
    font-weight: 600;
    margin-bottom: 6px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f8fafc;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* ── Status dot ── */
.status-online  { color: #22c55e; font-weight: 600; }
.status-offline { color: #ef4444; font-weight: 600; }

/* ── Example pill ── */
.example-q {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 0.82rem;
    color: #334155;
    margin-bottom: 5px;
    cursor: default;
    line-height: 1.4;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid #e8ecf0; margin: 0.8rem 0; }

/* ── Info box ── */
.info-box {
    background: #f0f7ff;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #1e3a5f;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[dict] = None) -> dict:
    try:
        r = requests.get(f"{API_BASE}{url}", params=params, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(url: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{url}", json=payload, timeout=360)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _api_ok() -> bool:
    try:
        requests.get(f"{API_BASE}/status", timeout=4)
        return True
    except Exception:
        return False


def _load_papers() -> list:
    return _get("/list_papers").get("papers", [])


# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in [
    ("messages",     []),
    ("paper_list",   []),
    ("selected_paper", ""),
    ("model_list",   []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Status
    online = _api_ok()
    status_cls = "status-online" if online else "status-offline"
    status_txt = "● Backend online" if online else "● Backend offline"
    st.markdown(f'<p class="{status_cls}">{status_txt}</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Model selector ───────────────────────────────────────────────────────
    st.markdown("**🤖 LLM Model**")

    if not st.session_state.model_list:
        st.session_state.model_list = _get("/list_models").get("models", [])

    current_model = _get("/current_model").get("model", "llama3.1:8b")
    model_options = st.session_state.model_list or [current_model]

    if current_model in model_options:
        default_idx = model_options.index(current_model)
    else:
        # prefer qwen3.5:9b, then any qwen, then first available
        preferred = [i for i, m in enumerate(model_options) if m == "qwen3.5:9b"]
        if not preferred:
            preferred = [i for i, m in enumerate(model_options) if "qwen3.5" in m.lower()]
        if not preferred:
            preferred = [i for i, m in enumerate(model_options) if "qwen" in m.lower()]
        default_idx = preferred[0] if preferred else 0

    selected_model = st.selectbox(
        "Model",
        options=model_options,
        index=default_idx,
        label_visibility="collapsed",
        help="Larger models give better extraction & reasoning quality",
    )

    if selected_model != current_model:
        res = _post("/set_model", {"model": selected_model})
        if "error" not in res:
            st.success(f"Switched to {selected_model}")

    if st.button("Refresh model list", use_container_width=True):
        st.session_state.model_list = _get("/list_models").get("models", [])
        st.rerun()

    st.markdown("---")

    # ── Query mode ───────────────────────────────────────────────────────────
    st.markdown("**⚙️ Query Mode**")
    mode = st.radio(
        "Mode",
        ["Auto", "Single-paper", "Cross-paper"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Auto: routes based on your question\n"
            "Single-paper: ask about one specific paper\n"
            "Cross-paper: compare across all studies"
        ),
    )

    if mode == "Single-paper":
        if not st.session_state.paper_list:
            st.session_state.paper_list = _load_papers()
        papers = st.session_state.paper_list
        if papers:
            sp = st.selectbox(
                "Paper",
                options=papers,
                label_visibility="collapsed",
                key="paper_selectbox",
            )
            st.session_state.selected_paper = sp
        else:
            st.warning("No papers found. Run /build_ce_table first.")
            st.session_state.selected_paper = ""

    top_k = st.slider("Chunks retrieved (RAG)", 4, 60, 12,
                      help="More chunks = more context, slower")

    st.markdown("---")
    st.caption("Physio CE Analyser · v2.0")


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>🏥 Physiotherapy Cost-Effectiveness Analyser</h1>
</div>
""", unsafe_allow_html=True)


# ── Example questions ─────────────────────────────────────────────────────────

with st.expander("💡 What can I ask?", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fig 4 & Fig 5**")
        for q in [
            "What does Figure 4 show and how many comparisons are dominant?",
            "What does Figure 5 represent? How is it different from Fig 4?",
            "Why do Fig 4 and Fig 5 have different quadrant distributions?",
        ]:
            st.markdown(f'<div class="example-q">💬 {q}</div>', unsafe_allow_html=True)

        st.markdown("**Extraction quality**")
        for q in [
            "Is the content of Tables 1 and 2 correctly extracted?",
            "Do you agree with the quadrant location of comparisons in Figure 5?",
        ]:
            st.markdown(f'<div class="example-q">💬 {q}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**By body region**")
        for q in [
            "Is there a difference between CE and non-CE interventions for knee?",
            "For shoulder, which interventions are cost-effective and why?",
            "For hip, what drives cost-effectiveness?",
            "For low back pain, are exercise programs cost-effective?",
        ]:
            st.markdown(f'<div class="example-q">💬 {q}</div>', unsafe_allow_html=True)

        st.markdown("**Intervention dose (Table 2)**")
        for q in [
            "For knee, how does session frequency differ between CE and non-CE studies?",
            "Does intervention duration affect cost-effectiveness for shoulder?",
            "For hip, what is the typical session length in cost-effective studies?",
        ]:
            st.markdown(f'<div class="example-q">💬 {q}</div>', unsafe_allow_html=True)


# ── Cross-paper routing keywords ──────────────────────────────────────────────

CROSS_KW = [
    # Figures
    "fig 4", "figure 4", "fig4", "fig 5", "figure 5", "fig5",
    "both figure", "compare figure",
    # CE outcomes
    "dominant", "dominated", "quadrant", "cost-effective", "cost effective",
    "icer", "cost-effectiveness plane",
    # Extraction / validation
    "table 1", "table 2", "table2", "correctly extracted", "extraction correct",
    "is it correct", "are these correct", "did you extract", "is the content",
    "do you agree", "agree with", "placement", "location of", "correctly placed",
    # Cross-study patterns
    "across studies", "across papers", "overall", "all studies",
    "difference between", "perspective", "time horizon",
    "which intervention", "why",
    # Body regions
    "for knee", "for shoulder", "for hip", "for back", "for neck",
    "for the knee", "for the shoulder", "for the hip", "for low back",
    # Dose
    "frequency", "sessions per week", "session length", "duration",
    "how often", "how long", "dose", "supervision",
]


# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────

user_q = st.chat_input("Ask anything about the cost-effectiveness papers…")

if user_q:
    # Guard: single-paper mode needs a paper selected
    if mode == "Single-paper" and not st.session_state.selected_paper:
        st.warning("Please select a paper in the sidebar first.")
        st.stop()

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Route
    q_lower = user_q.lower()
    if mode == "Cross-paper":
        use_compare = True
    elif mode == "Single-paper":
        use_compare = False
    else:
        use_compare = any(k in q_lower for k in CROSS_KW)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if use_compare:
                route_label = "cross-paper"
                data = _post("/ask_compare", {"question": user_q})
            else:
                route_label = f"single-paper · {st.session_state.selected_paper or 'all'}"
                data = _post("/ask", {
                    "question": user_q,
                    "top_k": int(top_k),
                    "paper_id": st.session_state.selected_paper or None,
                })

        if "error" in data:
            answer = f"⚠️ Error: {data['error']}"
            sources = []
        else:
            answer = data.get("answer", "No answer returned.")
            sources = data.get("sources", [])
            # Deduplicate sources
            seen: set = set()
            unique_sources = []
            for s in sources:
                key = (s.get("paper_id"), s.get("page"))
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)
            sources = unique_sources

        # Mode badge + answer
        st.markdown(
            f'<div class="mode-badge">{"🔀" if use_compare else "📄"} {route_label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(answer)

        # Store
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""

""", unsafe_allow_html=True)
