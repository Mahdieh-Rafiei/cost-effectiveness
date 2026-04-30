"""
Physiotherapy Cost-Effectiveness — Chat Interface
Single-page conversational UI. No tabs. Just ask questions.
"""

import os
import re
import json
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


def _stream_ask(payload: dict):
    """
    Yield (meta, text_chunk) tuples from /ask_stream.
    First item has meta dict with confidence info; subsequent items have meta=None.
    """
    buffer = ""
    meta_done = False
    meta = {}
    try:
        with requests.post(
            f"{API_BASE}/ask_stream",
            json=payload,
            stream=True,
            timeout=360,
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                if not meta_done:
                    buffer += chunk
                    if "\n---\n" in buffer:
                        parts = buffer.split("\n---\n", 1)
                        try:
                            meta = json.loads(parts[0])
                        except Exception:
                            pass
                        meta_done = True
                        if parts[1]:
                            yield meta, parts[1]
                            meta = None
                    # else still accumulating meta line
                else:
                    yield None, chunk
    except Exception as e:
        yield None, f"\n\n⚠️ Stream error: {e}"


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
    ("messages",       []),
    ("paper_list",     []),
    ("selected_paper", ""),
    ("paper_b",        ""),
    ("model_list",     []),
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
        ["Auto", "Single-paper", "Cross-paper", "Compare papers"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Auto: routes based on your question\n"
            "Single-paper: ask about one specific paper\n"
            "Cross-paper: compare across all studies\n"
            "Compare papers: side-by-side two papers"
        ),
    )

    if mode in ("Single-paper", "Compare papers"):
        if not st.session_state.paper_list:
            st.session_state.paper_list = _load_papers()
        papers = st.session_state.paper_list
        if papers:
            search = st.text_input(
                "Search paper", placeholder="Type to filter papers…",
                label_visibility="collapsed",
            )
            filtered = (
                [p for p in papers if search.lower() in p.lower()]
                if search else papers
            )
            if not filtered:
                filtered = papers

            label_a = "Paper A" if mode == "Compare papers" else "Paper"
            sp = st.selectbox(
                label_a, options=filtered,
                label_visibility="collapsed", key="paper_selectbox",
            )
            st.session_state.selected_paper = sp

            if mode == "Compare papers":
                sp_b = st.selectbox(
                    "Paper B", options=papers,
                    label_visibility="visible", key="paper_b_selectbox",
                )
                st.session_state.paper_b = sp_b
        else:
            st.warning("No papers found.")
            st.session_state.selected_paper = ""

    top_k = st.slider("Chunks retrieved (RAG)", 4, 60, 12,
                      help="More chunks = more context, slower")

    # ── Paper metadata card (info only, no buttons) ───────────────────────────
    if mode == "Single-paper" and st.session_state.get("selected_paper"):
        pid = st.session_state.selected_paper
        info = _get("/paper_info", params={"paper_id": pid})
        if info.get("found"):
            st.markdown("---")
            st.markdown("**📋 Paper snapshot**")

            _QUAD_ICON = {
                "dominant":  "🟢 Dominant",
                "dominated": "🔴 Dominated",
                "NE":        "🟡 NE",
                "SW":        "🔵 SW",
                "unclear":   "⚪ Unclear",
            }
            quadrants = info.get("quadrants", [])
            quad_display = (
                _QUAD_ICON.get(quadrants[0], quadrants[0])
                if len(set(quadrants)) == 1
                else f"{len(quadrants)} comparisons"
            )

            meta_lines = []
            if info.get("year") != "unknown":
                meta_lines.append(f"**Year:** {info['year']}")
            if info.get("body_region") != "unknown":
                meta_lines.append(f"**Region:** {info['body_region'].replace('_',' ').title()}")
            if info.get("intervention_type") != "unknown":
                meta_lines.append(f"**Intervention:** {info['intervention_type'].replace('_',' ')}")
            if info.get("comparator_type") != "unknown":
                meta_lines.append(f"**vs:** {info['comparator_type'].replace('_',' ')}")
            if info.get("time_horizon") != "unknown":
                meta_lines.append(f"**Follow-up:** {info['time_horizon']}")
            icer = info.get("icer", "unknown")
            if icer and icer != "unknown":
                meta_lines.append(f"**ICER:** {icer if isinstance(icer, str) else icer[0]}")
            meta_lines.append(f"**Quadrant:** {quad_display}")
            st.markdown("\n\n".join(meta_lines))

            # Suggested questions (display only)
            st.markdown("---")
            st.markdown("**💡 Try asking:**")
            region = info.get("body_region", "unknown").replace("_", " ")
            interv = info.get("intervention_type", "unknown").replace("_", " ")
            quad   = quadrants[0] if quadrants else "unclear"
            _is_review = ("systematic review" in pid.lower() or "review of trial" in pid.lower())

            if _is_review:
                suggestions = [
                    "What does Figure 4 show and how many comparisons are dominant?",
                    "Is the content of Tables 1 and 2 correctly extracted?",
                    "Why do Fig 4 and Fig 5 have different quadrant distributions?",
                    "Which body regions have the most cost-effective interventions?",
                ]
            else:
                suggestions = [
                    "What is the main cost-effectiveness finding of this study?",
                    f"How was the {interv} intervention delivered?" if interv != "unknown" else "How was the intervention delivered?",
                    f"What outcome measures were used?" if region == "unknown" else f"What outcome measures were used for {region}?",
                    "Why was this intervention cost-effective?" if quad == "dominant" else "What was the ICER value?",
                ]

            for sq in suggestions[:4]:
                st.markdown(f'<div class="example-q">💬 {sq}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Export chat ───────────────────────────────────────────────────────────
    if st.session_state.messages:
        lines = ["# Physio CE — Q&A Export\n"]
        for m in st.session_state.messages:
            role = "**You**" if m["role"] == "user" else "**Assistant**"
            lines.append(f"{role}\n\n{m['content']}\n\n---\n")
        export_text = "\n".join(lines)
        st.download_button(
            label="⬇️ Download chat (MD)",
            data=export_text,
            file_name="physio_ce_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

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
    # Extraction / validation (table 1/2 removed — handled by _is_validate path)
    "correctly extracted", "extraction correct",
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
    if mode == "Single-paper" and not st.session_state.selected_paper:
        st.warning("Please select a paper in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Build conversation history (last 6 messages = 3 exchanges)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-7:-1]  # exclude the just-added user msg
        if m["role"] in ("user", "assistant")
    ]

    q_lower = user_q.lower()

    # paper_id to use for the request
    active_paper_id = st.session_state.selected_paper or None

    # If the question explicitly references a figure and no paper is selected,
    # ask the user to specify rather than guessing wrong.
    _has_fig_ref = bool(re.search(r'\bfig(?:ure)?\b', q_lower))
    if _has_fig_ref and not active_paper_id and mode != "Cross-paper":
        clarify = (
            "I can describe figures with full visual analysis, but I need to know "
            "which paper you're referring to — each paper has its own Figure 4, "
            "Figure 5, etc.\n\n"
            "Please **select a paper from the sidebar** and ask again."
        )
        with st.chat_message("assistant"):
            st.markdown(clarify)
        st.session_state.messages.append({"role": "assistant", "content": clarify})
        st.stop()

    # Detect validation intent — route to validation endpoint, not vision
    _VALIDATE_KW = {
        "correctly extracted", "is correct", "are these correct",
        "is it correct", "correctly represented", "validate",
        "agree with the", "do you agree", "is the content",
        "correctly placed", "extraction correct", "run deep validation",
        "coverage", "how well extracted", "extraction quality",
    }
    _is_validate = any(k in q_lower for k in _VALIDATE_KW)

    _is_review_selected = active_paper_id and (
        "systematic review" in active_paper_id.lower() or
        "review of trial" in active_paper_id.lower()
    )

    # "Fix extractions" / "patch" triggers Table 1 patch (instant, no LLM)
    _PATCH_KW = {"patch table", "fix table 1", "apply table 1", "import table 1",
                 "fix body region", "fix country", "fix extraction",
                 "fix the extraction", "rebuild extraction", "re-extract",
                 "reextract", "fix incorrect", "improve extraction"}
    if any(k in q_lower for k in _PATCH_KW):
        with st.spinner("Patching database from Table 1…"):
            result = _post("/patch_table1", {})
        n = result.get("updated", "?")
        not_found = result.get("not_found", 0)
        answer = (
            f"Done — patched **{n} papers** with authoritative Table 1 data "
            f"(body region, country, study design, perspective, outcome measure, time horizon). "
            f"{not_found} papers could not be matched by author+year."
            f"\n\nAsk 'Is the content of Tables 1 and 2 correctly extracted?' to see the new coverage."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    # "Fix extractions" / "rebuild" triggers re-extraction of failed papers (uses LLM)
    _REBUILD_KW = {"fix incorrect", "improve extraction"}
    if any(k in q_lower for k in _REBUILD_KW):
        _post("/rebuild_failed", {})
        answer = (
            "Re-extraction started for all papers with incorrect/partial extractions. "
            "This uses the systematic review to fill in missing fields (body region, ICER, intervention, etc.). "
            "Takes ~5-10 minutes. When done, the data quality will be much better — "
            "ask 'Is the content of Tables 1 and 2 correctly extracted?' to see the results."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    # "Run deep validation" explicitly triggers background LLM validation
    if "run deep validation" in q_lower and _is_review_selected:
        _post("/batch_validate", {})
        answer = (
            "Deep validation started — checking all 78 papers against the systematic review. "
            "Takes ~15 minutes. Ask 'Is the content of Tables 1 and 2 correctly extracted?' "
            "when done to see the accuracy results."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()

    if mode == "Cross-paper":
        use_compare = True
    elif mode in ("Single-paper", "Compare papers"):
        use_compare = False
    else:
        use_compare = any(k in q_lower for k in CROSS_KW)

    with st.chat_message("assistant"):
        badge_label = (
            "⚖️ compare papers" if mode == "Compare papers"
            else ("🔀 cross-paper" if use_compare else f"📄 {active_paper_id or 'all'}")
        )
        st.markdown(
            f'<div class="mode-badge">{badge_label}</div>',
            unsafe_allow_html=True,
        )

        answer = ""

        # ── Table 2 coverage: get stats, then let LLM explain ───────────────
        _TABLE2_KW = {"table 2 coverage", "table2 coverage", "coverage of table 2",
                      "table 2 extraction", "what is the table 2", "show table 2 coverage",
                      "intervention coverage", "frequency coverage", "sessions coverage",
                      "icer coverage", "table 2 fields"}
        _is_table2_q = any(k in q_lower for k in _TABLE2_KW)

        if _is_table2_q and _is_review_selected:
            qv = _get("/quick_validate")
            t2 = qv.get("table2_fields", {})
            total = qv.get("total_papers", 0)
            avg2 = qv.get("table2_avg_coverage_pct", 0)

            # Show stats table
            def _bar2(pct):
                filled = round(pct / 10)
                return "█" * filled + "░" * (10 - filled) + f" {pct}%"
            t2_rows = "\n".join(
                f"| {k.replace('_', ' ').title()} | {_bar2(v['pct'])} | {v['extracted']}/{total} |"
                for k, v in t2.items()
            )
            stats_md = (
                f"**Table 2 — Intervention details ({total} papers, avg: {avg2}%)**\n\n"
                f"| Field | Coverage | Extracted |\n|---|---|---|\n{t2_rows}\n\n---\n\n"
            )

            # Use single placeholder to avoid double-render (stats + streaming explanation)
            stats_text = "\n".join(
                f"- {k.replace('_', ' ').title()}: {v['pct']}% extracted ({v['extracted']}/{total} papers)"
                for k, v in t2.items()
            )
            crafted_q = (
                f"We have a database of {total} physiotherapy cost-effectiveness papers. "
                f"Table 2 captures intervention details. Here is the current extraction coverage:\n\n"
                f"{stats_text}\n\n"
                f"Please explain: (1) what each field represents clinically, "
                f"(2) why fields like frequency (12%) and session length (8%) have such low coverage, "
                f"(3) which research questions we can reliably answer with this data, "
                f"(4) which questions we cannot answer due to missing data, "
                f"and (5) which fields are most critical to improve first."
            )
            placeholder = st.empty()
            placeholder.markdown(stats_md + "_Analysing coverage…_")
            explanation = ""
            for meta, chunk in _stream_ask({
                "question": crafted_q,
                "top_k": 0,
                "paper_id": None,
                "history": [],
            }):
                explanation += chunk
                placeholder.markdown(stats_md + explanation + "◌")
            placeholder.markdown(stats_md + explanation)
            answer = stats_md + explanation
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        if _is_validate and active_paper_id and not use_compare:
            if _is_review_selected:
                # Get Table 1 comparison data
                with st.spinner("Checking each paper against published Table 1…"):
                    vt1 = _get("/validate_table1")

                if "error" in vt1:
                    answer = f"⚠️ {vt1['error']}"
                    st.markdown(answer)
                else:
                    total_p = vt1.get("total_papers_checked", 0)
                    patched = vt1.get("patches_applied", 0)
                    unmatched_n = vt1.get("unmatched_count", 0)
                    fs = vt1.get("field_summary", {})
                    paper_results = vt1.get("paper_results", [])

                    def _bar(n, tp):
                        p = round(n / max(tp, 1) * 100)
                        filled = round(p / 10)
                        return "█" * filled + "░" * (10 - filled) + f" {p}% ({n}/{tp})"

                    field_rows = "\n".join(
                        f"| {k.replace('_',' ').title()} "
                        f"| {_bar(v['correct'], total_p)} "
                        f"| {v['incorrect']} differ "
                        f"| {v['missing']} missing |"
                        for k, v in fs.items()
                    )
                    differ_papers = [p for p in paper_results if p["counts"]["incorrect"] > 0]

                    stats_md = (
                        f"**Table 1 — {total_p} papers checked and patched from published SR**\n"
                        f"_{patched} updated · {unmatched_n} PDFs not matched by author+year_\n\n"
                        f"| Field | Correct | Minor diff | Missing |\n|---|---|---|---|\n{field_rows}\n\n---\n\n"
                    )
                    # Build LLM prompt from the stats
                    field_summary_text = "\n".join(
                        f"- {k.replace('_',' ').title()}: {v['correct']}/{total_p} correct, "
                        f"{v['incorrect']} differ, {v['missing']} missing"
                        for k, v in fs.items()
                    )
                    differ_text = ""
                    if differ_papers:
                        differ_text = "\n".join(
                            f"- {p['paper_id'].split('-')[0]}: " +
                            "; ".join(f"{f}='{d['db']}' (should be '{d['correct']}')"
                                      for f, d in p["fields"].items() if d["status"] == "incorrect")
                            for p in differ_papers[:8]
                        )

                    crafted_q = (
                        f"We checked {total_p} physiotherapy cost-effectiveness papers "
                        f"against the published systematic review Table 1. Results:\n\n"
                        f"{field_summary_text}\n\n"
                        + (f"Papers with minor formatting differences:\n{differ_text}\n\n" if differ_text else "")
                        + f"{unmatched_n} PDFs could not be matched (no PDF available for those papers).\n\n"
                        f"Please explain: (1) how accurate our Table 1 extraction is overall, "
                        f"(2) what the minor differences mean (e.g. 'USA' vs 'US'), "
                        f"(3) whether these differences affect research conclusions, "
                        f"(4) what we can reliably answer about study design, perspective, and outcomes."
                    )

                    # Single placeholder — stats + streaming explanation, no double render
                    placeholder = st.empty()
                    placeholder.markdown(stats_md + "_Analysing extraction quality…_")
                    explanation = ""
                    for meta, chunk in _stream_ask({
                        "question": crafted_q,
                        "top_k": 0,
                        "paper_id": None,
                        "history": [],
                    }):
                        explanation += chunk
                        placeholder.markdown(stats_md + explanation + "◌")
                    placeholder.markdown(stats_md + explanation)
                    answer = stats_md + explanation

                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                # Individual paper — validate vs systematic review
                with st.spinner("Cross-checking with systematic review…"):
                    val = _get(f"/validate_vs_review/{active_paper_id}")
                    answer = val.get("validation", val.get("error", "Could not validate."))
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        elif mode == "Compare papers":
            if not active_paper_id or not st.session_state.paper_b:
                answer = "Please select both Paper A and Paper B in the sidebar."
                st.markdown(answer)
            else:
                with st.spinner("Comparing papers…"):
                    data = _post("/compare_papers", {
                        "paper_id_a": active_paper_id,
                        "paper_id_b": st.session_state.paper_b,
                        "question": user_q,
                    })
                answer = data.get("answer", "No answer.") if "error" not in data else f"⚠️ {data['error']}"
                st.markdown(answer)

        elif use_compare:
            with st.spinner("Thinking…"):
                data = _post("/ask_compare", {"question": user_q})
            answer = (
                data.get("answer", "No answer returned.")
                if "error" not in data else f"⚠️ Error: {data['error']}"
            )
            st.markdown(answer)

        else:
            # Single-paper: stream with confidence indicator
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_")
            conf_placeholder = st.empty()
            conf_meta = {}

            for meta, chunk in _stream_ask({
                "question": user_q,
                "top_k": int(top_k),
                "paper_id": active_paper_id,
                "history": history,
            }):
                if meta:
                    conf_meta = meta
                answer += chunk
                placeholder.markdown(answer + "◌")

            placeholder.markdown(answer)

            if conf_meta:
                _CONF = {
                    "high":   ("\U0001f7e2", "#d4edda", "#155724"),
                    "medium": ("\U0001f7e1", "#fff3cd", "#856404"),
                    "low":    ("\U0001f534", "#f8d7da", "#721c24"),
                }
                lvl = conf_meta.get("confidence", "medium")
                icon, bg, fg = _CONF.get(lvl, _CONF["medium"])
                n = conf_meta.get("n_relevant", 0)
                conf_placeholder.markdown(
                    f'<div style="background:{bg};color:{fg};padding:3px 10px;'
                    f'border-radius:6px;font-size:0.74rem;display:inline-block;margin-top:4px">'
                    f'{icon} Evidence confidence: <b>{lvl}</b> · {n} relevant chunks</div>',
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })


# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""

""", unsafe_allow_html=True)
