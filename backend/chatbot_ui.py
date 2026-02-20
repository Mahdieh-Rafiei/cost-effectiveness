import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cost-Effectiveness RAG Chat", layout="wide")
st.title("Cost-Effectiveness RAG Chatbot")


# -----------------------------
# Helpers
# -----------------------------
def load_paper_list():
    try:
        r = requests.get(f"{API_BASE}/list_papers", timeout=30)
        r.raise_for_status()
        papers = r.json().get("papers", [])
        return papers if isinstance(papers, list) else []
    except Exception:
        return []


def maybe_reload_papers_if_mode_changed():
    """Reload list when entering single-paper mode (so dropdown is fresh)."""
    prev_mode = st.session_state.get("_prev_mode")
    curr_mode = st.session_state.get("query_mode")
    if prev_mode != curr_mode:
        # entering/leaving mode -> store current
        st.session_state["_prev_mode"] = curr_mode
        if curr_mode == "Single-paper":
            st.session_state["paper_list"] = load_paper_list()


# -----------------------------
# Session state init
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "paper_list" not in st.session_state:
    st.session_state.paper_list = load_paper_list()

if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = ""

if "_prev_mode" not in st.session_state:
    st.session_state["_prev_mode"] = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k chunks (single-paper RAG)", 3, 80, 12)

    st.divider()
    st.subheader("Query mode")
    mode = st.radio(
        "Choose how to query",
        ["Auto", "Single-paper", "Cross-paper"],
        index=0,
        key="query_mode",
        help="Auto routes based on the question. Single-paper uses /ask with one selected paper. Cross-paper uses /ask_compare.",
    )

    maybe_reload_papers_if_mode_changed()

    single_paper_active = (mode == "Single-paper")

    st.divider()
    st.subheader("Single-paper testing")
    st.caption("Use this for your 6 general test questions (active only in Single-paper mode).")

    # Dropdown options (no Clear/Refresh buttons)
    paper_options = st.session_state.paper_list

    # keep selected valid
    if st.session_state.selected_paper and st.session_state.selected_paper not in paper_options:
        st.session_state.selected_paper = ""

    # If no papers available
    if not paper_options:
        st.warning("No papers found. Run /build_ce_table first.")
        selected = ""
        st.selectbox(
            "Paper ID",
            options=["No papers available"],
            index=0,
            disabled=True,
        )
    else:
        # choose selected index
        default_idx = 0
        if st.session_state.selected_paper in paper_options:
            default_idx = paper_options.index(st.session_state.selected_paper)

        selected = st.selectbox(
            "Paper ID",
            options=paper_options,
            index=default_idx,
            key="paper_selectbox",
            disabled=not single_paper_active,
            help="Pick one paper. Changing the dropdown changes the paper used for single-paper search.",
        )
        st.session_state.selected_paper = selected

    paper_filter = st.session_state.selected_paper if single_paper_active else None

    if mode == "Single-paper":
        if paper_filter:
            st.success(f"Selected paper: {paper_filter}")
        else:
            st.warning("Please select a paper.")
    elif mode == "Cross-paper":
        st.info("Cross-paper mode uses all papers. Paper dropdown is disabled.")
    else:
        st.info("Auto mode chooses route automatically. Paper dropdown is disabled.")

    st.divider()
    show_debug = st.checkbox("Show debug rows (cross-paper)", value=False)
    st.caption(
        "Cross-paper questions (Fig4/Fig5, dominant vs dominated, perspective, "
        "condition-specific differences) use /ask_compare."
    )


# -----------------------------
# Chat history
# -----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# -----------------------------
# Chat input
# -----------------------------
user_q = st.chat_input("Ask a question...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                q_lower = user_q.lower().strip()

                compare_keywords = [
                    "fig 4", "figure 4", "fig4",
                    "fig 5", "figure 5", "fig5",
                    "dominant", "dominated",
                    "across studies", "across papers",
                    "difference between", "compare studies",
                    "perspective", "time horizon", "quadrant",
                    "all cost effective interventions",
                    "all cost-effective interventions",
                    "less cost-effective", "less cost effective",
                    "interventions differ", "differ from",
                ]

                # Route logic
                if mode == "Cross-paper":
                    use_compare = True
                elif mode == "Single-paper":
                    use_compare = False
                else:  # Auto
                    use_compare = any(k in q_lower for k in compare_keywords)

                # Guard for Single-paper mode
                if mode == "Single-paper" and not paper_filter:
                    msg = "Single-paper mode is active, but no paper is selected. Please choose a paper in the sidebar."
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.stop()

                # Backend call
                if use_compare:
                    st.caption("Mode: Cross-paper comparison (/ask_compare)")
                    endpoint = f"{API_BASE}/ask_compare"
                    payload = {"question": user_q}
                else:
                    st.caption("Mode: Single-paper RAG (/ask)")
                    endpoint = f"{API_BASE}/ask"
                    payload = {
                        "question": user_q,
                        "top_k": int(top_k),
                        "paper_id": paper_filter,
                    }

                r = requests.post(endpoint, json=payload, timeout=300)
                r.raise_for_status()
                data = r.json()

                answer = data.get("answer", "")

                # Prefix answer in single-paper mode
                if not use_compare and paper_filter:
                    prefix = f"Based on the paper **{paper_filter}**, "
                    if isinstance(answer, str) and answer.strip():
                        # avoid duplicate punctuation weirdness
                        if answer[:1].islower():
                            answer = prefix + answer
                        else:
                            answer = prefix + answer
                    elif not isinstance(answer, str):
                        # fallback for unexpected structured response
                        answer = f"{prefix}I received a structured response instead of a narrative answer."

                if isinstance(answer, str):
                    assistant_text = answer if answer else "No answer returned."
                    st.markdown(assistant_text)
                else:
                    assistant_text = "Structured result returned (no narrative answer yet)."
                    st.warning("Backend returned structured data instead of a narrative answer.")
                    st.json(answer)

                # Debug rows (cross-paper)
                rows = data.get("rows")
                if show_debug and rows:
                    with st.expander("Show structured rows (debug)"):
                        st.json(rows)

                # Sources (single-paper)
                sources = data.get("sources", [])
                if sources:
                    st.divider()
                    st.subheader("Sources")
                    seen = set()
                    for s in sources:
                        key = (s.get("paper_id"), s.get("page"))
                        if key in seen:
                            continue
                        seen.add(key)
                        st.write(f"- **{s.get('paper_id')}**, p.{s.get('page')}")

                if use_compare and st.session_state.selected_paper:
                    st.info("Cross-paper mode ignores the selected paper and uses all papers.")

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

            except Exception as e:
                st.error(f"Error calling backend: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
