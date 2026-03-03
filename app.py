"""
app.py  —  Now Assist  |  ServiceNow AI Assistant
Run: streamlit run app.py
"""

import streamlit as st
from pipeline import run_query, get_collection_info, ingest_pdfs

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Now Assist — ServiceNow AI",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2' fill='%2300c2a8'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Expander (Pipeline route) ── */
details { margin-top: 10px !important; max-width: 400px !important; }
details, details > div { background: #10121d !important; border: 1px solid #1e2235 !important; border-radius: 10px !important; }
details summary { color: #4a5568 !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ── SVG icon library (Lucide-style, inline) ────────────────────────────────────
def icon(path_d, size=16, color="currentColor", stroke_width=1.75, extra_attrs=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;" {extra_attrs}>'
        f'{path_d}</svg>'
    )

ICONS = {
    # App chrome
    "zap":      '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "bot":      '<rect width="18" height="10" x="3" y="11" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/>',
    "user":     '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "globe":    '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "cpu":      '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "search":   '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "upload":   '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
    "sliders":  '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "trash":    '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    "check":    '<polyline points="20 6 9 17 4 12"/>',
    "alert":    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "layers":   '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "message":  '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "chevron":  '<polyline points="6 9 12 15 18 9"/>',
}

def svg(name, size=15, color="#94a3b8"):
    return icon(ICONS[name], size=size, color=color)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "n_results" not in st.session_state: st.session_state.n_results = 10


# ── Helper: build tool/confidence HTML ───────────────────────────────────────
TOOL_META = {
    "vector_db":   {"label": "Vector DB",   "icon": "database", "color": "#06b6d4", "bg": "rgba(6,182,212,0.08)",  "border": "rgba(6,182,212,0.2)"},
    "web_search":  {"label": "Web Search",  "icon": "globe",    "color": "#a78bfa", "bg": "rgba(167,139,250,0.08)","border": "rgba(167,139,250,0.2)"},
    "llm_response":{"label": "LLM",         "icon": "cpu",      "color": "#60a5fa", "bg": "rgba(96,165,250,0.08)", "border": "rgba(96,165,250,0.2)"},
}

def render_badges(tools_used, confidence):
    html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center;">'
    for t in tools_used:
        m = TOOL_META.get(t, {"label": t, "icon": "activity", "color": "#94a3b8", "bg": "rgba(148,163,184,0.08)", "border": "rgba(148,163,184,0.2)"})
        ico = icon(ICONS[m["icon"]], size=12, color=m["color"])
        html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
            f'background:{m["bg"]};border:1px solid {m["border"]};border-radius:20px;'
            f'font-size:0.72rem;font-weight:600;color:{m["color"]};letter-spacing:0.03em;">'
            f'{ico}&nbsp;{m["label"]}</span>'
        )
    # Confidence chip
    if confidence > 0:
        pct = f"{confidence:.0%}"
        if   confidence >= 0.85: c, bg, border = "#4ade80", "rgba(74,222,128,0.08)", "rgba(74,222,128,0.2)"
        elif confidence >= 0.60: c, bg, border = "#fbbf24", "rgba(251,191,36,0.08)",  "rgba(251,191,36,0.2)"
        else:                    c, bg, border = "#f87171", "rgba(248,113,113,0.08)", "rgba(248,113,113,0.2)"
        bar_icon = icon(ICONS["activity"], size=12, color=c)
        html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;'
            f'background:{bg};border:1px solid {border};border-radius:20px;'
            f'font-size:0.72rem;font-weight:600;color:{c};">'
            f'{bar_icon}&nbsp;{pct} confidence</span>'
        )
    html += '</div>'
    return html


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo wordmark
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:0.25rem 0 1.5rem;">
        <div style="background:linear-gradient(135deg,#1e56c5,#06b6d4);border-radius:8px;
                    width:32px;height:32px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            {icon(ICONS['zap'], size=16, color='#ffffff')}
        </div>
        <div>
            <div style="font-weight:700;font-size:0.95rem;color:#e2e8f0;line-height:1.2;">Now Assist</div>
            <div style="font-size:0.7rem;color:#4a5568;line-height:1.2;">ServiceNow AI Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # — New Chat button —
    if st.button("+ New Chat", use_container_width=True, key="new_chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # — Knowledge Base status —
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;font-weight:600;'
        f'color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">'
        f'{svg("database", 12, "#475569")}&nbsp;Knowledge Base</div>',
        unsafe_allow_html=True,
    )
    kb = get_collection_info()
    if kb["exists"]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;'
            f'background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.18);'
            f'border-radius:8px;font-size:0.82rem;color:#4ade80;margin-bottom:1rem;">'
            f'{icon(ICONS["check"], 13, "#4ade80")}&nbsp;&nbsp;{kb["count"]:,} chunks indexed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:8px 12px;'
            f'background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.18);'
            f'border-radius:8px;font-size:0.82rem;color:#f87171;margin-bottom:1rem;">'
            f'{icon(ICONS["alert"], 13, "#f87171")}&nbsp;&nbsp;No knowledge base found</div>',
            unsafe_allow_html=True,
        )

    # — Ingest PDFs —
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;font-weight:600;'
        f'color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">'
        f'{svg("upload", 12, "#475569")}&nbsp;Ingest PDFs</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "PDF upload",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        st.caption(f"{len(uploaded)} file(s) selected")
    if st.button("Build Knowledge Base", use_container_width=True):
        if not uploaded:
            st.warning("Select at least one PDF first.")
        else:
            with st.spinner("Ingesting PDFs…"):
                r = ingest_pdfs(uploaded)
            (st.success(r["message"]) if r["success"] else st.error(r["message"]))
            if r["success"]:
                st.rerun()

    st.divider()

    # — Settings —
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;font-weight:600;'
        f'color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">'
        f'{svg("sliders", 12, "#475569")}&nbsp;Settings</div>',
        unsafe_allow_html=True,
    )
    st.session_state.n_results = st.slider(
        "Context chunks retrieved",
        min_value=3, max_value=20,
        value=st.session_state.n_results,
    )

    st.markdown(
        '<div style="margin-top:2rem;text-align:center;font-size:0.68rem;color:#2a3045;">'
        'Groq &middot; ChromaDB &middot; DuckDuckGo</div>',
        unsafe_allow_html=True,
    )


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #10121d 0%, #121828 100%);
    border: 1px solid #1e2235;
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    position: relative;
    overflow: hidden;
">
  <div style="
      background: linear-gradient(135deg, #1e2f6e, #0c4a6e);
      border-radius: 12px;
      width: 52px; height: 52px;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      box-shadow: 0 0 24px rgba(30,86,197,0.3);
  ">
      {icon(ICONS['zap'], 24, '#60a5fa')}
  </div>
  <div>
      <div style="font-size:1.55rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.4px;line-height:1.2;">
          Now Assist
      </div>
      <div style="font-size:0.85rem;color:#4a5568;margin-top:3px;">
          AI-powered ServiceNow assistant · RAG · Vector Search · Live Web
      </div>
      <div style="
          display:inline-flex;align-items:center;gap:6px;
          margin-top:8px;padding:3px 10px;
          background:rgba(30,86,197,0.12);border:1px solid rgba(30,86,197,0.25);
          border-radius:20px;font-size:0.72rem;color:#60a5fa;font-weight:500;
      ">
          <span style="width:6px;height:6px;background:#60a5fa;border-radius:50%;
                        box-shadow:0 0 6px #60a5fa;display:inline-block;"></span>
          Connected &amp; Ready
      </div>
  </div>
  <!-- decorative glow -->
  <div style="
      position:absolute;top:-60px;right:-60px;
      width:200px;height:200px;
      background:radial-gradient(circle,rgba(30,86,197,0.08) 0%,transparent 70%);
      pointer-events:none;
  "></div>
</div>
""", unsafe_allow_html=True)


# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(f"""
    <div style="
        text-align:center;
        padding:4rem 2rem;
        display:flex;flex-direction:column;align-items:center;gap:0.75rem;
    ">
        <div style="
            background:linear-gradient(135deg,#1e2235,#1a2040);
            border:1px solid #252b3b;border-radius:16px;
            width:56px;height:56px;display:flex;align-items:center;justify-content:center;
        ">
            {icon(ICONS['message'], 26, '#475569')}
        </div>
        <div style="font-size:1.1rem;font-weight:600;color:#475569;margin-top:0.5rem;">
            Ask anything about ServiceNow
        </div>
        <div style="font-size:0.85rem;color:#2e3a52;">
            Concepts &middot; Scripting &middot; Architecture &middot; Live data
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example query buttons (real Streamlit buttons, so they work on click)
    examples = [
        "What is ITSM in ServiceNow?",
        "Write a GlideRecord query for active incidents",
        "Who founded ServiceNow and when?",
        "Explain the CMDB schema in ServiceNow",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()

else:
    # Render chat history
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        if role == "user":
            with st.chat_message("user", avatar=None):
                st.markdown(
                    f'<div style="color:#e2e8f0;font-size:0.9rem;line-height:1.6;">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant", avatar=None):
                st.markdown(
                    f'<div style="color:#e2e8f0;font-size:0.9rem;line-height:1.6;">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                meta = msg.get("meta", {})
                if meta.get("tools_used"):
                    st.markdown(
                        render_badges(meta["tools_used"], meta.get("confidence", 0)),
                        unsafe_allow_html=True,
                    )
                    if meta.get("sub_queries"):
                        with st.expander("Pipeline route", expanded=False):
                            conf  = meta.get("confidence", 0)
                            tools = meta["tools_used"]
                            sqs   = meta["sub_queries"]

                            # Confidence color
                            if conf >= 0.85:  cc = "#4ade80"
                            elif conf >= 0.60: cc = "#fbbf24"
                            else:              cc = "#f87171"

                            # ── Step row helper ──────────────────────
                            def step_row(ico_html, label, value, color="#94a3b8"):
                                return (
                                    f'<div style="display:flex;align-items:flex-start;gap:10px;'
                                    f'padding:8px 10px;border-radius:8px;margin-bottom:4px;'
                                    f'background:#13151f;border:1px solid #1e2235;">'
                                    f'<div style="flex-shrink:0;margin-top:1px;">{ico_html}</div>'
                                    f'<div>'
                                    f'<div style="font-size:0.72rem;font-weight:600;color:{color};'
                                    f'text-transform:uppercase;letter-spacing:0.05em;">{label}</div>'
                                    f'<div style="font-size:0.83rem;color:#94a3b8;margin-top:2px;">{value}</div>'
                                    f'</div></div>'
                                )

                            arrow = f'<div style="text-align:center;color:#1e2235;font-size:0.9rem;margin:2px 0;">│</div>'

                            rows = ""

                            # 1. User Query
                            rows += step_row(
                                svg("message", 13, "#60a5fa"), "User Query",
                                msg.get("content", "—") if i == 0 else "…", "#60a5fa"
                            )
                            rows += arrow

                            # 2. Confidence Score
                            rows += step_row(
                                svg("activity", 13, cc), "Relevance Check",
                                f"Confidence score: {conf:.0%} — query accepted", cc
                            )
                            rows += arrow

                            # 3. Router Decision
                            tool_labels_str = " + ".join(
                                TOOL_META.get(t, {}).get("label", t) for t in tools
                            )
                            rows += step_row(
                                svg("layers", 13, "#a78bfa"), "Router",
                                f"Selected tools: {tool_labels_str}", "#a78bfa"
                            )

                            # 4. Each tool + sub-query
                            for t, sq in zip(tools, sqs):
                                m = TOOL_META.get(t, {"label": t, "icon": "activity", "color": "#94a3b8"})
                                rows += arrow
                                rows += step_row(
                                    svg(m["icon"], 13, m["color"]), m["label"],
                                    sq, m["color"]
                                )

                            # 5. Final synthesis
                            rows += arrow
                            rows += step_row(
                                svg("zap", 13, "#fbbf24"), "Final Synthesis",
                                "Groq LLM combines all context → answer", "#fbbf24"
                            )

                            st.markdown(rows, unsafe_allow_html=True)


# ── CHAT INPUT ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a ServiceNow question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# ── PROCESS PENDING USER MESSAGE ──────────────────────────────────────────────
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query = st.session_state.messages[-1]["content"]

    # Extract conversation history (last 50 messages, ignoring the current query)
    # We only keep 'role' and 'content' to save tokens and avoid passing UI metadata
    raw_history = st.session_state.messages[:-1][-50:]
    chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in raw_history]

    # We use a chat_message block so the spinner appears inside the chat flow
    with st.chat_message("assistant", avatar=None):
        status_text = st.empty()
        
        tool_labels = {
            "vector_db":    "Searching knowledge base",
            "web_search":   "Searching the web",
            "llm_response": "Reasoning with LLM",
            "synthesizing": "Synthesizing final answer",
        }

        def status_cb(tool_name):
            label = tool_labels.get(tool_name, f"Running {tool_name}")
            m = TOOL_META.get(tool_name, {"color": "#94a3b8", "icon": "activity"})
            ico = icon(ICONS.get(m["icon"], ICONS["activity"]), 14, m["color"])
            status_text.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;'
                f'background:#161928;border:1px solid #1e2235;border-radius:10px;'
                f'color:#64748b;font-size:0.83rem;margin:0.5rem 0;">'
                f'{ico}<span style="color:#94a3b8;">{label}…</span></div>',
                unsafe_allow_html=True,
            )

        with st.spinner(""):
            result = run_query(query, chat_history=chat_history, status_cb=status_cb)

        status_text.empty()

        # If answer is a generator (streaming enabled), write it to UI and capture final text
        if hasattr(result["answer"], "__iter__") and not isinstance(result["answer"], str):
            final_answer = st.write_stream(result["answer"])
        else:
            final_answer = result["answer"]
            st.markdown(
                f'<div style="color:#e2e8f0;font-size:0.9rem;line-height:1.6;">{final_answer}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "meta": {
            "tools_used":  result["tools_used"],
            "confidence":  result["confidence"],
            "sub_queries": result["sub_queries"],
        },
    })
    st.rerun()
