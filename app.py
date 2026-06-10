"""
app.py
HEPID — Hybrid Explainable Prompt Injection Defense Framework
═════════════════════════════════════════════════════════════
Streamlit UI aligned with evaluate.py and detector.py pipeline.

Dual-layer protection:
  Layer 1 — sanitize_query()   : user query before retrieval
  Layer 2 — clean_context()    : retrieved document chunks

Metric cards:
  Malicious Blocked / Successfully Sanitized / Sanitization Failed
"""

import streamlit as st
import os
import io
import re
import html as _html
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


from detector import (
    clean_context,
    ml_predict,
    risk_fusion_predict,
    risk_tier,
    sanitize_query,
    get_lime_weights,
    THREAT_CATEGORIES,
    INJECTION_KEYWORDS,
)

try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

PDF_FOLDER = "medical_db"

st.set_page_config(page_title="HEPID — Secure Medical RAG",
                   layout="wide", page_icon="🛡️")

st.markdown("""<style>
.step-bar{display:flex;align-items:center;margin-bottom:2rem;padding:1rem 1.5rem;
  background:#0f172a;border-radius:12px;border:1px solid #1e293b;}
.step-item{display:flex;align-items:center;gap:8px;flex:1;}
.step-circle{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:.85rem;flex-shrink:0;}
.step-circle.done{background:#22c55e;color:#fff;}
.step-circle.active{background:#3b82f6;color:#fff;}
.step-circle.idle{background:#1e293b;color:#64748b;border:1px solid #334155;}
.step-label{font-size:.78rem;color:#94a3b8;}
.step-label.active-lbl{color:#93c5fd;font-weight:600;}
.step-label.done-lbl{color:#86efac;}
.step-sep{width:28px;height:1px;background:#334155;flex-shrink:0;margin:0 4px;}
.section-card{background:#0f172a;border:1px solid #1e293b;border-radius:14px;
  padding:1.5rem 1.5rem 1rem;margin-bottom:1.2rem;}
.card-title{font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:#64748b;margin-bottom:.7rem;}
.section-head{font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:#64748b;margin:1.4rem 0 .5rem 0;}
.sentence-wrap{background:#1c0a0a;border:1px solid #7f1d1d;border-left:3px solid #ef4444;
  border-radius:10px;padding:12px 16px;margin:8px 0;line-height:1.9;font-size:.88rem;}
.span-benign{color:#e2e8f0;}
.span-malicious{background:#7f1d1d;color:#fef2f2;border-radius:4px;padding:1px 5px;
  margin:0 2px;font-weight:700;text-decoration:underline wavy #ef4444;}
.span-label{display:inline-block;background:#ef4444;color:#fff;border-radius:4px;
  padding:1px 6px;font-size:.65rem;font-weight:700;margin-left:4px;vertical-align:middle;}
.fragment-box{background:#0d2010;border:1px dashed #166534;border-radius:8px;
  padding:8px 12px;margin:6px 0;font-size:.82rem;color:#86efac;}
.fragment-label{font-size:.65rem;font-weight:700;color:#4ade80;text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:3px;}
.rewritten-card{background:#071a0f;border:1px solid #14532d;border-left:3px solid #22c55e;
  border-radius:10px;padding:12px 16px;margin:8px 0;}
.rewritten-sentence{color:#86efac;font-size:.88rem;line-height:1.6;}
.blocked-card{background:#1c0a0a;border:1px solid #7f1d1d;border-left:3px solid #ef4444;
  border-radius:10px;padding:12px 16px;margin:8px 0;}
.blocked-text{color:#fca5a5;font-size:.88rem;font-weight:700;}
.metric-box{background:#1e293b;border-radius:10px;padding:14px;text-align:center;
  min-height:78px;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.metric-val{font-size:1.5rem;font-weight:700;line-height:1.2;}
.metric-lbl{color:#94a3b8;font-size:.72rem;margin-top:4px;}
.threat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0;}
.threat-item{border-radius:6px;padding:4px 10px;font-size:.72rem;display:flex;
  align-items:center;gap:6px;}
.threat-hit{background:#1c0a0a;color:#f87171;border:1px solid #7f1d1d;}
.threat-miss{background:#0f172a;color:#475569;border:1px solid #1e293b;}
.chat-user{background:#1e3a5f;border-radius:12px;padding:12px 16px;margin:6px 0;}
.chat-bot{background:#1a3a2a;border-radius:12px;padding:12px 16px;margin:6px 0;}
.context-box{background:#0a0f1a;border:1px solid #1e3a5f;border-radius:10px;
  padding:14px 16px;font-size:.82rem;color:#94a3b8;line-height:1.7;max-height:280px;
  overflow-y:auto;white-space:pre-wrap;font-family:monospace;}
.lime-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:6px;}
.lime-table th{color:#64748b;font-weight:600;font-size:.68rem;text-transform:uppercase;
  letter-spacing:.08em;padding:4px 8px;border-bottom:1px solid #1e293b;text-align:left;}
.lime-table td{padding:4px 8px;color:#cbd5e1;}
.lime-table tr:nth-child(even) td{background:#0f172a;}
.pos{color:#ef4444!important;} .neg{color:#22c55e!important;}
</style>""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
DEFAULTS = dict(
    step=1, user_prompt="", chat_history=[],
    current_doc=None, current_doc_name=None,
    reconstructed=None, scan_done=False,
    malicious_items=[], rewrite_map={},
    query_decision=None, safe_query=None,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# PDF HELPERS
# =========================================================
def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(p.get_text() for p in doc)

def extract_text_from_path(path: str) -> str:
    doc = fitz.open(path)
    return "".join(p.get_text() for p in doc)

def generate_pdf(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)
    styles = getSampleStyleSheet()
    story  = []
    for para in text.split("\n"):
        if para.strip():
            story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 8))
    doc.build(story or [Paragraph("No content.", styles["Normal"])])
    buf.seek(0)
    return buf

def retrieve_pdf(query: str):
    if not os.path.exists(PDF_FOLDER):
        return None, None
    GENERIC = {"virus","viral","disease","infection","patient","patients","medical",
               "clinical","treatment","symptoms","syndrome","fever","acute","chronic",
               "data","report","case","cases","health","care","the","and","for"}
    qwords = [w for w in query.lower().split() if len(w) > 2 and w not in GENERIC]
    if not qwords:
        return None, None
    best_score = best_name = best_text = 0, None, None
    for fname in os.listdir(PDF_FOLDER):
        if not fname.endswith(".pdf"):
            continue
        try:
            text = extract_text_from_path(os.path.join(PDF_FOLDER, fname))
        except Exception:
            continue
        tl = text.lower(); fl = fname.lower(); score = 0; hit = False
        for w in qwords:
            if w in fl:              score += 3; hit = True
            elif re.search(r'\b' + re.escape(w) + r'\b', tl): score += 2; hit = True
            elif len(w) >= 5 and any(w in t or t in w for t in tl.split() if len(t) > 4):
                score += 1; hit = True
        if hit and score > best_score[0]:
            best_score = score, fname, text
    return (best_score[1], best_score[2]) if best_score[0] >= 3 else (None, None)


# =========================================================
# LIME CHART
# =========================================================
def render_lime_chart(lime_weights: list):
    if not lime_weights:
        st.caption("No LIME weights.")
        return
    words   = [p[0] for p in lime_weights]
    weights = [p[1] for p in lime_weights]
    colors  = ["#ef4444" if w > 0 else "#22c55e" for w in weights]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(words, weights, color=colors, height=0.58)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + (0.002 if w >= 0 else -0.002),
                bar.get_y() + bar.get_height()/2,
                f"{w:+.4f}", va="center", ha="left" if w >= 0 else "right",
                color="#fca5a5" if w > 0 else "#86efac", fontsize=8.5, fontweight="bold")
    ax.axvline(0, color="#475569", linestyle="--", lw=0.9)
    ax.set_xlabel("Contribution weight", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.set_facecolor("#0f172a"); fig.patch.set_facecolor("#0f172a")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#334155")
    plt.tight_layout(pad=0.8)
    st.pyplot(fig); plt.close(fig)
    rows = "".join(
        f"<tr><td><b>{w}</b></td>"
        f"<td class='{'pos' if v>0 else 'neg'}'>{v:+.6f}</td>"
        f"<td class='{'pos' if v>0 else 'neg'}'>{'→ Malicious' if v>0 else '→ Benign'}</td></tr>"
        for w, v in lime_weights)
    st.markdown(f"<table class='lime-table'><tr><th>Token</th><th>Weight</th>"
                f"<th>Signal</th></tr>{rows}</table>", unsafe_allow_html=True)


# =========================================================
# SPAN SENTENCE RENDERER
# =========================================================
def render_span_sentence(item: dict):
    sentence  = item["sentence"]
    rs        = item["risk_score"]
    tier      = item["tier"]
    span_info = item.get("span_info") or {}
    details   = item.get("threat_details", {})
    rrs       = item.get("rescore_risk")
    rt        = item.get("rescore_tier")

    tier_colors = {"Malicious":"#ef4444","Suspicious":"#fbbf24","Benign":"#4ade80"}
    tc = tier_colors.get(tier, "#fbbf24")

    if span_info.get("span_text") and span_info.get("span_start") is not None:
        s   = span_info["span_start"]; e = span_info["span_end"]
        bef = _html.escape(sentence[:s])
        spn = _html.escape(sentence[s:e])
        aft = _html.escape(sentence[e:])
        det = span_info.get("detection_type","ml")
        highlighted = (f"<span class='span-benign'>{bef}</span>"
                       f"<span class='span-malicious'>{spn}"
                       f"<span class='span-label'>INJECTION</span></span>"
                       f"<span class='span-benign'>{aft}</span>")
        det_badge = (f"<span style='background:#1e1a0a;color:#fbbf24;border-radius:4px;"
                     f"padding:1px 8px;font-size:.65rem;margin-left:6px'>"
                     f"{'keyword' if det=='keyword' else 'ML-semantic'}</span>")
    else:
        highlighted = f"<span class='span-malicious'>{_html.escape(sentence)}<span class='span-label'>ML DETECTED</span></span>"
        det_badge   = "<span style='background:#172033;color:#93c5fd;border-radius:4px;padding:1px 8px;font-size:.65rem;margin-left:6px'>ML-semantic</span>"

    st.markdown(
        f"<div class='sentence-wrap'>{highlighted}<br>"
        f"<span style='background:#7f1d1d;color:#fca5a5;border-radius:6px;"
        f"padding:2px 9px;font-size:.72rem;font-weight:700;margin-top:6px'>"
        f"Risk: {rs:.4f}</span>"
        f"<span style='background:#1c0a0a;color:{tc};border-radius:4px;"
        f"padding:1px 8px;font-size:.65rem;font-weight:700;margin-left:6px'>{tier}</span>"
        f"{det_badge}</div>",
        unsafe_allow_html=True)

    # Clean fragment
    frag = span_info.get("clean_fragment","")
    if frag:
        st.markdown(f"<div class='fragment-box'>"
                    f"<div class='fragment-label'>Clean fragment (keyword spans removed)</div>"
                    f"{_html.escape(frag)}</div>", unsafe_allow_html=True)

    # Threat categories
    if details:
        cats = "<div class='threat-grid'>"
        for cat, hit in details.items():
            cls = "threat-hit" if hit else "threat-miss"
            cats += f"<div class='threat-item {cls}'>{'✓' if hit else '–'} {cat.replace('_',' ').title()}</div>"
        cats += "</div>"
        st.markdown(cats, unsafe_allow_html=True)

    # Re-score row
    if rrs is not None:
        rc = tier_colors.get(rt, "#fbbf24")
        st.markdown(
            f"<div style='font-size:.78rem;color:#94a3b8;margin:6px 0'>"
            f"Risk before: <b style='color:{tc}'>{rs:.4f}</b>"
            f" → After sanitization: <b style='color:{rc}'>{rrs:.4f}</b>"
            f" <span style='background:#0f172a;color:{rc};border-radius:4px;"
            f"padding:1px 8px;font-size:.65rem;font-weight:700;margin-left:4px'>{rt}</span>"
            f"</div>",
            unsafe_allow_html=True)


# =========================================================
# GEMINI Q&A
# =========================================================
from google import genai as _genai_qa
from google.genai import types as _genai_types_qa

def _gemini_client():
    key = os.environ.get("GEMINI_API_KEY","")
    return _genai_qa.Client(api_key=key) if key else None

def extract_relevant_context(question: str, full_text: str, max_chars: int = 400) -> str:
    stop = {"what","is","are","the","a","an","of","in","on","for","to","and","or",
            "how","why","when","where","does","do","did","was","were","this","that",
            "these","those","with","from","at","by","about"}
    kws  = [w.lower() for w in re.findall(r'\b\w{3,}\b', question) if w.lower() not in stop]
    chunks = [c.strip() for c in re.split(r'\n{2,}|(?<=[.!?])\s+', full_text) if len(c.strip()) > 40]
    scored = sorted(chunks, key=lambda c: sum(1 for kw in kws if kw in c.lower()), reverse=True)
    selected, total = [], 0
    for chunk in scored:
        if total + len(chunk) > max_chars: break
        selected.append(chunk); total += len(chunk)
    return "\n\n".join(selected) if selected else full_text[:max_chars]

def gemini_answer(question: str, context: str, history: list) -> str:
    client = _gemini_client()
    if not client:
        return "No GEMINI_API_KEY found."
    ctx  = extract_relevant_context(question, context)
    hist = ""
    for t in history[-2:]:
        hist += f"{'User' if t['role']=='user' else 'Assistant'}: {t['content'][:150]}\n\n"
    prompt = (
        "You are a secure medical information assistant. "
        "Answer using ONLY the context below. Be concise.\n\n"
        f"CONTEXT:\n{ctx}\n\n{hist}User: {question}\nAssistant:"
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config=_genai_types_qa.GenerateContentConfig(
                max_output_tokens=512, temperature=0.2))
        return resp.text.strip()
    except Exception as e:
        err = str(e)
        m = re.search(r'retry in ([\d.]+)s', err)
        wait = f" Retry in ~{int(float(m.group(1)))+1}s." if m else ""
        return f"Quota exceeded.{wait}\n\nWait and retry, or use a paid API key."


# =========================================================
# STEP INDICATOR
# =========================================================
def render_steps(current: int):
    steps = [(1,"Query + Upload"),(2,"Results & Chat")]
    html  = "<div class='step-bar'>"
    for i,(num,label) in enumerate(steps):
        if num < current:   circ,lbl,icon = "done","done-lbl","✓"
        elif num == current: circ,lbl,icon = "active","active-lbl",str(num)
        else:                circ,lbl,icon = "idle","",str(num)
        html += (f"<div class='step-item'><div class='step-circle {circ}'>{icon}</div>"
                 f"<span class='step-label {lbl}'>{label}</span></div>")
        if i < len(steps)-1: html += "<div class='step-sep'></div>"
    st.markdown(html + "</div>", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("HEPID Framework")
st.sidebar.caption("Hybrid Explainable Prompt Injection Defense")
st.sidebar.divider()
st.sidebar.markdown("### Medical PDF Database")
pdf_files = []
if os.path.exists(PDF_FOLDER):
    try: pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    except: pass
if pdf_files:
    sel = st.sidebar.selectbox("Available PDFs", pdf_files)
    if sel:
        try:
            prev = extract_text_from_path(os.path.join(PDF_FOLDER, sel))
            st.sidebar.text_area("Preview", prev[:800], height=180)
        except: pass
else:
    st.sidebar.info("No PDFs in medical_db/")
st.sidebar.divider()
st.sidebar.markdown("### Settings")
run_lime = st.sidebar.toggle("Enable LIME Explainability", value=True,
    help="Token-level weights for display only. Sanitization uses keyword removal.")
st.sidebar.divider()
st.sidebar.markdown("### Risk Thresholds")
st.sidebar.caption("Benign   : risk < 0.50")
st.sidebar.caption("Suspicious: 0.50 – 0.80")
st.sidebar.caption("Malicious: risk > 0.80")
st.sidebar.caption("Fusion: 0.7 × DistilBERT + 0.3 × Threat Score")
st.sidebar.divider()
if st.sidebar.button("Start Over", use_container_width=True):
    for k, v in DEFAULTS.items(): st.session_state[k] = v
    st.rerun()


# =========================================================
# MAIN
# =========================================================
st.title("🛡️ HEPID — Secure Medical RAG System")
st.caption("Risk Score Fusion · Keyword Span Removal · LIME Explainability · Dual-Layer Protection")
st.divider()
render_steps(st.session_state.step)


# =========================================================
# STEP 1
# =========================================================
if st.session_state.step == 1:
    col_form, col_info = st.columns([3,2], gap="large")

    with col_form:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Your question</div>", unsafe_allow_html=True)
        prompt_val = st.text_area("prompt", value=st.session_state.user_prompt,
            placeholder="e.g. What are the transmission routes of hantavirus?",
            height=110, label_visibility="collapsed", key="prompt_textarea")
        st.markdown("<div class='card-title' style='margin-top:.8rem'>Quick prompts</div>",
                    unsafe_allow_html=True)
        chips = ["What are the key symptoms?","Summarise treatment options",
                 "Explain the transmission route","What viruses are discussed?",
                 "What is the case fatality rate?","Explain this document"]
        for i, chip in enumerate(chips):
            with st.columns(3)[i % 3]:
                if st.button(chip, key=f"chip_{i}", use_container_width=True):
                    st.session_state.user_prompt = chip; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Upload PDF document</div>", unsafe_allow_html=True)
        st.caption("Upload your own PDF or leave blank — system auto-searches medical_db.")
        uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        if not uploaded_file:
            st.info(f"No file — will auto-search medical_db "
                    f"({'  '.join(pdf_files[:5]) + ('…' if len(pdf_files)>5 else '') if pdf_files else 'empty'}).")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Scan & Sanitise", use_container_width=True, type="primary"):
            final_prompt = prompt_val.strip() or st.session_state.user_prompt.strip()
            if not final_prompt:
                st.warning("Please enter a question first."); st.stop()

            st.session_state.user_prompt = final_prompt
            text = name = None

            if uploaded_file:
                with st.spinner("Extracting text…"):
                    text = extract_text_from_pdf(uploaded_file)
                    name = uploaded_file.name
            else:
                with st.spinner(f"Searching medical_db for: {final_prompt}…"):
                    name, text = retrieve_pdf(final_prompt)
                if not text:
                    st.error("No matching document found. Upload a PDF or rephrase."); st.stop()
                st.success(f"Found: {name}")

            # Reset state
            for k in ["scan_done","chat_history","reconstructed","malicious_items",
                      "rewrite_map","query_decision","safe_query"]:
                st.session_state[k] = [] if k in ["chat_history","malicious_items"] else \
                                       {} if k == "rewrite_map" else None
            st.session_state.current_doc      = text
            st.session_state.current_doc_name = name

            # LAYER 1: sanitize query
            with st.spinner("Layer 1 — Scanning query for injections…"):
                q_dec = sanitize_query(final_prompt)
            st.session_state.query_decision = q_dec

            if not q_dec["safe"]:
                spans_str = ", ".join(f"`{s}`" for s in q_dec.get("spans_removed",[]))
                st.error(
                    f"**Query blocked by HEPID Layer 1**\n\n"
                    f"Tier: **{q_dec['tier']}**  |  Risk: **{q_dec['risk_score']:.4f}**\n\n"
                    f"Spans detected: {spans_str or 'none (pure injection)'}\n\n"
                    "No safe query could be extracted. Please rephrase.")
                st.stop()

            safe_q = q_dec["safe_query"]
            st.session_state.safe_query = safe_q

            # LAYER 2: scan document
            with st.spinner("Layer 2 — Scanning document for injections…"):
                malicious, cleaned = clean_context(text, run_lime=run_lime)

            st.session_state.malicious_items = malicious
            st.session_state.rewrite_map     = {i["sentence"]: i.get("rewritten","") for i in malicious}
            st.session_state.reconstructed   = cleaned
            st.session_state.scan_done       = True

            n_inj = len(malicious)
            q_note = f"\n\nQuery sanitized: `{final_prompt}` → `{safe_q}`" if q_dec["was_sanitized"] else ""
            st.session_state.chat_history.append({"role":"assistant","content":(
                f"**Document loaded:** `{name}`\n\n"
                f"**Layer 1 (Query):** {'Injection removed — query sanitized.' if q_dec['was_sanitized'] else 'Clean.'}"
                f"{q_note}\n\n"
                f"**Layer 2 (Document):** "
                f"{'**'+str(n_inj)+' injection(s) detected and sanitised.**' if n_inj else 'No injections found.'}\n\n"
                "Ask me anything about the sanitised document.")})
            st.session_state.step = 2
            st.rerun()

    with col_info:
        st.markdown("#### How it works")
        st.markdown("""
1. **Enter question** or pick a quick prompt.
2. **Upload PDF** or auto-search `medical_db`.
3. **Layer 1** — user query scanned: `risk = 0.7×DistilBERT + 0.3×ThreatScore`
4. **Layer 2** — document sentences scanned with same pipeline:
   - Benign (< 0.50) → direct pass
   - Suspicious (0.50–0.80) → keyword removal → re-score → keep if benign
   - Malicious (> 0.80) → hard block
5. **Safe Q&A** — Gemini answers from sanitised document only.
""")
        if pdf_files:
            st.markdown("#### Available in database")
            for f in pdf_files: st.markdown(f"- `{f}`")


# =========================================================
# STEP 2
# =========================================================
elif st.session_state.step == 2:
    malicious     = st.session_state.malicious_items
    rewrite_map   = st.session_state.rewrite_map
    reconstructed = st.session_state.reconstructed
    name          = st.session_state.current_doc_name
    n_inj         = len(malicious)
    q_dec         = st.session_state.get("query_decision")
    safe_q_disp   = st.session_state.get("safe_query") or st.session_state.user_prompt

    # ── Query sanitization banner ─────────────────────────────────────────────
    if q_dec and q_dec.get("was_sanitized"):
        orig      = st.session_state.user_prompt
        spans     = q_dec.get("spans_removed", [])
        orig_rs   = q_dec.get("risk_score", 0)
        new_rs    = q_dec.get("rescore_risk") or 0
        orig_tier = q_dec.get("tier","")
        new_tier  = q_dec.get("rescore_tier","Benign")
        tc = {"Malicious":"#ef4444","Suspicious":"#fbbf24","Benign":"#4ade80"}

        # Highlight removed spans in original
        highlighted = _html.escape(orig)
        for span in spans:
            highlighted = re.sub(
                re.escape(_html.escape(span)),
                f"<span style='background:#7f1d1d;color:#fef2f2;border-radius:3px;"
                f"padding:1px 5px;font-weight:700;text-decoration:line-through'>"
                f"{_html.escape(span)}</span>",
                highlighted, flags=re.IGNORECASE)

        pills = " ".join(
            f"<code style='background:#1e293b;color:#fca5a5;border-radius:3px;"
            f"padding:1px 6px'>{_html.escape(s)}</code>" for s in spans)
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #334155;border-radius:12px;"
            f"padding:14px 18px;margin-bottom:14px'>"
            f"<div style='font-size:.65rem;color:#ef4444;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.1em;margin-bottom:10px'>Layer 1 — Query Injection Detected &amp; Removed</div>"
            f"<div style='font-size:.82rem;margin-bottom:8px;line-height:1.8'>"
            f"<span style='color:#64748b;font-size:.7rem;text-transform:uppercase;"
            f"letter-spacing:.08em'>Original query</span><br>"
            f"{highlighted} "
            f"<span style='background:#1c0a0a;color:{tc.get(orig_tier,'#fbbf24')};"
            f"border-radius:4px;padding:1px 8px;font-size:.65rem;font-weight:700;margin-left:4px'>"
            f"{orig_tier} {orig_rs:.3f}</span></div>"
            f"<div style='border-top:1px solid #1e293b;margin:8px 0'></div>"
            f"<div style='font-size:.82rem;line-height:1.8'>"
            f"<span style='color:#64748b;font-size:.7rem;text-transform:uppercase;"
            f"letter-spacing:.08em'>Safe query</span><br>"
            f"<span style='color:#86efac;font-weight:500'>{_html.escape(safe_q_disp)}</span> "
            f"<span style='background:#052e16;color:#4ade80;border-radius:4px;"
            f"padding:1px 8px;font-size:.65rem;font-weight:700;margin-left:4px'>"
            f"BENIGN {new_rs:.3f}</span></div>"
            f"{'<div style=margin-top:8px;font-size:.72rem;color:#475569>Removed: ' + pills + '</div>' if pills else ''}"
            f"</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:#0f172a;border:1px solid #1e3a5f;border-left:3px solid #3b82f6;"
            f"border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:.88rem;color:#93c5fd'>"
            f"Query: {_html.escape(st.session_state.user_prompt)}"
            f"  |  Document: {name or 'unknown'}</div>",
            unsafe_allow_html=True)

    col_reset, _ = st.columns([1,6])
    with col_reset:
        if st.button("← Start Over"):
            for k,v in DEFAULTS.items(): st.session_state[k] = v
            st.rerun()
    st.divider()

    # ── Metric cards ──────────────────────────────────────────────────────────
    n_malicious = sum(1 for i in malicious if i["tier"] == "Malicious")
    n_sanitized = sum(1 for i in malicious
                      if i["tier"] == "Suspicious" and i.get("rescore_tier") == "Benign")
    n_failed    = sum(1 for i in malicious
                      if i["tier"] == "Suspicious" and i.get("rescore_tier") != "Benign")
    doc_words   = len((st.session_state.current_doc or "").split())

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label, color in [
        (c1, "UNSAFE" if n_inj else "SAFE", "Security Status", "#ef4444" if n_inj else "#22c55e"),
        (c2, str(n_malicious),  "Malicious Blocked",      "#ef4444"),
        (c3, str(n_sanitized),  "Successfully Sanitized", "#22c55e"),
        (c4, str(n_failed),     "Sanitization Failed",    "#f97316"),
        (c5, f"{doc_words:,}",  "Words in Document",      "#60a5fa"),
    ]:
        with col:
            st.markdown(f"<div class='metric-box'><div class='metric-val' style='color:{color}'>"
                        f"{val}</div><div class='metric-lbl'>{label}</div></div>",
                        unsafe_allow_html=True)
    st.markdown("")

    left, right = st.columns([1,1], gap="large")

    # ── LEFT: Detection Results ───────────────────────────────────────────────
    with left:
        if malicious:
            st.markdown("<div class='section-head'>Detection Results (Risk Score Fusion)</div>",
                        unsafe_allow_html=True)
            st.caption("risk = 0.7 × DistilBERT + 0.3 × Threat Score. "
                       "Keyword removal then re-score for Suspicious tier.")
            for idx, item in enumerate(malicious):
                tier = item["tier"]; rs = item["risk_score"]
                rw   = item.get("rewritten","")
                is_blocked = rw.startswith("[BLOCKED")
                with st.expander(f"Sentence {idx+1} — {tier}  risk {rs:.4f}", expanded=(idx==0)):
                    render_span_sentence(item)
                    lw = item.get("lime_weights",[])
                    if lw:
                        st.markdown("**LIME token importance (display only):**")
                        render_lime_chart(lw)
                    elif not run_lime:
                        st.caption("Enable LIME in sidebar for token weights.")
        else:
            st.success("✅ No injections detected — document is clean.")

        # Rewriting comparison
        if rewrite_map:
            st.markdown(f"<div class='section-head'>Sanitization Results ({len(rewrite_map)} sentences)</div>",
                        unsafe_allow_html=True)
            st.caption("Keyword injection spans removed. Re-scored below 0.50 = Sanitized. "
                       "Still risky = Blocked.")
            for orig, rw in rewrite_map.items():
                si          = next((i.get("span_info") for i in malicious if i["sentence"]==orig), None)
                is_blocked  = rw.startswith("[BLOCKED")
                c_a, c_b    = st.columns(2)
                with c_a:
                    st.markdown(
                        f"<div style='font-size:.66rem;color:#ef4444;font-weight:700;"
                        f"margin-bottom:4px'>ORIGINAL (with injection)</div>",
                        unsafe_allow_html=True)
                    if si and si.get("span_text"):
                        s = si["span_start"]; e = si["span_end"]
                        bef = _html.escape(orig[:s])
                        spn = _html.escape(orig[s:e])
                        aft = _html.escape(orig[e:])
                        st.markdown(
                            f"<div class='sentence-wrap' style='font-size:.82rem'>"
                            f"{bef}<span class='span-malicious'>{spn}</span>{aft}</div>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='sentence-wrap' style='font-size:.82rem'>"
                                    f"{_html.escape(orig)}</div>", unsafe_allow_html=True)
                with c_b:
                    label_color = "#ef4444" if is_blocked else "#22c55e"
                    label_txt   = "BLOCKED" if is_blocked else "SANITISED"
                    card_cls    = "blocked-card" if is_blocked else "rewritten-card"
                    text_cls    = "blocked-text" if is_blocked else "rewritten-sentence"
                    st.markdown(
                        f"<div style='font-size:.66rem;color:{label_color};"
                        f"font-weight:700;margin-bottom:4px'>{label_txt}</div>",
                        unsafe_allow_html=True)
                    st.markdown(f"<div class='{card_cls}'><div class='{text_cls}'>"
                                f"{_html.escape(rw)}</div></div>", unsafe_allow_html=True)
                    mi = next((i for i in malicious if i["sentence"]==orig), None)
                    if mi and mi.get("rescore_risk") is not None:
                        rrs = mi["rescore_risk"]; rt = mi.get("rescore_tier","")
                        rc  = {"Benign":"#4ade80","Suspicious":"#fbbf24","Malicious":"#ef4444"}.get(rt,"#fbbf24")
                        st.markdown(
                            f"<div style='font-size:.72rem;color:#64748b;margin-top:4px'>"
                            f"Re-score: <span style='color:{rc};font-weight:700'>"
                            f"{rrs:.4f} ({rt})</span></div>",
                            unsafe_allow_html=True)

        # Document context
        st.markdown("<div class='section-head'>Document Context Sent to LLM</div>",
                    unsafe_allow_html=True)
        preview = (reconstructed or "")[:3000]
        st.markdown(f"<div class='context-box'>{preview}"
                    f"{'…' if len(reconstructed or '')>3000 else ''}</div>",
                    unsafe_allow_html=True)
        if reconstructed and reconstructed.strip():
            st.download_button("📄 Download Sanitised PDF",
                data=generate_pdf(reconstructed),
                file_name=f"sanitised_{name or 'document'}.pdf",
                mime="application/pdf", use_container_width=True)

    # ── RIGHT: Chat ───────────────────────────────────────────────────────────
    with right:
        st.markdown("### 💬 Ask About the Document")
        st.markdown(
            f"<div style='display:inline-flex;align-items:center;gap:6px;"
            f"background:#1e293b;border:1px solid #334155;border-radius:20px;"
            f"padding:4px 12px;font-size:.75rem;color:#94a3b8;margin-bottom:10px'>"
            f"📄 {name or 'Unknown'}"
            f"{'  |  🔴 '+str(n_inj)+' sanitised' if n_inj else '  |  🟢 Clean'}"
            f"</div>",
            unsafe_allow_html=True)

        with st.container(height=380):
            for msg in st.session_state.chat_history:
                cls  = "chat-bot" if msg["role"]=="assistant" else "chat-user"
                icon = "AI" if msg["role"]=="assistant" else "You"
                st.markdown(f"<div class='{cls}'><b>{icon}:</b> {msg['content']}</div>",
                            unsafe_allow_html=True)

        # Quick questions — always use SAFE query as first button
        safe_initial = st.session_state.get("safe_query") or st.session_state.user_prompt
        quick_qs = ["Explain this document","What are the key symptoms?",
                    "Summarise treatment options","What is the transmission route?"]
        if safe_initial not in quick_qs:
            quick_qs.insert(0, safe_initial)

        q_cols = st.columns(2)
        for i, q in enumerate(quick_qs[:6]):
            with q_cols[i % 2]:
                if st.button(q[:50], key=f"quick_{i}", use_container_width=True):
                    ctx   = reconstructed or st.session_state.current_doc
                    q_res = sanitize_query(q)
                    if not q_res["safe"]:
                        st.warning(f"Blocked (Layer 1). Risk: {q_res['risk_score']:.3f}")
                    else:
                        sq       = q_res["safe_query"]
                        disp     = q if not q_res["was_sanitized"] else f"{q} *(→ `{sq}`)*"
                        st.session_state.chat_history.append({"role":"user","content":disp})
                        with st.spinner("Thinking…"):
                            ans = gemini_answer(sq, ctx, st.session_state.chat_history[:-1])
                        st.session_state.chat_history.append({"role":"assistant","content":ans})
                        st.rerun()

        with st.form("chat_form", clear_on_submit=True):
            user_q = st.text_input("Ask a question…",
                placeholder="e.g. What viruses cause HFRS?",
                label_visibility="collapsed")
            send = st.form_submit_button("Send ➤", use_container_width=True)

        if send and user_q.strip():
            ctx   = reconstructed or st.session_state.current_doc
            q_res = sanitize_query(user_q)
            if not q_res["safe"]:
                st.warning(f"Query blocked (Layer 1). Risk: {q_res['risk_score']:.3f} | "
                           f"Tier: {q_res['tier']}. Please rephrase.")
            else:
                sq   = q_res["safe_query"]
                disp = user_q if not q_res["was_sanitized"] else \
                       f"{user_q} *(sanitized → `{sq}`)*"
                st.session_state.chat_history.append({"role":"user","content":disp})
                with st.spinner("Thinking…"):
                    ans = gemini_answer(sq, ctx, st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role":"assistant","content":ans})
                st.rerun()