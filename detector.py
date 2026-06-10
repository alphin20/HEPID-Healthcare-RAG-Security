"""
detector.py
HEPID — Hybrid Explainable Prompt Injection Defense Framework
═════════════════════════════════════════════════════════════
Exact mirror of evaluate.py pipeline.

Every constant, function, and decision rule is copied verbatim
from evaluate.py so the live app and offline evaluation are identical.
"""

import os
import re
import logging
import numpy as np
import torch

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from google import genai as _genai
from google.genai import types as _genai_types

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="injection_log.txt", level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# CONFIG  — identical to evaluate.py
# =========================================================
DISTILBERT_PATH    = "/content/drive/MyDrive/major_project/ckpt"
ML_THRESHOLD       = 0.85
DISTILBERT_MAX_LEN = 512
W_ML               = 0.7
W_THREAT           = 0.3
RISK_MALICIOUS     = 0.80
RISK_SUSPICIOUS    = 0.50
LIME_NUM_FEATURES  = 10
LIME_NUM_SAMPLES   = 300   # same as evaluate.py
SPAN_THRESHOLD     = 0.05  # same as evaluate.py SPAN_REMOVAL_THRESHOLD

# =========================================================
# THREAT INDICATOR CATEGORIES — identical to evaluate.py
# =========================================================
THREAT_CATEGORIES = {
    "prompt_disclosure": {
        "weight": 0.25,
        "keywords": [
            "reveal your system prompt", "reveal your instructions",
            "reveal your training", "disclose your system prompt",
            "disclose your instructions", "show me your instructions",
            "system prompt", "hidden instructions", "internal prompt",
            "what are your instructions", "tell me your prompt",
        ],
    },
    "role_override": {
        "weight": 0.25,
        "keywords": [
            "ignore previous instructions", "ignore all instructions",
            "ignore your instructions", "ignore guidelines",
            "ignore your training", "forget your instructions",
            "forget previous instructions", "override your",
            "override instructions", "new instructions",
            "you are now", "you are no longer", "act as if you have no",
            "pretend you are", "your true self", "no longer an ai",
            "you have been reprogrammed",
        ],
    },
    "data_exfiltration": {
        "weight": 0.25,
        "keywords": [
            "extract all data", "extract private data", "exfiltrate",
            "leak the", "leak patient", "leak medical", "leak confidential",
            "leak private", "leak all", "dump all", "expose all",
            "patient data", "medical records", "confidential patient data",
        ],
    },
    "jailbreak_intent": {
        "weight": 0.25,
        "keywords": [
            "jailbreak", "dan mode", "developer mode",
            "bypass safety", "bypass restrictions",
            "disable restrictions", "disable safety",
            "no restrictions", "unrestricted mode",
            "do anything now", "roleplay as", "simulate being",
            "prompt injection", "how to kill", "how to harm",
            "privileged admin", "operating in unrestricted",
            "adversarial injection", "adversarial prompt",
            "injected prompt", "embedded injection",
            "adversarial prompts embedded", "contains embedded adversarial",
            "adversarial evaluation", "adversarial attack",
            "injection prompts", "embedded adversarial",
        ],
    },
}

# Flat list for backward-compat
INJECTION_KEYWORDS = [kw for cat in THREAT_CATEGORIES.values() for kw in cat["keywords"]]

# =========================================================
# DEVICE + MODEL
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[HEPID] device={device}")
print("[HEPID] Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH, local_files_only=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_PATH, local_files_only=True).to(device)
model.eval()
print("[HEPID] DistilBERT ready")

_lime_explainer = LimeTextExplainer(class_names=["Benign", "Malicious"])
_lime_cache: dict = {}

# =========================================================
# PRIMITIVES — copied verbatim from evaluate.py
# =========================================================
def ml_predict(text: str) -> tuple:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=DISTILBERT_MAX_LEN,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
    p = probs[0][1].item()
    return (1 if p >= ML_THRESHOLD else 0), p


def threat_indicator_score(text: str) -> tuple:
    tl = text.lower(); score = 0.0; details = {}
    for cat, meta in THREAT_CATEGORIES.items():
        hit = any(kw in tl for kw in meta["keywords"])
        details[cat] = hit
        if hit:
            score += meta["weight"]
    return min(score, 1.0), details


def compute_risk_score(ml_prob: float, threat_score: float) -> float:
    return W_ML * ml_prob + W_THREAT * threat_score


def risk_tier(rs: float) -> str:
    if rs > RISK_MALICIOUS:  return "Malicious"
    if rs > RISK_SUSPICIOUS: return "Suspicious"
    return "Benign"


def risk_fusion_predict(text: str) -> tuple:
    """Detection only. Returns (label, risk_score, threat_details)."""
    _, ml_prob            = ml_predict(text)
    threat_score, details = threat_indicator_score(text)
    rs                    = compute_risk_score(ml_prob, threat_score)
    return (0 if risk_tier(rs) == "Benign" else 1), rs, details


# =========================================================
# KEYWORD SPAN REMOVAL — copied verbatim from evaluate.py
# =========================================================
def keyword_span_removal(text: str) -> tuple:
    all_kws = sorted(
        [kw for cat in THREAT_CATEGORIES.values() for kw in cat["keywords"]],
        key=len, reverse=True
    )
    removed = []; sanitized = text
    for kw in all_kws:
        if kw in sanitized.lower():
            sanitized = re.compile(re.escape(kw), re.IGNORECASE).sub("", sanitized)
            removed.append(kw)
    sanitized = re.sub(
        r'\s+(and|or|also|then|but|plus)\s+(provide|give|show|tell|reveal|expose|send|share|output|dump)\b',
        ' ', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        r'\s+(provide|give|show|tell|reveal|expose|send|share|output|dump)\s*$',
        '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'^\s*(and|or|also|then|but|plus)\s+', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'\s+(and|or|also|then|but|plus)\s*$',  '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'\s{2,}', ' ', sanitized).strip().rstrip(".,; ")
    return sanitized, removed


# =========================================================
# FULL HEPID PREDICT — copied verbatim from evaluate.py
# (used by sanitize_query and clean_context)
# =========================================================
def _lime_proba_batch(texts: list) -> np.ndarray:
    """Batched LIME wrapper."""
    if not texts:
        return np.empty((0, 2))
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=DISTILBERT_MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()
    return probs


def full_hepid_predict(text: str) -> dict:
    """
    Three-tier pipeline — identical to evaluate.py full_hepid_predict.

    Benign     → label 0, direct pass
    Suspicious → keyword removal → LIME fallback → re-score
                 new_rs < RISK_SUSPICIOUS AND spans → label 0 (sanitized)
                 else                               → label 1 (blocked)
    Malicious  → label 1, hard block
    """
    _, ml_prob            = ml_predict(text)
    threat_score, details = threat_indicator_score(text)
    rs                    = compute_risk_score(ml_prob, threat_score)
    tier                  = risk_tier(rs)

    if tier == "Benign":
        return dict(tier=tier, risk_score=rs, label=0,
                    action="Direct Pass", sanitized_text=None,
                    rescore_risk=None, rescore_tier=None,
                    spans_removed=[], threat_details=details)

    if tier == "Malicious":
        return dict(tier=tier, risk_score=rs, label=1,
                    action="Hard Block", sanitized_text=None,
                    rescore_risk=None, rescore_tier=None,
                    spans_removed=[], threat_details=details)

    # Suspicious → keyword removal → LIME fallback → re-score
    sanitized, spans = keyword_span_removal(text)

    # LIME fallback: if keywords found nothing, use LIME high-weight tokens
    if not spans:
        key = f"fb|{text}"
        if key not in _lime_cache:
            exp = _lime_explainer.explain_instance(
                text, _lime_proba_batch,
                num_features=LIME_NUM_FEATURES, num_samples=100)
            _lime_cache[key] = [tok for tok, w in exp.as_list() if w > SPAN_THRESHOLD]
        lime_spans = _lime_cache[key]
        if lime_spans:
            lime_san = text
            for span in lime_spans:
                lime_san = re.compile(re.escape(span), re.IGNORECASE).sub("", lime_san)
            lime_san = re.sub(r"\s{2,}", " ", lime_san).strip().rstrip(".,; ")
            if lime_san and len(lime_san.split()) >= 2:
                sanitized = lime_san
                spans     = lime_spans

    if not sanitized.strip() or len(sanitized.split()) < 2:
        return dict(tier=tier, risk_score=rs, label=1,
                    action="Blocked (empty after removal)",
                    sanitized_text="", rescore_risk=1.0,
                    rescore_tier="Malicious", spans_removed=spans,
                    threat_details=details)

    _, ml2   = ml_predict(sanitized)
    ts2, _   = threat_indicator_score(sanitized)
    new_rs   = compute_risk_score(ml2, ts2)
    new_tier = risk_tier(new_rs)

    if new_rs < RISK_SUSPICIOUS and spans:
        label  = 0
        action = "Sanitized"
    else:
        label  = 1
        action = "Blocked (sanitization failed)" if spans else "Blocked (no spans found)"

    return dict(tier=tier, risk_score=rs, label=label,
                action=action, sanitized_text=sanitized,
                rescore_risk=new_rs, rescore_tier=new_tier,
                spans_removed=spans, threat_details=details)


# =========================================================
# LIME DISPLAY WEIGHTS  (UI only — not used in decisions)
# =========================================================
def get_lime_weights(text: str) -> list:
    """Returns [(token, weight), ...] for the UI chart. Cached."""
    key = f"w|{text}"
    if key in _lime_cache:
        return _lime_cache[key]
    exp = _lime_explainer.explain_instance(
        text, _lime_proba_batch,
        num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
    weights = exp.as_list()
    _lime_cache[key] = weights
    return weights


# =========================================================
# SPAN INFO — for UI renderer
# =========================================================
def build_span_info(sentence: str, result: dict) -> dict:
    """
    Builds span_info dict for the UI renderer.
    Uses character positions of the first removed span for highlighting.
    """
    spans          = result.get("spans_removed", [])
    clean_fragment = result.get("sanitized_text") or ""
    details        = result.get("threat_details", {})
    any_keyword    = bool([s for s in spans if s in INJECTION_KEYWORDS])

    best_start = best_end = None
    best_span  = None
    sl = sentence.lower()
    for span in spans:
        idx = sl.find(span.lower())
        if idx != -1:
            if best_start is None or idx < best_start:
                best_start = idx
                best_end   = idx + len(span)
                best_span  = span

    return {
        "span_text"     : sentence[best_start:best_end] if best_span else None,
        "span_start"    : best_start,
        "span_end"      : best_end,
        "clean_fragment": clean_fragment,
        "detection_type": "keyword" if any_keyword else "ml",
        "tier"          : result.get("tier", ""),
        "risk_score"    : result.get("risk_score", 0.0),
    }


# =========================================================
# LAYER 1 — QUERY SANITIZATION
# =========================================================
def sanitize_query(query: str) -> dict:
    """
    Layer 1: sanitizes user query using the same full_hepid_predict pipeline.

    Returns dict with:
      safe, original_query, safe_query, tier, risk_score,
      rescore_risk, rescore_tier, threat_details, spans_removed, was_sanitized
    """
    result = full_hepid_predict(query)
    tier   = result["tier"]
    rs     = result["risk_score"]

    if tier == "Benign":
        return dict(safe=True, original_query=query, safe_query=query,
                    tier=tier, risk_score=rs, rescore_risk=None,
                    rescore_tier=None, threat_details=result["threat_details"],
                    spans_removed=[], was_sanitized=False)

    if result["label"] == 0:
        # Suspicious but successfully sanitized
        return dict(safe=True, original_query=query,
                    safe_query=result["sanitized_text"],
                    tier=tier, risk_score=rs,
                    rescore_risk=result["rescore_risk"],
                    rescore_tier=result["rescore_tier"],
                    threat_details=result["threat_details"],
                    spans_removed=result["spans_removed"],
                    was_sanitized=bool(result["spans_removed"]))

    # label == 1: blocked (malicious or sanitization failed)
    return dict(safe=False, original_query=query, safe_query=None,
                tier=tier, risk_score=rs,
                rescore_risk=result.get("rescore_risk"),
                rescore_tier=result.get("rescore_tier"),
                threat_details=result["threat_details"],
                spans_removed=result.get("spans_removed", []),
                was_sanitized=bool(result.get("spans_removed")))


# =========================================================
# LAYER 2 — CLEAN CONTEXT  (document chunk protection)
# =========================================================
def clean_context(context: str, run_lime: bool = True) -> tuple:
    """
    Layer 2: runs full_hepid_predict on every sentence.
    Builds rich item dicts for the UI.
    Returns (malicious_items, clean_text).
    """
    sentences       = sent_tokenize(context)
    malicious_items = []
    clean_parts     = []

    for sentence in sentences:
        result = full_hepid_predict(sentence)
        tier   = result["tier"]

        if tier == "Benign":
            clean_parts.append(sentence)
            continue

        # Get LIME weights for display
        lime_weights = get_lime_weights(sentence) if run_lime else []
        span_info    = build_span_info(sentence, result)

        label     = result["label"]
        spans     = result.get("spans_removed", [])
        rescore_rs = result.get("rescore_risk")
        rescore_t  = result.get("rescore_tier")

        if tier == "Malicious":
            rewritten = "[BLOCKED: MALICIOUS CONTENT]"
        elif label == 0:
            # Successfully sanitized
            rewritten = result["sanitized_text"] or ""
            if rewritten:
                rewritten = rewritten[0].upper() + rewritten[1:]
                if rewritten[-1] not in ".!?":
                    rewritten += "."
        else:
            rewritten = "[BLOCKED: Sanitization Failed]"

        item = dict(
            sentence=sentence, tier=tier,
            risk_score=result["risk_score"], confidence=result["risk_score"],
            threat_details=result["threat_details"],
            spans=spans, lime_weights=lime_weights, span_info=span_info,
            rescore_risk=rescore_rs, rescore_tier=rescore_t,
            final_label=label, rewritten=rewritten,
        )
        malicious_items.append(item)
        logger.info(f"{result['action']} | risk={result['risk_score']:.4f} | {sentence[:80]}")

        if label == 0 and rewritten:
            clean_parts.append(rewritten)

    return malicious_items, " ".join(clean_parts)


# =========================================================
# GEMINI CLIENT
# =========================================================
def get_gemini_client():
    key = os.environ.get("GEMINI_API_KEY", "")
    return _genai.Client(api_key=key) if key else None