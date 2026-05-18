"""
Streamlit frontend — AneRBC Anemia Classification System (light theme)
"""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AneRBC – Anemia Classification",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND = "http://localhost:8000"
CBC_FEATURES = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "MPV", "RDW_CV"]
CLASS_COLORS = {
    "Healthy":    "#22c55e",
    "Microcytic": "#ef4444",
    "Normocytic": "#f59e0b",
    "Macrocytic": "#3b82f6",
}

# ── CSS: light theme matching original UI ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Hide Streamlit chrome */
[data-testid="stDeployButton"]      { display: none !important; }
[data-testid="stToolbar"]           { display: none !important; }
#MainMenu                           { display: none !important; }
footer                              { display: none !important; }
header[data-testid="stHeader"]      { display: none !important; }

html, body, [class*="css"]          { font-family: 'Inter', sans-serif; }

/* White background */
.stApp                              { background: #f8f9fa; }
.block-container                    { padding-top: 1.5rem; max-width: 1200px; }

/* ── Header ── */
.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 0 8px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 24px;
}
.app-header .icon   { font-size: 2rem; }
.app-header h1      { margin: 0; font-size: 1.9rem; font-weight: 700; color: #111827; }
.app-header .sub    { margin: 2px 0 0; font-size: 0.88rem; color: #6b7280; }

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"]   { background: transparent; border-bottom: 2px solid #e5e7eb; gap: 0; }
.stTabs [data-baseweb="tab"]        { color: #6b7280; font-size: 0.9rem; padding: 10px 20px; border-bottom: 2px solid transparent; margin-bottom: -2px; }
.stTabs [aria-selected="true"]      { color: #ef4444; border-bottom: 2px solid #ef4444; font-weight: 600; }

/* ── Cards ── */
.card {
    background: #fff;
    color: #374151;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.card-title { font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 12px; }

/* ── Success banner ── */
.success-banner {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 10px 16px;
    color: #15803d; font-weight: 600; font-size: 0.88rem;
    margin-bottom: 16px;
}

/* ── Prediction box (pink/red like original) ── */
.pred-box {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 12px;
    padding: 20px 24px;
}
.pred-label { font-size: 1.5rem; font-weight: 700; color: #be123c; }

/* ── Confidence card ── */
.conf-card {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 16px 20px;
    text-align: center;
}
.conf-value { font-size: 2rem; font-weight: 800; }
.conf-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }

/* ── Gradient bar ── */
.grad-bar-wrap {
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
    border-radius: 6px; height: 26px; position: relative; margin: 10px 0 20px;
}
.grad-bar-label {
    position: absolute; top: 50%; transform: translate(-50%, -50%);
    color: #fff; font-weight: 700; font-size: 0.85rem;
    text-shadow: 0 1px 3px rgba(0,0,0,.5);
}

/* ── XAI header ── */
.xai-head {
    font-size: 1.15rem; font-weight: 700; color: #111827;
    border-left: 4px solid #6366f1; padding-left: 12px;
    margin: 24px 0 14px;
}

/* ── Info box ── */
.info-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.85rem; color: #78350f; margin-top: 12px;
    line-height: 1.6;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; padding: 10px 32px !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    width: 100%;
}

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 8px; }

/* Fix truncated file names in uploader */
div[data-testid="stUploadedFile"] * {
    white-space: normal !important;
    text-overflow: clip !important;
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def b64_img(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def backend_ok() -> bool:
    try:
        return requests.get(f"{BACKEND}/health", timeout=3).status_code == 200
    except Exception:
        return False


# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="icon">🩸</span>
  <div>
    <h1>Anemia Classification System</h1>
    <p class="sub">Multimodal Classification using RBC Images &amp; CBC Reports</p>
  </div>
</div>
""", unsafe_allow_html=True)

is_backend_ok = backend_ok()
if not is_backend_ok:
    st.error("⚠️ FastAPI backend is not running. Start with: `bash webApp/run.sh`")

# ── tabs: Upload Mode | About ─────────────────────────────────────────────────
tab_upload, tab_about = st.tabs(["📤 Upload Mode", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════
# UPLOAD TAB
# ═══════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload RBC Image & CBC Report")

    col_img, col_cbc = st.columns(2, gap="large")

    with col_img:
        st.markdown('<div class="card"><div class="card-title">📷 RBC Microscopic Image</div>', unsafe_allow_html=True)
        st.caption("Upload RBC image (JPG, PNG)")
        uploaded_image = st.file_uploader("RBC Image", type=["jpg", "jpeg", "png"],
                                          key="img_up", label_visibility="collapsed")
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded RBC Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cbc:
        st.markdown('<div class="card"><div class="card-title">📋 CBC Report Data</div>', unsafe_allow_html=True)
        st.caption("Upload CBC Report (CSV or TXT)")
        uploaded_csv = st.file_uploader("CBC Report", type=["csv", "txt"],
                                        key="csv_up", label_visibility="collapsed")
        if uploaded_csv:
            try:
                df_prev = pd.read_csv(uploaded_csv)
                uploaded_csv.seek(0)
                cols_to_show = [c for c in df_prev.columns if c.lower() not in ("final_class",)]
                df_to_show = df_prev[cols_to_show].head(1).T
                df_to_show.columns = ["Value"]
                st.dataframe(df_to_show, use_container_width=True)
            except Exception:
                st.info("Preview unavailable — file will still be sent to the model.")
        st.markdown('</div>', unsafe_allow_html=True)

    # predict button
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        run = st.button("🔍 Run Prediction", key="btn_run", disabled=not is_backend_ok, use_container_width=True)

    if run:
        if uploaded_image is None:
            st.warning("Please upload an RBC image first.")
        else:
            with st.spinner("Running inference & computing Grad-CAM / SHAP…"):
                files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                if uploaded_csv:
                    files["cbc_csv"] = (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")
                try:
                    resp = requests.post(f"{BACKEND}/predict", files=files, timeout=180)
                    if resp.status_code == 200:
                        result_data = resp.json()
                        st.session_state["result"] = result_data
                        
                        with st.spinner("Generating AI Interpretation..."):
                            try:
                                payload = {
                                    "predicted_class": result_data["predicted_class"],
                                    "cbc_data": result_data.get("cbc_data", {}),
                                    "shap_results": result_data.get("shap_results", []),
                                    "image_b64": result_data.get("original_image"),
                                    "gradcam_b64": result_data.get("gradcam_image")
                                }
                                llm_resp = requests.post(f"{BACKEND}/what_it_is", json=payload, timeout=60)
                                if llm_resp.status_code == 200:
                                    st.session_state["what_it_is"] = llm_resp.json()
                                else:
                                    st.session_state["what_it_is"] = {"text": "Failed to generate explanation.", "provider": "error"}
                            except Exception:
                                st.session_state["what_it_is"] = {"text": "Failed to connect to LLM.", "provider": "error"}
                    else:
                        st.error(f"Backend error {resp.status_code}: {resp.text[:300]}")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

# ═══════════════════════════════════════════════════════════════
# ABOUT TAB
# ═══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div class="card">
<div class="card-title">About This System</div>

**AneRBC Anemia Classification System** uses a multimodal VGG16 + CBC Fusion model to classify
anemia into four categories.

| Class | Description |
|-------|-------------|
| 🟢 Healthy | Normal RBC morphology |
| 🔴 Microcytic | Small RBCs — iron deficiency / thalassaemia |
| 🟡 Normocytic | Normal-size but low count — chronic disease |
| 🔵 Macrocytic | Large RBCs — B12/folate deficiency |

**Explainability:**
- **Grad-CAM** — heatmap overlay on the RBC image showing which regions drove the decision.
- **SHAP** — quantifies each CBC feature's contribution to the prediction.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# RESULTS  (persists across tab switches via session_state)
# ═══════════════════════════════════════════════════════════════
if "result" in st.session_state:
    res = st.session_state["result"]
    pred_class = res["predicted_class"]
    confidence = res["confidence"] * 100
    color = CLASS_COLORS.get(pred_class, "#6366f1")

    st.markdown("---")
    st.markdown('<div class="success-banner">✅ Prediction Complete</div>', unsafe_allow_html=True)

    # Prediction + Confidence
    col_pred, col_conf = st.columns([3, 1], gap="large")
    with col_pred:
        st.markdown(f'<div class="pred-box"><div class="pred-label">🎯 Prediction: {pred_class}</div></div>',
                    unsafe_allow_html=True)
    with col_conf:
        st.markdown(f"""
<div class="conf-card">
  <div class="conf-label">Confidence</div>
  <div class="conf-value" style="color:{color};">{confidence:.1f}%</div>
</div>""", unsafe_allow_html=True)

    # Gradient confidence bar
    st.markdown(f"""
<div class="grad-bar-wrap">
  <span class="grad-bar-label" style="left:{min(confidence, 97):.0f}%;">{confidence:.1f}%</span>
</div>""", unsafe_allow_html=True)

    # Class probabilities chart
    st.markdown('<div class="card"><div class="card-title">📊 Class Probabilities</div>',
                unsafe_allow_html=True)
    probs = res["class_probabilities"]
    names = [p["class"] for p in probs]
    vals  = [p["probability"] for p in probs]
    bar_colors = [CLASS_COLORS.get(n, "#6366f1") for n in names]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(names, vals, color=bar_colors, alpha=0.75)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", color="#374151", fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", color="#6b7280")
    ax.set_title("Class Probabilities", color="#111827", fontsize=11)
    ax.tick_params(colors="#374151")
    for s in ax.spines.values():
        s.set_edgecolor("#e5e7eb")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── XAI ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="xai-head">🔍 Explainable AI</div>', unsafe_allow_html=True)

    xai_l, xai_r = st.columns(2, gap="large")

    with xai_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 Grad-CAM Visualization (RBC Focus)")
        if res.get("gradcam_image"):
            st.image(b64_img(res["gradcam_image"]),
                     caption="Grad-CAM Explanation (Heatmap Overlay)",
                     use_container_width=True)
        st.markdown("""
<div class="info-box">
🟧 <strong>Red regions</strong> indicate areas of the RBC image that most influenced
the model's prediction.<br><br>
This helps identify which morphological features were important for classification.
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with xai_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📈 SHAP Analysis")
        if res.get("shap_chart"):
            st.image(b64_img(res["shap_chart"]), use_container_width=True)
        shap_data = res.get("shap_results", [])
        if shap_data:
            df_shap = pd.DataFrame(shap_data)
            df_shap.columns = ["Feature", "Importance"]
            df_shap["Importance"] = df_shap["Importance"].map(lambda x: f"{x:.4f}")
            st.dataframe(df_shap, use_container_width=True, hide_index=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── What it is? ───────────────────────────────────────────────────────────
    if "what_it_is" in st.session_state:
        st.markdown('<div class="xai-head">🤖 What it is? (AI Interpretation)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        llm_data = st.session_state["what_it_is"]
        st.markdown(llm_data.get("text", ""))
        st.caption(f"Provider: {llm_data.get('provider', 'unknown')}")
        st.markdown('</div>', unsafe_allow_html=True)
