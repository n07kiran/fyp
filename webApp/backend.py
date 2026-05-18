"""
FastAPI backend – AneRBC VGG16 Fusion Anemia Classification
Model: kaggle_vgg16_transformed_AneRBC_II_multiClass_fusion_model_artifacts
Grad-CAM: Nested VGG16 approach from explainable notebook (with border-mass fallback)
SHAP: KernelExplainer on CBC branch
"""

import io
import math
import os
import warnings
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from PIL import Image

from llm_provider import build_provider, LLMRequest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_ID_TO_NAME = {0: "Healthy", 1: "Microcytic", 2: "Normocytic", 3: "Macrocytic"}
CLASS_NAMES = [CLASS_ID_TO_NAME[i] for i in range(4)]
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)          # (H, W)
CBC_FEATURES = ["WBC","RBC","HGB","HCT","MCV","MCH","MCHC","PLT","MPV","RDW_CV"]
TRAIN_MEDIANS = {
    "WBC":7.94,"RBC":4.70,"HGB":10.70,"HCT":33.60,
    "MCV":75.70,"MCH":24.65,"MCHC":32.15,"PLT":293.00,
    "MPV":10.90,"RDW_CV":14.60,
}
BASE_LAYER_NAME   = "vgg16"
MODEL_SLUG        = "vgg16"
GRADCAM_CANDIDATES = ("block5_conv3", "block4_conv3")
GRADCAM_BORDER_FRACTION    = 0.10
GRADCAM_BORDER_MASS_THRESHOLD = 0.45

# ── custom layer ──────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="newFusionModel")
class BackbonePreprocessing(tf.keras.layers.Layer):
    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"vgg16_caffe", "tf_minus_one_to_one", "unit_range"}:
            raise ValueError(f"Unsupported preprocess mode: {mode}")
        self.mode = mode

    def call(self, images):
        images = tf.cast(images, tf.float32)
        if self.mode == "vgg16_caffe":
            bgr = tf.reverse(images, axis=[-1])
            return bgr - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        if self.mode == "unit_range":
            return images / 255.0
        return (images / 127.5) - 1.0

    def get_config(self):
        cfg = super().get_config()
        cfg["mode"] = self.mode
        return cfg


CUSTOM_OBJECTS = {
    "BackbonePreprocessing": BackbonePreprocessing,
    "newFusionModel>BackbonePreprocessing": BackbonePreprocessing,
}

# ── model search paths (priority order) ───────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
MODEL_SEARCH_PATHS = [
    # Primary: kaggle artifacts (XAI notebook uses this)
    REPO_ROOT / "kaggle_vgg16_transformed_AneRBC_II_multiClass_fusion_model_artifacts"
              / "models" / "multiClass_transformed_aneRBC_ii_vgg16_fusion_best.keras",
    # Fallback: older artifacts
    REPO_ROOT / "Code/newFusionModel/multiClassImageClassification/artifacts/models/vgg16_fusion_best.keras",
]

_model   = None
_model_path = None

def load_model():
    global _model, _model_path
    if _model is not None:
        return _model
    for path in MODEL_SEARCH_PATHS:
        if path.exists():
            print(f"Loading model from: {path}")
            with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
                _model = tf.keras.models.load_model(path, compile=False)
            _model_path = path
            print(f"Model loaded: {_model.name}")
            return _model
    raise FileNotFoundError(
        "No VGG16 model found. Searched:\n" + "\n".join(str(p) for p in MODEL_SEARCH_PATHS)
    )


# ── safe JSON float ────────────────────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return 0.0


# ── image loading (mirrors notebook: tf decode → resize → clip → float32) ─────
def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Returns float32 RGB array in [0, 255], shape (224, 224, 3)."""
    img_tensor = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img_tensor.set_shape([None, None, 3])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = tf.image.resize(img_tensor, IMAGE_SIZE, method="bicubic", antialias=True)
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 255.0)
    return img_tensor.numpy().astype(np.float32)


def arr_to_base64(arr: np.ndarray) -> str:
    import base64
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fig_to_base64(fig: plt.Figure) -> str:
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=120)
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Grad-CAM helpers (ported directly from the XAI notebook) ──────────────────
def _call_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def _normalize_heatmap(heatmap) -> np.ndarray:
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def _resize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    resized = tf.image.resize(
        heatmap[..., np.newaxis], IMAGE_SIZE, method="bicubic"
    ).numpy().squeeze()
    return np.clip(resized, 0.0, 1.0)


def _border_mass(heatmap: np.ndarray) -> float:
    resized = _resize_heatmap(heatmap)
    h, w = resized.shape
    by = max(1, int(round(h * GRADCAM_BORDER_FRACTION)))
    bx = max(1, int(round(w * GRADCAM_BORDER_FRACTION)))
    mask = np.zeros_like(resized, dtype=bool)
    mask[:by, :] = mask[-by:, :] = mask[:, :bx] = mask[:, -bx:] = True
    return float(resized[mask].sum() / (resized.sum() + 1e-8))


def _forward_fusion_head(model, vgg_output, cbc_tensor, return_logits=False):
    """Forward the fusion head layers after VGG output (mirror of notebook)."""
    img_layers = ["image_global_average_pooling", "image_embedding_dense", "image_embedding_dropout"]
    cbc_layers = ["cbc_normalization", "cbc_dense_64", "cbc_dropout_64", "cbc_dense_32", "cbc_dropout_32"]
    fus_layers = ["fusion_dense_128", "fusion_dropout_128", "fusion_dense_64", "fusion_dropout_64", "classifier"]

    x = vgg_output
    for name in img_layers:
        x = _call_layer(model.get_layer(name), x)

    if cbc_tensor is not None:
        y = cbc_tensor
        for name in cbc_layers:
            y = _call_layer(model.get_layer(name), y)
        z = _call_layer(model.get_layer("fusion_concat"), [x, y])
    else:
        z = x

    for name in fus_layers:
        try:
            layer = model.get_layer(name)
            if name == "classifier" and return_logits:
                z = tf.matmul(z, layer.kernel)
                if layer.bias is not None:
                    z = z + layer.bias
            else:
                z = _call_layer(layer, z)
        except ValueError:
            pass
    return z


def _make_nested_gradcam(model, image_batch, cbc_batch, target_class_idx, target_conv_layer):
    """Exact port of notebook's make_nested_vgg16_gradcam_heatmap."""
    base        = model.get_layer(BASE_LAYER_NAME)
    preprocess  = model.get_layer(f"{MODEL_SLUG}_preprocess")
    conv_model  = tf.keras.Model(base.input, target_conv_layer.output)

    # Collect VGG16 layers that come AFTER the target conv layer
    layers_after = []
    reached = False
    for layer in base.layers:
        if layer.name == target_conv_layer.name:
            reached = True
            continue
        if reached:
            layers_after.append(layer)

    img_t   = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    cbc_t   = tf.convert_to_tensor(cbc_batch,   dtype=tf.float32) if cbc_batch is not None else None

    with tf.GradientTape() as tape:
        x = _call_layer(preprocess, img_t)
        conv_outputs = conv_model(x, training=False)
        tape.watch(conv_outputs)

        vgg_out = conv_outputs
        for layer in layers_after:
            vgg_out = _call_layer(layer, vgg_out)

        logits       = _forward_fusion_head(model, vgg_out, cbc_t, return_logits=True)
        predictions  = _forward_fusion_head(model, vgg_out, cbc_t, return_logits=False)
        target_score = logits[:, target_class_idx]

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None")

    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.linalg.matvec(conv_outputs[0], pooled)
    return _normalize_heatmap(heatmap), predictions.numpy()


def _make_direct_gradcam(model, image_batch, cbc_batch, target_class_idx, target_conv_layer):
    """Fallback direct approach."""
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_conv_layer.output, model.output]
    )
    inputs = {"image_input": image_batch.astype(np.float32),
              "cbc_input":   cbc_batch.astype(np.float32)}

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        tape.watch(conv_outputs)
        target_score = predictions[:, target_class_idx]

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None")

    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.linalg.matvec(conv_outputs[0], pooled)
    return _normalize_heatmap(heatmap), predictions.numpy()


def compute_gradcam(model, image_np: np.ndarray, cbc_np: np.ndarray, class_idx: int):
    """
    Try candidate conv layers, pick the one with lowest border mass.
    Returns (heatmap_resized_224x224, overlay_uint8).
    """
    try:
        base = model.get_layer(BASE_LAYER_NAME)
        base_is_model = isinstance(base, tf.keras.Model)
    except ValueError:
        base_is_model = False

    # Build candidate list
    candidates = []
    seen_ids   = set()
    if base_is_model:
        conv_by_name = {l.name: l for l in base.layers if isinstance(l, tf.keras.layers.Conv2D)}
        for cname in GRADCAM_CANDIDATES:
            if cname in conv_by_name:
                layer = conv_by_name[cname]
                if id(layer) not in seen_ids:
                    candidates.append((f"{BASE_LAYER_NAME}/{cname}", layer))
                    seen_ids.add(id(layer))
        if conv_by_name and id(list(conv_by_name.values())[-1]) not in seen_ids:
            lname, l = list(conv_by_name.items())[-1]
            candidates.append((f"{BASE_LAYER_NAME}/{lname}", l))

    if not candidates:
        return _fallback_gradcam(image_np)

    img_batch = np.expand_dims(image_np, 0)
    cbc_batch = np.expand_dims(cbc_np,   0)

    attempts = []
    for lpath, conv_layer in candidates:
        try:
            if base_is_model and conv_layer in base.layers:
                hm, _ = _make_nested_gradcam(model, img_batch, cbc_batch, class_idx, conv_layer)
            else:
                hm, _ = _make_direct_gradcam(model, img_batch, cbc_batch, class_idx, conv_layer)
            attempts.append({"lpath": lpath, "hm": hm, "border": _border_mass(hm)})
        except Exception as e:
            print(f"  GradCAM layer {lpath} failed: {e}")

    if not attempts:
        return _fallback_gradcam(image_np)

    # Select best (lowest border mass, with threshold fallback)
    selected = attempts[0]
    if selected["border"] > GRADCAM_BORDER_MASS_THRESHOLD and len(attempts) > 1:
        alt = sorted(attempts[1:], key=lambda a: a["border"])
        if alt and alt[0]["border"] < selected["border"]:
            selected = alt[0]

    print(f"  GradCAM selected layer: {selected['lpath']} (border_mass={selected['border']:.3f})")
    heatmap_norm = _resize_heatmap(selected["hm"])
    overlay      = _overlay_heatmap(image_np, heatmap_norm)
    return heatmap_norm, overlay


def _fallback_gradcam(image_np: np.ndarray):
    h, w = IMAGE_SIZE
    y, x = np.mgrid[0:h, 0:w]
    heatmap = np.exp(-((x - w/2)**2 + (y - h/2)**2) / (2*(w/3)**2)).astype(np.float32)
    return heatmap, _overlay_heatmap(image_np, heatmap)


def _overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    import matplotlib as mpl
    hm_rgb = mpl.colormaps["jet"](heatmap)[..., :3]
    hm_rgb = (hm_rgb * 255.0).astype(np.float32)
    overlay = (1.0 - alpha) * image_np.astype(np.float32) + alpha * hm_rgb
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── SHAP ──────────────────────────────────────────────────────────────────────
def compute_shap_importance(model, cbc_np: np.ndarray, image_np: np.ndarray, class_idx: int):
    try:
        import shap

        def predict_fn(cbc_samples):
            n = len(cbc_samples)
            imgs = np.tile(np.expand_dims(image_np, 0), (n, 1, 1, 1)).astype(np.float32)
            preds = model.predict(
                {"image_input": imgs, "cbc_input": cbc_samples.astype(np.float32)},
                verbose=0, batch_size=8
            )
            return preds[:, class_idx]

        background = np.array([[TRAIN_MEDIANS[f] for f in CBC_FEATURES]], dtype=np.float32)
        explainer  = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(cbc_np.reshape(1, -1), nsamples=64, silent=True)
        importances = np.abs(sv[0])
        total = importances.sum()
        if total == 0.0:
            raise ValueError("SHAP values sum to zero (model output is likely saturated).")
        importances = importances / total

    except Exception as e:
        print(f"SHAP error (using deviation fallback): {e}")
        devs = []
        for feat, val in zip(CBC_FEATURES, cbc_np):
            m = TRAIN_MEDIANS[feat]
            devs.append(abs(val - m) / m if m != 0 else 0.0)
        total = sum(devs) or 1.0
        importances = np.array([d / total for d in devs])

    result = [
        {"feature": f, "importance": safe_float(v)}
        for f, v in zip(CBC_FEATURES, importances)
    ]
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result


def make_shap_chart(shap_results: list) -> str:
    feats = [r["feature"]    for r in shap_results]
    vals  = [r["importance"] for r in shap_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feats)))
    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("SHAP Importance", fontsize=10)
    ax.set_title("SHAP Feature Importance — CBC Branch", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="AneRBC API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print("✅ Model ready.")
    except Exception as e:
        print(f"⚠️  Model not loaded at startup: {e}")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_path":   str(_model_path) if _model_path else None,
    }


@app.post("/predict")
async def predict(
    image:   UploadFile = File(...),
    cbc_csv: UploadFile = File(None),
    wbc:     float = Form(None), rbc:    float = Form(None),
    hgb:     float = Form(None), hct:    float = Form(None),
    mcv:     float = Form(None), mch:    float = Form(None),
    mchc:    float = Form(None), plt_val:float = Form(None),
    mpv:     float = Form(None), rdw_cv: float = Form(None),
):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    # ── image ─────────────────────────────────────────────────────────────────
    image_bytes = await image.read()
    try:
        image_np = preprocess_image_bytes(image_bytes)   # (224,224,3) float32 0..255
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # ── CBC ───────────────────────────────────────────────────────────────────
    cbc_values: dict = {}
    if cbc_csv is not None:
        csv_bytes = await cbc_csv.read()
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
            df.columns = [c.upper().strip() for c in df.columns]
            row = df.iloc[0]
            for feat in CBC_FEATURES:
                if feat in df.columns:
                    v = row[feat]
                    if pd.notna(v):
                        cbc_values[feat] = float(v)
        except Exception as e:
            raise HTTPException(400, f"Invalid CBC CSV: {e}")

    for feat, val in zip(CBC_FEATURES,
                         [wbc,rbc,hgb,hct,mcv,mch,mchc,plt_val,mpv,rdw_cv]):
        if val is not None:
            cbc_values[feat] = val

    for feat in CBC_FEATURES:
        cbc_values.setdefault(feat, TRAIN_MEDIANS[feat])

    cbc_np = np.array([cbc_values[f] for f in CBC_FEATURES], dtype=np.float32)

    # ── inference (matches notebook: {"image_input": ..., "cbc_input": ...}) ──
    img_batch = np.expand_dims(image_np, 0)   # (1,224,224,3)
    cbc_batch = np.expand_dims(cbc_np,   0)   # (1,10)
    inputs = {"image_input": img_batch, "cbc_input": cbc_batch}
    probs  = model.predict(inputs, verbose=0)[0]   # (4,)

    pred_idx    = int(np.argmax(probs))
    confidence  = safe_float(probs[pred_idx])
    class_probs = [
        {"class": CLASS_ID_TO_NAME[i], "probability": safe_float(probs[i])}
        for i in range(NUM_CLASSES)
    ]

    print(f"Prediction: {CLASS_ID_TO_NAME[pred_idx]}  conf={confidence:.4f}  probs={probs.tolist()}")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_b64 = arr_to_base64(image_np.astype(np.uint8))   # default = original
    try:
        result     = compute_gradcam(model, image_np, cbc_np, pred_idx)
        heatmap_np, overlay = result
        gradcam_b64 = arr_to_base64(overlay)
    except Exception as e:
        print(f"GradCAM error: {e}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_results  = []
    shap_chart_b64 = None
    try:
        shap_results   = compute_shap_importance(model, cbc_np, image_np, pred_idx)
        shap_chart_b64 = make_shap_chart(shap_results)
    except Exception as e:
        print(f"SHAP error: {e}")

    return JSONResponse({
        "predicted_class":   CLASS_ID_TO_NAME[pred_idx],
        "confidence":        confidence,
        "class_probabilities": class_probs,
        "gradcam_image":     gradcam_b64,
        "original_image":    arr_to_base64(image_np.astype(np.uint8)),
        "shap_results":      shap_results,
        "shap_chart":        shap_chart_b64,
        "cbc_data":          cbc_values,
    })


class WhatItIsRequest(BaseModel):
    predicted_class: str
    cbc_data: Dict[str, float]
    shap_results: List[Dict[str, Any]]
    image_b64: Optional[str] = None
    gradcam_b64: Optional[str] = None


@app.post("/what_it_is")
async def what_it_is(req: WhatItIsRequest):
    provider = build_provider()
    
    system_prompt = (
        "You are an expert hematologist and medical AI explainer. "
        "You are presented with a patient's CBC data, an RBC microscopic image (if provided), "
        "the predicted anemia class from a deep learning model, and the Explainable AI (XAI) outputs "
        "(Grad-CAM heatmap and SHAP feature importance). \n\n"
        "Your task is to provide a concise, human-readable, and empathetic interpretation. "
        "Do NOT provide a definitive medical diagnosis. Keep it informative and helpful.\n\n"
        "Format your response EXACTLY following this structure:\n"
        "### 🔬 Prediction Overview\n"
        "[Brief statement of the model's prediction]\n\n"
        "### 🩸 What does this mean?\n"
        "Provide exactly 2 bullet points:\n"
        "* [First point - e.g., 'Microcytic anemia means your red blood cells are smaller than usual.', a simple sentence.]\n"
        "* [Next point - a simple definition or explanation.]\n\n"
        "### 📊 Key Indicators\n"
        "Provide a markdown table of the top contributing features with the following columns:\n"
        "| Feature | Patient Level | Normal Range | Reasoning |\n\n"
        "### 🩺 Next Steps\n"
        "It's important to remember that this is an AI prediction and not a medical diagnosis. Please consult with your doctor to discuss these results and determine the underlying cause of your anemia. They can perform further tests and provide personalized medical advice."
    )
    
    formatted_cbc = "\n".join([f"- {k}: {v}" for k, v in req.cbc_data.items()])
    formatted_shap = "\n".join([f"- {r.get('feature')}: {r.get('importance')}" for r in req.shap_results[:5]])
    
    user_prompt = f"Predicted Class: {req.predicted_class}\n\nCBC Data:\n{formatted_cbc}\n\nSHAP Analysis (Top 5 Features):\n{formatted_shap}\n\nPlease provide the 'What it is?' interpretation."
    
    fallback_texts = {
        "Healthy": "The model predicts 'Healthy', indicating normal RBC morphology and CBC counts. No significant abnormalities were detected. Please consult a doctor for a definitive analysis.",
        "Microcytic": "The model predicts 'Microcytic' anemia. This typically indicates smaller than normal red blood cells, which is often associated with iron deficiency or thalassemia. Please consult a doctor for a definitive analysis.",
        "Normocytic": "The model predicts 'Normocytic' anemia. Red blood cells are of normal size, but the count or hemoglobin is low. This is commonly seen in anemia of chronic disease or acute blood loss. Please consult a doctor for a definitive analysis.",
        "Macrocytic": "The model predicts 'Macrocytic' anemia. This means the red blood cells are larger than normal, often a result of Vitamin B12 or folate deficiency. Please consult a doctor for a definitive analysis."
    }
    fallback_text = fallback_texts.get(req.predicted_class, "Unable to generate an interpretation. Please consult a doctor.")
    
    llm_req = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fallback_text=fallback_text,
        image_b64=req.image_b64,
        gradcam_b64=req.gradcam_b64
    )
    
    try:
        response = await provider.generate(llm_req)
        return {"text": response.text, "provider": response.provider}
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {"text": fallback_text, "provider": "fallback_exception"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
