# app.py
# OCULAIRE - Streamlit UI (refreshed visual layout & UX)
# Keep your existing model files (bscan_cnn.h5, rnflt_scaler.joblib, rnflt_kmeans.joblib,
# avg_map_healthy.npy, avg_map_glaucoma.npy) in the same folder as this app.

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import os
from typing import Tuple

# ---------------------------
# Page config & CSS styling
# ---------------------------
st.set_page_config(
    page_title="OCULAIRE — Glaucoma Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👁️"
)

# Lightweight CSS to improve look
st.markdown(
    """
    <style>
    /* General font + container styling */
    .reportview-container { font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
    }
    /* Card shadow for containers */
    .card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 6px 16px rgba(17, 24, 39, 0.08);
        margin-bottom: 1rem;
    }
    /* Smaller captions & muted text */
    .muted { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Top banner / header
# ---------------------------
with st.container():
    left, middle, right = st.columns([1, 6, 1])
    with left:
        # Place a small logo image file in ./assets/logo.png (optional)
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=92)
        else:
            st.write("")  # keep alignment if no logo
    with middle:
        st.markdown("<h1 style='margin:0'>👁️ OCULAIRE</h1>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Automated glaucoma screening using RNFLT maps & B-scan CNN</div>", unsafe_allow_html=True)
    with right:
        st.markdown("")  # keep header aligned

st.markdown("---")

# ---------------------------
# Sidebar controls + help
# ---------------------------
with st.sidebar:
    st.header("Analysis Controls")
    analysis_type = st.radio(
        "Select analysis type",
        ("🩺 RNFLT Map Analysis (.npz)", "👁️ B-Scan Slice Analysis (Image)"),
    )
    st.markdown("---")
    st.info("Upload your RNFLT (.npz) map or a B-scan image and press Upload.")
    st.markdown("**Tips**")
    st.caption("• Use high-quality B-Scan (cropped to retina) for better Grad-CAM.\n• RNFLT maps should match shape of avg maps if possible.")
    st.markdown("---")
    with st.expander("Sample files & help"):
        st.write("Place sample files in the app folder or upload via the UI.")
        st.write("- `bscan_cnn.h5` (model)")
        st.write("- `rnflt_scaler.joblib`, `rnflt_kmeans.joblib`")
        st.write("- `avg_map_healthy.npy`, `avg_map_glaucoma.npy`")

# ---------------------------
# Model loading helpers
# ---------------------------
@st.cache_resource
def load_bscan_model():
    try:
        with st.spinner("Loading B-scan CNN model..."):
            model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading B-Scan CNN model: {e}")
        return None

@st.cache_resource
def load_rnflt_models_safe():
    try:
        with st.spinner("Loading RNFLT artifacts..."):
            scaler = joblib.load("rnflt_scaler.joblib")
            kmeans = joblib.load("rnflt_kmeans.joblib")
            avg_healthy = np.load("avg_map_healthy.npy")
            avg_glaucoma = np.load("avg_map_glaucoma.npy")
        # Decide mapping of thick/thin cluster heuristically:
        thick_cluster, thin_cluster = (1, 0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0, 1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception as e:
        st.warning("RNFLT artifacts missing or failed to load.")
        st.error(f"Error loading RNFLT artifacts: {e}")
        return None, None, None, None, None, None

# ---------------------------
# RNFLT helpers (keep your logic)
# ---------------------------
def process_uploaded_npz(uploaded_file) -> Tuple[np.ndarray, dict]:
    try:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        npz = np.load(file_bytes, allow_pickle=True)
        # try to pick 'volume' or the first array
        rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
        if rnflt_map.ndim == 3:
            rnflt_map = rnflt_map[0, :, :]
        vals = rnflt_map.flatten().astype(float)
        metrics = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals))
        }
        return rnflt_map, metrics
    except Exception as e:
        st.error(f"Error processing .npz file: {e}")
        return None, None

def compute_risk_map(rnflt_map, healthy_avg, threshold=-10):
    if rnflt_map.shape != healthy_avg.shape:
        healthy_avg = cv2.resize(healthy_avg, (rnflt_map.shape[1], rnflt_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < threshold, diff, np.nan)
    total_pixels = np.isfinite(diff).sum()
    risky_pixels = np.isfinite(risk).sum()
    severity = (risky_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    return diff, risk, severity

# ---------------------------
# B-scan helpers (keep your logic)
# ---------------------------
def preprocess_bscan_image(image_pil, img_size=(224, 224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_resized = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_resized[..., None], 3, axis=-1)
    img_batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return img_batch, arr_resized

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        st.error("Could not find a Conv2D layer for Grad-CAM.")
        return None
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        # if model outputs more than 1 neuron, adjust this
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()

# ---------------------------
# Main App UI Logic
# ---------------------------
if "RNFLT" in analysis_type:
    st.header("RNFLT Map Analysis — Unsupervised (Phase D)")
    container = st.container()
    with container:
        scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models_safe()
        if scaler is None:
            st.info("Place RNFLT artifacts in the app folder to enable RNFLT analysis.")
            st.stop()

        uploaded_file = st.file_uploader("Upload an RNFLT .npz file", type=["npz"])

    if uploaded_file is not None:
        rnflt_map, metrics = process_uploaded_npz(uploaded_file)
        if rnflt_map is not None:
            # Prediction box
            X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            X_scaled = scaler.transform(X_new)
            cluster = int(kmeans.predict(X_scaled)[0])
            label = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            diff, risk, severity = compute_risk_map(rnflt_map, avg_healthy)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Diagnosis Summary")
            cols = st.columns(4)
            status_emoji = "🚨" if label == "Glaucoma-like" else "✅"
            cols[0].metric("Predicted Status", f"{status_emoji} {label}")
            cols[1].metric("Mean RNFLT (µm)", f"{metrics['mean']:.2f}")
            cols[2].metric("Severity Score", f"{severity:.2f}%")
            cols[3].metric("K-Means Cluster", cluster)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            with st.expander("🔬 Detailed RNFLT Visualization", expanded=True):
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                im1 = axes[0].imshow(rnflt_map, cmap='turbo')
                axes[0].set_title("Uploaded RNFLT Map")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], shrink=0.8, label="Thickness (µm)")

                im2 = axes[1].imshow(diff, cmap='bwr', vmin=-25, vmax=25)
                axes[1].set_title("Difference Map (vs. Healthy)")
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], shrink=0.8, label="Δ Thickness (µm)")

                im3 = axes[2].imshow(risk, cmap='hot')
                axes[2].set_title("Risk Map (Thinner Zones)")
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2], shrink=0.8, label="Δ Thickness (µm)")

                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("📊 Numeric Metrics and Histogram", expanded=False):
                st.write("Basic numeric metrics for the uploaded RNFLT map")
                df = pd.DataFrame([metrics])
                st.table(df.style.format("{:.2f}"))

                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.hist(rnflt_map.flatten(), bins=60)
                ax2.set_title("RNFLT Distribution")
                ax2.set_xlabel("Thickness (µm)")
                ax2.set_ylabel("Pixel count")
                plt.tight_layout()
                st.pyplot(fig2)

elif "B-Scan" in analysis_type:
    st.header("B-Scan Slice Analysis — Supervised CNN (Phase S)")
    container = st.container()
    with container:
        model = load_bscan_model()
        if model is None:
            st.info("Place the B-scan model file `bscan_cnn.h5` in the app folder.")
            st.stop()
        uploaded_file = st.file_uploader("Upload a B-Scan image (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        img_batch, processed_img_display = preprocess_bscan_image(image_pil)
        with st.spinner("Running model prediction..."):
            pred_raw = model.predict(img_batch, verbose=0)[0][0]
        label = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
        confidence = pred_raw * 100 if label == "Glaucoma-like" else (1 - pred_raw) * 100
        status_emoji = "🚨" if label == "Glaucoma-like" else "✅"

        # top-level metric
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric(label="Prediction", value=f"{status_emoji} {label}", delta=f"{confidence:.2f}% Confidence")
        st.markdown("</div>", unsafe_allow_html=True)

        col_img, col_cam = st.columns([1, 2])
        with col_img:
            st.subheader("Original Image")
            st.image(image_pil, caption="Uploaded B-Scan", use_column_width=True)
        with col_cam:
            st.subheader("Grad-CAM Interpretation")
            heatmap = make_gradcam_heatmap(img_batch, model)
            if heatmap is not None:
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap_img = (heatmap * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                superimposed_img = (np.stack([processed_img_display]*3, axis=-1) * 255).astype(np.uint8)
                superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap_color, 0.4, 0)
                c1, c2 = st.columns(2)
                c1.image(heatmap_color, caption="Heatmap", use_column_width=True)
                c2.image(superimposed_img, caption="Overlay: Areas of Focus", use_column_width=True)
            else:
                st.warning("Could not generate Grad-CAM visualization.")

# Footer
st.markdown("---")
st.markdown("<div class='muted'>Built for research & demo purposes. For clinical use, validate with expert ophthalmologists.</div>", unsafe_allow_html=True)

