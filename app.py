# app.py — OCULAIRE (Dark mode, fixed titles + readable plots)
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

# -----------------------
# Page config (restored name)
# -----------------------
st.set_page_config(page_title="OCULAIRE: Glaucoma Detection Dashboard",
                   layout="wide",
                   page_icon="👁️")

# -----------------------
# Matplotlib dark-friendly config
# -----------------------
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "#0b1220",
    "axes.facecolor": "#0b1220",
    "axes.edgecolor": "#e6eef8",
    "axes.labelcolor": "#e6eef8",
    "xtick.color": "#e6eef8",
    "ytick.color": "#e6eef8",
    "text.color": "#e6eef8",
    "font.size": 12,
    "axes.titleweight": "bold",
})

# -----------------------
# CSS: dark UI tweaks
# -----------------------
st.markdown("""
<style>
:root{
  --bg:#071016;
  --card:#0f1724;
  --muted:#9aa4b2;
  --accent:#7dd3fc;
}
.stApp { background: linear-gradient(180deg,#071116 0%, #05060a 100%); color: #e6eef8; }
.header-title { font-size:34px; font-weight:800; margin:0; color:#fff; }
.header-sub { color:var(--muted); margin-top:4px; font-size:14px; }
.card { background: linear-gradient(180deg, #0b1722, #09121a); padding:14px; border-radius:10px; box-shadow:0 8px 20px rgba(0,0,0,0.6); }
.uploader-card { background:#0d1720; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
.large-metric { font-size:22px; font-weight:700; color:#e6eef8; }
.metric-label { font-size:12px; color:#9aa4b2; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
logo = "assets/logo.png"
col1, col2, col3 = st.columns([1, 12, 1])
with col1:
    if os.path.exists(logo):
        st.image(logo, width=64)
with col2:
    st.markdown("<div class='header-title'>👁️ OCULAIRE: Glaucoma Detection Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>Automated glaucoma screening — RNFLT maps & B-scan CNN</div>", unsafe_allow_html=True)
with col3:
    st.write("")

st.markdown("---")

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("Analysis Controls")
    analysis_type = st.radio("Select Analysis Type",
                             ("🩺 RNFLT Map Analysis (.npz)", "👁️ B-Scan Slice Analysis (Image)"))
    st.markdown("---")
    st.markdown("**Demo**")
    if st.button("Load example RNFLT"):
        st.session_state["_load_example_rnflt"] = True
    if st.button("Load example B-scan"):
        st.session_state["_load_example_bscan"] = True
    st.markdown("---")
    st.markdown("Hints:")
    st.caption("- Use high-quality B-scan for better Grad-CAM\n- RNFLT maps will be resized if needed")

# -----------------------
# Model loaders (unchanged logic)
# -----------------------
@st.cache_resource
def load_bscan_model():
    try:
        with st.spinner("Loading B-scan model..."):
            model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading B-scan model: {e}")
        return None

@st.cache_resource
def load_rnflt_models_safe():
    try:
        with st.spinner("Loading RNFLT artifacts..."):
            scaler = joblib.load("rnflt_scaler.joblib")
            kmeans = joblib.load("rnflt_kmeans.joblib")
            avg_healthy = np.load("avg_map_healthy.npy")
            avg_glaucoma = np.load("avg_map_glaucoma.npy")
        thick_cluster, thin_cluster = (1, 0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0, 1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception as e:
        st.warning("RNFLT artifacts missing or failed to load.")
        return None, None, None, None, None, None

# -----------------------
# RNFLT helpers (unchanged)
# -----------------------
def process_uploaded_npz(uploaded_file):
    try:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        npz = np.load(file_bytes, allow_pickle=True)
        rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
        if rnflt_map.ndim == 3:
            rnflt_map = rnflt_map[0, :, :]
        vals = rnflt_map.flatten().astype(float)
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return rnflt_map, metrics
    except Exception as e:
        st.error(f"Error reading .npz: {e}")
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
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()

# -----------------------
# Main UI
# -----------------------
if "RNFLT" in analysis_type:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("RNFLT Map Analysis (Unsupervised)")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<div class='uploader-card'>", unsafe_allow_html=True)
        st.markdown("<strong style='color:#e6eef8'>Upload an RNFLT .npz file</strong>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag & drop or click to choose (max 200MB)", type=["npz"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>Upload status</b><br>", unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown("<span class='metric-label'>No file uploaded</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='metric-label'>File: {uploaded_file.name}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        rnflt_map, metrics = process_uploaded_npz(uploaded_file)
        scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models_safe()
        if scaler is None:
            st.error("RNFLT artifacts missing. Place `.joblib` and `.npy` files next to app.py.")
        else:
            X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            X_scaled = scaler.transform(X_new)
            cluster = int(kmeans.predict(X_scaled)[0])
            label = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            diff, risk, severity = compute_risk_map(rnflt_map, avg_healthy)

            # Big, clear numeric metrics (high contrast)
            m1, m2, m3, m4 = st.columns([2,2,2,1])
            m1.markdown("<div class='metric-label'>Predicted Status</div><div class='large-metric'>{}</div>".format("🚨 "+label if "Glaucoma" in label else "✅ "+label), unsafe_allow_html=True)
            m2.markdown("<div class='metric-label'>Mean RNFLT (µm)</div><div class='large-metric'>{:.2f}</div>".format(metrics['mean']), unsafe_allow_html=True)
            m3.markdown("<div class='metric-label'>Severity %</div><div class='large-metric'>{:.2f}%</div>".format(severity), unsafe_allow_html=True)
            m4.markdown("<div class='metric-label'>K-Means Cluster</div><div class='large-metric'>{}</div>".format(cluster), unsafe_allow_html=True)

            st.markdown("---")

            # Visualization with colorbars forced to light ticks/labels
            with st.expander("🔬 Detailed RNFLT Visualization", expanded=True):
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

                im0 = axes[0].imshow(rnflt_map, cmap='turbo')
                axes[0].axis('off'); axes[0].set_title("Uploaded RNFLT Map", color="#e6eef8", fontsize=16)
                c0 = plt.colorbar(im0, ax=axes[0], shrink=0.85, label="Thickness (µm)")
                c0.ax.yaxis.set_tick_params(color='#e6eef8'); c0.ax.yaxis.label.set_color('#e6eef8')
                c0.outline.set_edgecolor('#e6eef8')

                im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30)
                axes[1].axis('off'); axes[1].set_title("Difference Map (vs Healthy)", color="#e6eef8", fontsize=16)
                c1 = plt.colorbar(im1, ax=axes[1], shrink=0.85, label="Δ Thickness (µm)")
                c1.ax.yaxis.set_tick_params(color='#e6eef8'); c1.ax.yaxis.label.set_color('#e6eef8')
                c1.outline.set_edgecolor('#e6eef8')

                im2 = axes[2].imshow(risk, cmap='hot')
                axes[2].axis('off'); axes[2].set_title("Risk Map (Thinner Zones)", color="#e6eef8", fontsize=16)
                c2 = plt.colorbar(im2, ax=axes[2], shrink=0.85, label="Δ Thickness (µm)")
                c2.ax.yaxis.set_tick_params(color='#e6eef8'); c2.ax.yaxis.label.set_color('#e6eef8')
                c2.outline.set_edgecolor('#e6eef8')

                # ensure figure background is dark so plots appear integrated
                fig.patch.set_facecolor('#071116')
                for ax in axes: ax.set_facecolor('#071116')

                st.pyplot(fig)

elif "B-Scan" in analysis_type:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("B-Scan Slice Analysis (Supervised CNN)")
    st.markdown("</div>", unsafe_allow_html=True)

    model = load_bscan_model()
    if model is None:
        st.error("B-scan model not found. Put `bscan_cnn.h5` next to app.py.")
        st.stop()

    up_col, info_col = st.columns([3,1])
    with up_col:
        uploaded_img = st.file_uploader("Upload B-Scan image (jpg/png)", type=["jpg","png","jpeg"])
    with info_col:
        st.markdown("<div class='card'><b>Tips</b><br>- Crop close to retina<br>- High contrast images give better Grad-CAM</div>", unsafe_allow_html=True)

    if uploaded_img is not None:
        image_pil = Image.open(uploaded_img).convert("L")
        img_batch, proc = preprocess_bscan_image(image_pil)
        with st.spinner("Predicting..."):
            pred_raw = model.predict(img_batch, verbose=0)[0][0]
        label = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
        conf = pred_raw*100 if label=="Glaucoma-like" else (1-pred_raw)*100

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Prediction</div><div class='large-metric'>{'🚨' if 'Glaucoma' in label else '✅'} {label} — {conf:.2f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        cA, cB = st.columns([1,1.2])
        with cA:
            st.subheader("Original B-scan")
            st.image(image_pil, use_column_width=True)
        with cB:
            st.subheader("Grad-CAM Interpretation")
            heatmap = make_gradcam_heatmap(img_batch, model)
            if heatmap is not None:
                heatmap = cv2.resize(heatmap, (224,224))
                hm_img = (heatmap * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1)*255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                st.image(hm_color, caption="Heatmap", use_column_width=True)
                st.image(overlay, caption="Overlay", use_column_width=True)
            else:
                st.warning("Could not generate Grad-CAM.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#9aa4b2'>For research/demo use only — not for clinical decision making.</div>", unsafe_allow_html=True)
