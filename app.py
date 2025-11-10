# app.py (visual polish version)
# Replace your current app.py with this file. Logic unchanged — only layout/CSS/UX improved.

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

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(page_title="OCULAIRE — Glaucoma Detection", layout="wide", page_icon="👁️")

# --------------------------------
# CSS: theme, card, uploader styling
# --------------------------------
st.markdown(
    """
    <style>
    /* Page */
    .stApp { background: linear-gradient(180deg,#f6fbff 0%, #ffffff 100%); font-family: Inter, Roboto, Arial, sans-serif; }
    /* Top bar */
    .topbar { display:flex; align-items:center; gap:1rem; padding:0.6rem 1rem; border-radius:12px; background:rgba(0,0,0,0.6); color: white; margin-bottom:1rem; }
    .topbar h1 { margin:0; font-size:20px; letter-spacing:0.4px; font-weight:600; }
    .topbar p { margin:0; opacity:0.85; font-size:12px; }
    /* Cards */
    .card { background:white; padding:14px; border-radius:12px; box-shadow:0 8px 24px rgba(15,23,42,0.06); margin-bottom:12px; }
    .muted { color:#6b7280; font-size:0.9rem; }
    /* Uploader - bigger rounded panel */
    .uploader-card { background:#0f1724; color:#ffffff; padding:18px; border-radius:12px; text-align:center; }
    .uploader-card .desc { color:#c7d2fe; margin-top:8px; font-size:13px; }
    /* Hide default streamlit footer */
    footer { visibility: hidden; }
    /* Metrics compact */
    .metric-label { font-weight:600; color:#374151; }
    </style>
    """, unsafe_allow_html=True
)

# --------------------------------
# Top banner
# --------------------------------
logo_path = "assets/logo.png"
left_col, mid_col, right_col = st.columns([1, 8, 1])
with left_col:
    if os.path.exists(logo_path):
        st.image(logo_path, width=72)
with mid_col:
    st.markdown("<div class='topbar'><div><h1>👁️ OCULAIRE</h1><p>Automated glaucoma screening — RNFLT maps & B-scan CNN</p></div></div>", unsafe_allow_html=True)
with right_col:
    st.write("")

# --------------------------------
# Sidebar: controls + demo
# --------------------------------
with st.sidebar:
    st.title("Analysis Controls")
    analysis_type = st.radio(
        "Select analysis type",
        ("🩺 RNFLT Map Analysis (.npz)", "👁️ B-Scan Slice Analysis (Image)"),
        index=0
    )
    st.markdown("---")
    st.markdown("**Quick demo**")
    if st.button("Load example RNFLT"):
        st.session_state["_load_example_rnflt"] = True
    if st.button("Load example B-scan"):
        st.session_state["_load_example_bscan"] = True
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Upload high-quality images for best Grad-CAM.")
    st.markdown("- RNFLT shape should roughly match average maps; the app will resize if needed.")
    st.markdown("---")
    st.caption("Place model/data files next to app.py: `bscan_cnn.h5`, `rnflt_scaler.joblib`, `rnflt_kmeans.joblib`, `avg_map_healthy.npy`, `avg_map_glaucoma.npy`")

# --------------------------------
# Model loaders (unchanged logic)
# --------------------------------
@st.cache_resource
def load_bscan_model():
    try:
        with st.spinner("Loading B-Scan model..."):
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
        # heuristic mapping
        thick_cluster, thin_cluster = (1, 0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0, 1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception as e:
        st.warning("RNFLT artifacts missing or failed to load.")
        return None, None, None, None, None, None

# --------------------------------
# RNFLT helper functions (same as yours)
# --------------------------------
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

# --------------------------------
# B-scan helper functions (same as yours)
# --------------------------------
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

# --------------------------------
# Main panels (visual improvements)
# --------------------------------
if "RNFLT" in analysis_type:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("RNFLT Map Analysis — Unsupervised")
    st.markdown("<div style='display:flex; gap:12px; align-items:center;'>", unsafe_allow_html=True)

    # Big rounded uploader panel
    uploader_col, preview_col = st.columns([3, 2])
    with uploader_col:
        st.markdown("<div class='uploader-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0;color:#fff'>Drop RNFLT .npz here</h3>", unsafe_allow_html=True)
        st.markdown("<div class='desc'>Drag & drop or click to choose a file — max 200MB</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["npz"], key="rnflt_uploader", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        # Example load using session flag (button in sidebar)
        if st.session_state.get("_load_example_rnflt", False):
            example_path = "assets/example_rnflt.npz"
            if os.path.exists(example_path):
                with open(example_path, "rb") as f:
                    example_bytes = f.read()
                uploaded_file = st.file_uploader("", type=["npz"], key="rnflt_uploader_reload", label_visibility="collapsed")
                # streamlit doesn't allow programmatic file upload; show message instead
                st.success("Place `assets/example_rnflt.npz` in the folder to use example. (Button demonstrates intent)")
            else:
                st.error("No example file found at assets/example_rnflt.npz")

    with preview_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Upload status**")
        if uploaded_file is None:
            st.write("No file uploaded yet.")
            st.markdown("You can also use the B-Scan analysis from the sidebar.")
        else:
            st.write(f"File: {uploaded_file.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # close outer card

    # proceed if file exists
    if uploaded_file is not None:
        rnflt_map, metrics = process_uploaded_npz(uploaded_file)
        scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models_safe()
        if scaler is None:
            st.error("RNFLT artifact files are missing. See sidebar for required filenames.")
        else:
            X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            X_scaled = scaler.transform(X_new)
            cluster = int(kmeans.predict(X_scaled)[0])
            label = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            diff, risk, severity = compute_risk_map(rnflt_map, avg_healthy)

            # Compact metric cards
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            emoji = "🚨" if label == "Glaucoma-like" else "✅"
            mcol1.metric("Status", f"{emoji} {label}")
            mcol2.metric("Mean RNFLT (µm)", f"{metrics['mean']:.2f}")
            mcol3.metric("Severity %", f"{severity:.2f}%")
            mcol4.metric("Cluster", f"{cluster}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Visuals: large horizontal plots
            with st.expander("🔬 Detailed RNFLT Visualization", expanded=True):
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

                im0 = axes[0].imshow(rnflt_map, cmap='turbo')
                axes[0].set_title("Uploaded RNFLT Map")
                axes[0].axis("off")
                plt.colorbar(im0, ax=axes[0], shrink=0.85, label="Thickness (µm)")

                im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30)
                axes[1].set_title("Difference Map (vs Healthy)")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], shrink=0.85, label="Δ Thickness (µm)")

                im2 = axes[2].imshow(risk, cmap='hot')
                axes[2].set_title("Risk Map (Thinner Zones)")
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], shrink=0.85, label="Δ Thickness (µm)")

                st.pyplot(fig)

            # Numeric table + histogram
            with st.expander("📊 Metrics & Distribution", expanded=False):
                st.dataframe(pd.DataFrame([metrics]).round(2))
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.hist(rnflt_map.flatten(), bins=60)
                ax2.set_xlabel("Thickness (µm)"); ax2.set_ylabel("Pixel count"); ax2.set_title("RNFLT distribution")
                st.pyplot(fig2)

elif "B-Scan" in analysis_type:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("B-Scan Slice Analysis — Supervised CNN")
    st.markdown("</div>", unsafe_allow_html=True)

    model = load_bscan_model()
    if model is None:
        st.error("B-scan model not found. Place `bscan_cnn.h5` next to app.py.")
        st.stop()

    # uploader
    up_col, info_col = st.columns([3, 1])
    with up_col:
        uploaded_img = st.file_uploader("Upload B-Scan image (jpg/png)", type=["jpg", "png", "jpeg"])
        if st.session_state.get("_load_example_bscan", False):
            example_bs = "assets/example_bscan.jpg"
            if os.path.exists(example_bs):
                st.info("Example available at assets/example_bscan.jpg")
            else:
                st.error("No example at assets/example_bscan.jpg")

    with info_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Tips")
        st.write("- Crop close to retina")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_img is not None:
        image_pil = Image.open(uploaded_img).convert("L")
        img_batch, processed_disp = preprocess_bscan_image(image_pil)
        with st.spinner("Running prediction..."):
            pred_raw = model.predict(img_batch, verbose=0)[0][0]
        label = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
        conf = pred_raw*100 if label=="Glaucoma-like" else (1-pred_raw)*100

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Prediction", f"{'🚨' if label.startswith('Glaucoma') else '✅'} {label}", delta=f"{conf:.2f}% Confidence")
        st.markdown("</div>", unsafe_allow_html=True)

        colA, colB = st.columns([1, 1.2])
        with colA:
            st.subheader("Original B-scan")
            st.image(image_pil, use_column_width=True)
        with colB:
            st.subheader("Grad-CAM Interpretation")
            heatmap = make_gradcam_heatmap(img_batch, model)
            if heatmap is not None:
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap_img = (heatmap * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                superimposed_img = (np.stack([processed_disp]*3, axis=-1) * 255).astype(np.uint8)
                superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap_color, 0.4, 0)
                st.image(heatmap_color, caption="Heatmap", use_column_width=True)
                st.image(superimposed_img, caption="Overlay", use_column_width=True)
            else:
                st.warning("Unable to compute Grad-CAM for this model.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("<div style='text-align:center;color:#6b7280'>For research/demo use only — not for clinical decision making.</div>", unsafe_allow_html=True)
