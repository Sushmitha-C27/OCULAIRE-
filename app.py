# app.py — OCULAIRE polished dark dashboard (drop-in)
# Keeps all models/filenames unchanged. Paste over your current app.py and run.
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2, io, os
import base64
import pandas as pd

# -------------------------
# Basic page + dark plotting
# -------------------------
st.set_page_config(page_title="OCULAIRE: Glaucoma Detection", layout="wide", page_icon="👁️")

# Matplotlib dark setup so plots & colorbars are readable
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0b1220",
    "axes.facecolor": "#0b1220",
    "axes.labelcolor": "#e6eef8",
    "xtick.color": "#e6eef8",
    "ytick.color": "#e6eef8",
    "text.color": "#e6eef8",
    "axes.titleweight": "bold",
    "font.size": 12,
})

# -------------------------
# CSS (polish + layout)
# -------------------------
st.markdown(
    """
    <style>
    :root {
      --bg: #071016;
      --panel: #0c1620;
      --muted: #9aa4b2;
      --accent: #2b9edb;
      --glass: rgba(255,255,255,0.02);
    }
    .app-body { background: linear-gradient(180deg,var(--bg), #02040a); padding:16px; }
    .left-nav {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding: 22px 12px;
        border-radius: 12px;
        height: calc(100vh - 48px);
        color: #e6eef8;
    }
    .brand { font-weight:800; letter-spacing:3px; font-size:20px; margin-bottom:10px; }
    .nav-item { color:var(--muted); padding:8px 0; font-size:15px; }
    .card { background: linear-gradient(180deg, var(--panel), #071018); padding:18px; border-radius:12px; box-shadow: 0 10px 40px rgba(2,6,23,0.6); }
    .uploader {
        background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
        padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.03);
    }
    .big-btn > button { background: linear-gradient(90deg,#1e90ff,#2fb6ff); color:#031418; padding:12px 22px; border-radius:10px; font-weight:700; border:none; }
    .metric-key { color:var(--muted); font-size:13px; }
    .metric-val { font-weight:800; font-size:20px; color:#e6eef8; }
    .small-btn { background:transparent; border:1px solid rgba(255,255,255,0.06); color:var(--muted); padding:8px 10px; border-radius:8px; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers: model loaders & image utilities
# -------------------------
@st.cache_resource
def load_bscan_model():
    try:
        model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return model
    except Exception as e:
        st.error("B-Scan model not found or failed to load.")
        return None

@st.cache_resource
def load_rnflt_models_safe():
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        thick_cluster, thin_cluster = (1, 0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0, 1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception as e:
        # don't spam errors; return Nones
        return None, None, None, None, None, None

def preprocess_bscan_image(image_pil, img_size=(224,224)):
    arr = np.array(image_pil.convert("L"))
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
        return None
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()

def process_uploaded_npz(uploaded_file):
    try:
        bio = io.BytesIO(uploaded_file.getvalue())
        npz = np.load(bio, allow_pickle=True)
        rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
        if rnflt_map.ndim == 3:
            rnflt_map = rnflt_map[0,:,:]
        vals = rnflt_map.flatten().astype(float)
        metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
        return rnflt_map, metrics
    except Exception as e:
        st.error("Could not read .npz file.")
        return None, None

def compute_risk_map(rnflt_map, healthy_avg, threshold=-10):
    if rnflt_map.shape != healthy_avg.shape:
        healthy_avg = cv2.resize(healthy_avg, (rnflt_map.shape[1], rnflt_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < threshold, diff, np.nan)
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky/total)*100 if total>0 else 0
    return diff, risk, severity

# -------------------------
# Layout: left nav, center content, right metrics
# -------------------------
left_col, center_col, right_col = st.columns([1.2, 4.5, 1.8], gap="large")

# Left nav
with left_col:
    st.markdown("<div class='left-nav card'>", unsafe_allow_html=True)
    st.markdown("<div class='brand'>OCULAIRE</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Home</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Upload</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Settings</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.03)'/>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); font-size:13px; margin-top:8px'>For research use only — validate clinically before use.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Center: main controls & visual
with center_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin:0;color:#e6eef8'>AI-Powered Glaucoma Detection</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted);margin-bottom:12px'>Upload a B-scan image and/or RNFLT map, then press Predict.</div>", unsafe_allow_html=True)

    # Uploaders side-by-side
    up_col1, up_col2 = st.columns(2, gap="large")
    with up_col1:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#e6eef8'>B-scan Image</b>", unsafe_allow_html=True)
        bscan_file = st.file_uploader("Drag & drop file or browse", type=["jpg","png","jpeg"], key="bscan", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with up_col2:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#e6eef8'>RNFLT Map (.npz)</b>", unsafe_allow_html=True)
        rnflt_file = st.file_uploader("Drag & drop file or browse", type=["npz"], key="rnflt", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Predict button - style wrapper to make large gradient button
    st.markdown("<div class='big-btn'>", unsafe_allow_html=True)
    run_predict = st.button("Predict", key="predict", help="Run selected analysis (B-scan / RNFLT)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Visualization area: left small Grad-CAM + big expander for full plots
    viz_col_left, viz_col_right = st.columns([1.8, 3], gap="large")
    with viz_col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<b>Grad-CAM</b>", unsafe_allow_html=True)
        gradcam_placeholder = st.empty()
        # action buttons
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Download Report", key="download_report"):
                st.info("Report generation not implemented — placeholder.")
        with c2:
            if st.button("Export DICOM", key="export_dcm"):
                st.info("Export DICOM placeholder.")
        st.markdown("</div>", unsafe_allow_html=True)

    with viz_col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.expander("Detailed RNFLT Visualization & Charts", expanded=False):
            plot_area = st.empty()
            # the actual plotting will occur after Predict is clicked
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close main card

# Right: metrics card
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<b style='color:#e6eef8'>Classification</b>", unsafe_allow_html=True)
    status_text = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-key'>Severity</div>", unsafe_allow_html=True)
    sev_text = st.empty()
    st.markdown("<div class='metric-key'>Risk Score</div>", unsafe_allow_html=True)
    risk_text = st.empty()

    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.03)'/>", unsafe_allow_html=True)
    st.markdown("<b style='color:#e6eef8'>RNFLT Metrics (µm)</b>", unsafe_allow_html=True)
    mean_text = st.empty()
    sup_text = st.empty()
    inf_text = st.empty()
    temp_text = st.empty()

    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.03)'/>", unsafe_allow_html=True)
    st.markdown("<b style='color:#e6eef8'>Analysis History</b>", unsafe_allow_html=True)
    st.markdown("Model Version v1.0.0", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    # toggles and slider
    st.checkbox("Include fundus/OCT", key="incl_fundus")
    st.checkbox("Batch upload", key="batch_upl")
    st.slider("Threshold (µm)", min_value=1, max_value=30, value=10, key="threshold")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Predict logic (run when button clicked)
# -------------------------
bscan_model = load_bscan_model()
scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models_safe()

# Prepare placeholders for plots to update after prediction
if run_predict:
    # Priority: if B-scan provided, run B-scan pipeline; also run RNFLT if provided
    classification_result = None
    severity_pct = None
    risk_score_val = None
    rnflt_metrics_out = {"mean": np.nan, "superior": np.nan, "inferior": np.nan, "temporal": np.nan}
    gradcam_img_to_show = None
    rnflt_fig = None

    # --- B-SCAN processing (if provided) ---
    if bscan_file is not None and bscan_model is not None:
        try:
            pil = Image.open(bscan_file)
            img_batch, proc = preprocess_bscan_image(pil)
            pred_raw = bscan_model.predict(img_batch, verbose=0)[0][0]
            label = "Glaucoma" if pred_raw > 0.5 else "Healthy"
            conf = pred_raw if pred_raw > 0.5 else (1 - pred_raw)
            classification_result = f"{label}"
            severity_pct = conf * 100
            # Grad-CAM
            heatmap = make_gradcam_heatmap(img_batch, bscan_model)
            if heatmap is not None:
                hm_resized = cv2.resize(heatmap, (224,224))
                hm_img = (hm_resized * 255).astype("uint8")
                hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1)*255).astype("uint8")
                overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                gradcam_img_to_show = overlay
        except Exception as e:
            st.error("B-scan processing failed.")

    # --- RNFLT processing (if provided) ---
    if rnflt_file is not None and scaler is not None and kmeans is not None and avg_healthy is not None:
        rnflt_map, metrics = process_uploaded_npz(rnflt_file)
        if rnflt_map is not None:
            X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            X_scaled = scaler.transform(X_new)
            cluster = int(kmeans.predict(X_scaled)[0])
            label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
            diff, risk, severity = compute_risk_map(rnflt_map, avg_healthy, threshold=st.session_state.get("threshold", 10))
            # summary
            classification_result = label_r if classification_result is None else classification_result + " + " + label_r
            severity_pct = (severity if severity_pct is None else severity_pct)
            risk_score_val = float(np.nanmean(np.nan_to_num(risk))) if np.isfinite(risk).any() else 0
            # some RNFLT metrics (mean + rough quadrants if available)
            rnflt_metrics_out["mean"] = metrics["mean"]
            # attempt simple quadrant approximations (if square)
            h, w = rnflt_map.shape
            try:
                sup = np.nanmean(rnflt_map[:h//2, :])
                inf = np.nanmean(rnflt_map[h//2:, :])
                temp = np.nanmean(rnflt_map[:, :w//3])
                rnflt_metrics_out.update({"superior": float(sup), "inferior": float(inf), "temporal": float(temp)})
            except:
                pass

            # create side-by-side RNFLT visualization fig
            fig, axes = plt.subplots(1,3, figsize=(14,5), constrained_layout=True)
            im0 = axes[0].imshow(rnflt_map, cmap="turbo")
            axes[0].axis("off"); axes[0].set_title("Uploaded RNFLT")
            c0 = plt.colorbar(im0, ax=axes[0], shrink=0.8, label="Thickness (µm)")
            im1 = axes[1].imshow(diff, cmap="bwr", vmin=-30, vmax=30)
            axes[1].axis("off"); axes[1].set_title("Difference (vs healthy)")
            c1 = plt.colorbar(im1, ax=axes[1], shrink=0.8, label="Δ Thickness (µm)")
            im2 = axes[2].imshow(risk, cmap="hot")
            axes[2].axis("off"); axes[2].set_title("Risk Map")
            c2 = plt.colorbar(im2, ax=axes[2], shrink=0.8, label="Δ Thickness (µm)")
            fig.patch.set_facecolor("#071116")
            for ax in axes: ax.set_facecolor("#071116")
            rnflt_fig = fig

    # --- Update UI placeholders with results ---
    # Classification block
    if classification_result is not None:
        status_text.markdown(f"<div class='metric-val'>{classification_result}</div>", unsafe_allow_html=True)
    else:
        status_text.markdown("<div class='metric-key'>No classification (no models or inputs)</div>", unsafe_allow_html=True)

    # Severity & risk
    if severity_pct is not None:
        sev_text.markdown(f"<div class='metric-val'>{severity_pct:.2f}%</div>", unsafe_allow_html=True)
    else:
        sev_text.markdown("<div class='metric-key'>—</div>", unsafe_allow_html=True)

    if risk_score_val is not None:
        risk_text.markdown(f"<div class='metric-val'>{risk_score_val:.2f}</div>", unsafe_allow_html=True)
    else:
        risk_text.markdown("<div class='metric-key'>—</div>", unsafe_allow_html=True)

    # RNFLT metrics
    mean_text.markdown(f"<div class='metric-val'>{rnflt_metrics_out.get('mean', np.nan):.2f}</div>", unsafe_allow_html=True)
    sup_text.markdown(f"<div class='metric-key'>Superior</div><div class='metric-val'>{rnflt_metrics_out.get('superior', np.nan):.2f}</div>", unsafe_allow_html=True)
    inf_text.markdown(f"<div class='metric-key'>Inferior</div><div class='metric-val'>{rnflt_metrics_out.get('inferior', np.nan):.2f}</div>", unsafe_allow_html=True)
    temp_text.markdown(f"<div class='metric-key'>Temporal</div><div class='metric-val'>{rnflt_metrics_out.get('temporal', np.nan):.2f}</div>", unsafe_allow_html=True)

    # Grad-CAM preview
    if gradcam_img_to_show is not None:
        gradcam_placeholder.image(gradcam_img_to_show, use_column_width=True, caption="Grad-CAM (B-scan)")
    else:
        gradcam_placeholder.markdown("<div style='color:var(--muted)'>Grad-CAM preview will appear here when B-scan is provided.</div>", unsafe_allow_html=True)

    # RNFLT plots
    if rnflt_fig is not None:
        plot_area.pyplot(rnflt_fig)
    else:
        plot_area.markdown("<div style='color:var(--muted)'>RNFLT visualizations appear here when an RNFLT .npz is uploaded.</div>", unsafe_allow_html=True)

    # Optionally provide download (image/PDF) functionality for the RNFLT figure
    if rnflt_fig is not None:
        buf = io.BytesIO()
        rnflt_fig.savefig(buf, format="png", bbox_inches="tight", facecolor=rnflt_fig.get_facecolor())
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:file/png;base64,{b64}" download="rnflt_viz.png" class="small-btn">Download RNFLT PNG</a>'
        st.markdown(href, unsafe_allow_html=True)

# -------------------------
# end
# -------------------------
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted)'>For research/demo only — not for clinical use.</div>", unsafe_allow_html=True)
