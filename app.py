# app.py — OCULAIRE: Polished dark dashboard (surprise UX)
# Drop-in replacement. Keep your model filenames as-is in the same folder.

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import cv2, io, os, base64, time
import pandas as pd

# ---------------------------
# Page config & dark plotting
# ---------------------------
st.set_page_config(page_title="OCULAIRE — Glaucoma Detection", layout="wide", page_icon="👁️")
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#071116",
    "axes.facecolor": "#071116",
    "axes.edgecolor": "#e6eef8",
    "axes.labelcolor": "#e6eef8",
    "xtick.color": "#e6eef8",
    "ytick.color": "#e6eef8",
    "text.color": "#e6eef8",
    "font.size": 12,
    "axes.titleweight": "bold",
})

# ---------------------------
# CSS (modern dark dashboard)
# ---------------------------
st.markdown(
    """
    <style>
    :root{
      --bg:#06080b;
      --panel:#0b1620;
      --muted:#9aa4b2;
      --accent:#2bb7ff;
      --glass: rgba(255,255,255,0.02);
    }
    .app-wrap { padding:16px; background: linear-gradient(180deg,var(--bg), #02040a); }
    .left-nav {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding: 22px 16px; border-radius: 12px; color: #e6eef8;
    }
    .brand { font-weight:800; letter-spacing:3px; font-size:20px; margin-bottom:8px; }
    .nav-item { color:var(--muted); padding:8px 0; font-size:15px; }
    .card { background: linear-gradient(180deg, var(--panel), #071116); padding:16px; border-radius:12px; box-shadow:0 10px 40px rgba(2,6,23,0.6); }
    .uploader { background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02)); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
    .big-btn > button { background: linear-gradient(90deg,#1f8bff,#2cc5ff); color:#011418; padding:11px 22px; border-radius:10px; font-weight:700; border: none; }
    .metric-key { color:var(--muted); font-size:13px; margin-bottom:4px; }
    .metric-val { font-weight:800; font-size:20px; color:#e6eef8; }
    .badge { display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.03); color:var(--muted); font-size:12px; }
    .small-btn { background:transparent; border:1px solid rgba(255,255,255,0.06); color:var(--muted); padding:8px 10px; border-radius:8px; display:inline-block; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utility functions & loaders
# ---------------------------
@st.cache_resource
def load_bscan_model():
    try:
        model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return model
    except Exception:
        return None

@st.cache_resource
def load_rnflt_models_safe():
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        thick_cluster, thin_cluster = (1,0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0,1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception:
        return None, None, None, None, None, None

def preprocess_bscan_image(image_pil, img_size=(224,224)):
    arr = np.array(image_pil.convert("L"))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_resized = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_resized[..., None], 3, axis=-1)
    batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return batch, arr_resized

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    try:
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
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
        return heatmap.numpy()
    except Exception:
        return None

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
    except Exception:
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

def pil_to_bytes(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

# ---------------------------
# Session-state history init
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Layout: nav, center, metrics
# ---------------------------
left_col, center_col, right_col = st.columns([1.1, 4.6, 1.8], gap="large")

# Left navigation (brand + nav links)
with left_col:
    st.markdown("<div class='left-nav card'>", unsafe_allow_html=True)
    st.markdown("<div class='brand'>OCULAIRE</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Home</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Upload</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-item'>Settings</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.03)'/>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); font-size:13px'>Research/demo only — validate clinically before use.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Center: hero, uploaders, predict, visual area
with center_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin:0;color:#e6eef8'>AI-Powered Glaucoma Detection</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); margin-top:6px'>Upload a B-scan image or RNFLT map (.npz). Use Predict to analyse and produce Grad-CAM and RNFLT visualizations.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Upload panels
    up1, up2 = st.columns([1,1], gap="large")
    with up1:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#e6eef8'>B-scan Image</b>", unsafe_allow_html=True)
        bscan_file = st.file_uploader("Drag & drop or browse", type=["jpg","png","jpeg"], key="bscan", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    with up2:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#e6eef8'>RNFLT Map (.npz)</b>", unsafe_allow_html=True)
        rnflt_file = st.file_uploader("Drag & drop or browse", type=["npz"], key="rnflt", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Controls row: threshold + Predict button + clear history
    ctrl1, ctrl2, ctrl3 = st.columns([2,1,1])
    with ctrl1:
        threshold = st.slider("Thinness threshold (µm)", min_value=5, max_value=40, value=10)
    with ctrl2:
        st.markdown("<div class='big-btn'>", unsafe_allow_html=True)
        run_predict = st.button("Predict", key="predict", help="Run analysis on provided files")
        st.markdown("</div>", unsafe_allow_html=True)
    with ctrl3:
        if st.button("Clear history"):
            st.session_state.history = []
            st.success("History cleared")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Visualization row: Grad-CAM preview + RNFLT expander
    vL, vR = st.columns([1.6, 3], gap="large")
    with vL:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#e6eef8'>Grad-CAM Preview</b>", unsafe_allow_html=True)
        gradcam_area = st.empty()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with vR:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.expander("Detailed RNFLT Visualization & Metrics", expanded=False):
            rnflt_plot_area = st.empty()
            rnflt_metrics_area = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end center card

# Right: KPI metrics, toggles, history
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<b style='color:#e6eef8'>Results</b>", unsafe_allow_html=True)
    status_disp = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-key'>Severity</div>", unsafe_allow_html=True)
    sev_disp = st.empty()
    st.markdown("<div class='metric-key'>Risk score</div>", unsafe_allow_html=True)
    risk_disp = st.empty()
    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.03)'/>", unsafe_allow_html=True)
    st.markdown("<b style='color:#e6eef8'>RNFLT Summary (µm)</b>", unsafe_allow_html=True)
    mean_disp = st.empty(); sup_disp = st.empty(); inf_disp = st.empty(); temp_disp = st.empty()
    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.03)'/>", unsafe_allow_html=True)
    st.markdown("<b style='color:#e6eef8'>Analysis History</b>", unsafe_allow_html=True)
    history_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Load models (deferred)
# ---------------------------
with st.spinner("Loading models (if available)..."):
    bscan_model = load_bscan_model()
    scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models_safe()
time.sleep(0.2)

# ---------------------------
# Predict behavior
# ---------------------------
def update_history(entry):
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 25:
        st.session_state.history = st.session_state.history[:25]

if run_predict:
    # placeholders for results
    classification_label = None
    severity_val = None
    risk_val = None
    rnflt_metrics = {"mean": np.nan, "superior": np.nan, "inferior": np.nan, "temporal": np.nan}
    gradcam_img = None
    rnflt_fig = None

    # show progress
    progress = st.progress(0)
    step = 0

    # B-scan path
    if bscan_file is not None and bscan_model is not None:
        try:
            step += 1
            progress.progress(int(100 * step / 4))
            with st.spinner("Processing B-scan..."):
                pil = Image.open(bscan_file)
                batch, proc = preprocess_bscan_image(pil)
                pred_raw = bscan_model.predict(batch, verbose=0)[0][0]
                label = "Glaucoma" if pred_raw > 0.5 else "Healthy"
                confidence = pred_raw if pred_raw > 0.5 else (1 - pred_raw)
                classification_label = label
                severity_val = float(confidence * 100)
                # Grad-CAM
                heat = make_gradcam_heatmap(batch, bscan_model)
                if heat is not None:
                    heat = cv2.resize(heat, (224,224))
                    hm = (heat * 255).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                    overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                    overlay = cv2.addWeighted(overlay, 0.6, hm_color, 0.4, 0)
                    gradcam_img = overlay
        except Exception as e:
            st.error(f"B-scan processing failed: {e}")

    # RNFLT path
    if rnflt_file is not None and scaler is not None and kmeans is not None and avg_healthy is not None:
        try:
            step += 1
            progress.progress(int(100 * step / 4))
            with st.spinner("Processing RNFLT map..."):
                rnflt_map, metrics = process_uploaded_npz(rnflt_file)
                if rnflt_map is not None:
                    X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
                    X_scaled = scaler.transform(X_new)
                    cluster = int(kmeans.predict(X_scaled)[0])
                    label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                    diff, risk_map, severity = compute_risk_map(rnflt_map, avg_healthy, threshold=threshold)
                    # integrate results
                    classification_label = label_r if classification_label is None else f"{classification_label} + {label_r}"
                    severity_val = float(severity) if severity_val is None else severity_val
                    risk_val = float(np.nanmean(np.nan_to_num(risk_map))) if np.isfinite(risk_map).any() else 0.0
                    rnflt_metrics["mean"] = metrics["mean"]
                    # quadrant approximations (best-effort)
                    h,w = rnflt_map.shape
                    rnflt_metrics["superior"] = float(np.nanmean(rnflt_map[:h//2, :]))
                    rnflt_metrics["inferior"] = float(np.nanmean(rnflt_map[h//2:, :]))
                    rnflt_metrics["temporal"] = float(np.nanmean(rnflt_map[:, :w//3]))
                    # create RNFLT figure (3 panels)
                    fig, axes = plt.subplots(1,3, figsize=(14,5), constrained_layout=True)
                    im0 = axes[0].imshow(rnflt_map, cmap="turbo")
                    axes[0].set_title("Uploaded RNFLT"); axes[0].axis("off")
                    c0 = plt.colorbar(im0, ax=axes[0], shrink=0.8, label="Thickness (µm)")
                    c0.ax.yaxis.set_tick_params(color='#e6eef8'); c0.ax.yaxis.label.set_color('#e6eef8'); c0.outline.set_edgecolor('#e6eef8')

                    im1 = axes[1].imshow(diff, cmap="bwr", vmin=-30, vmax=30)
                    axes[1].set_title("Difference (vs healthy)"); axes[1].axis("off")
                    c1 = plt.colorbar(im1, ax=axes[1], shrink=0.8, label="Δ Thickness (µm)")
                    c1.ax.yaxis.set_tick_params(color='#e6eef8'); c1.ax.yaxis.label.set_color('#e6eef8'); c1.outline.set_edgecolor('#e6eef8')

                    im2 = axes[2].imshow(risk_map, cmap="hot")
                    axes[2].set_title("Risk Map"); axes[2].axis("off")
                    c2 = plt.colorbar(im2, ax=axes[2], shrink=0.8, label="Δ Thickness (µm)")
                    c2.ax.yaxis.set_tick_params(color='#e6eef8'); c2.ax.yaxis.label.set_color('#e6eef8'); c2.outline.set_edgecolor('#e6eef8')

                    fig.patch.set_facecolor("#071116")
                    for ax in axes: ax.set_facecolor("#071116")
                    rnflt_fig = fig
        except Exception as e:
            st.error(f"RNFLT processing failed: {e}")

    progress.progress(100)
    time.sleep(0.2)
    progress.empty()

    # display results in right column placeholders
    if classification_label is not None:
        status_disp.markdown(f"<div class='metric-val'>{classification_label}</div>", unsafe_allow_html=True)
    else:
        status_disp.markdown("<div class='metric-key'>No classification (missing models/inputs)</div>", unsafe_allow_html=True)

    if severity_val is not None:
        sev_disp.markdown(f"<div class='metric-val'>{severity_val:.2f}%</div>", unsafe_allow_html=True)
    else:
        sev_disp.markdown("<div class='metric-key'>—</div>", unsafe_allow_html=True)

    if risk_val is not None:
        risk_disp.markdown(f"<div class='metric-val'>{risk_val:.2f}</div>", unsafe_allow_html=True)
    else:
        risk_disp.markdown("<div class='metric-key'>—</div>", unsafe_allow_html=True)

    # RNFLT metric cards
    mean_disp.markdown(f"<div class='metric-val'>{rnflt_metrics['mean']:.2f}</div>", unsafe_allow_html=True)
    sup_disp.markdown(f"<div class='metric-key'>Superior</div><div class='metric-val'>{rnflt_metrics['superior']:.2f}</div>", unsafe_allow_html=True)
    inf_disp.markdown(f"<div class='metric-key'>Inferior</div><div class='metric-val'>{rnflt_metrics['inferior']:.2f}</div>", unsafe_allow_html=True)
    temp_disp.markdown(f"<div class='metric-key'>Temporal</div><div class='metric-val'>{rnflt_metrics['temporal']:.2f}</div>", unsafe_allow_html=True)

    # Grad-CAM preview
    if gradcam_img is not None:
        gradcam_area.image(gradcam_img, use_column_width=True, caption="Grad-CAM (B-scan)")
        # enlarge option - present downloadable PNG and PDF
        col_dl1, col_dl2 = st.columns([1,1])
        with col_dl1:
            buf = cv2.imencode('.png', gradcam_img)[1].tobytes()
            b64 = base64.b64encode(buf).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="gradcam.png" class="small-btn">Download Grad-CAM PNG</a>'
            st.markdown(href, unsafe_allow_html=True)
        with col_dl2:
            # include in PDF below if RNFLT exists
            st.markdown("<div class='small-btn'>Grad-CAM ready</div>", unsafe_allow_html=True)
    else:
        gradcam_area.markdown("<div style='color:var(--muted)'>Grad-CAM preview will appear here when a B-scan model and image are available.</div>", unsafe_allow_html=True)

    # RNFLT figure show
    if 'rnflt_fig' in locals() and rnflt_fig is not None:
        rnflt_plot_area.pyplot(rnflt_fig)
        # provide downloads: PNG + PDF (RNFLT + optional Grad-CAM)
        colp1, colp2 = st.columns([1,1])
        with colp1:
            buf = io.BytesIO()
            rnflt_fig.savefig(buf, format='png', bbox_inches='tight', facecolor=rnflt_fig.get_facecolor())
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="rnflt_viz.png" class="small-btn">Download RNFLT PNG</a>'
            st.markdown(href, unsafe_allow_html=True)
        with colp2:
            # generate PDF combining RNFLT figure and gradcam (if any)
            pdf_buf = io.BytesIO()
            with PdfPages(pdf_buf) as pdf:
                pdf.savefig(rnflt_fig, bbox_inches='tight', facecolor=rnflt_fig.get_facecolor())
                if gradcam_img is not None:
                    # convert gradcam to PIL then to matplotlib figure for PDF
                    grad_pil = Image.fromarray(gradcam_img)
                    fig2, ax2 = plt.subplots(figsize=(6,6))
                    ax2.imshow(grad_pil)
                    ax2.axis('off')
                    fig2.patch.set_facecolor("#071116")
                    pdf.savefig(fig2, bbox_inches='tight', facecolor=fig2.get_facecolor())
                    plt.close(fig2)
            pdf_buf.seek(0)
            b64pdf = base64.b64encode(pdf_buf.read()).decode()
            href_pdf = f'<a href="data:application/octet-stream;base64,{b64pdf}" download="oculaire_report.pdf" class="small-btn">Download PDF Report</a>'
            st.markdown(href_pdf, unsafe_allow_html=True)
    else:
        rnflt_plot_area.markdown("<div style='color:var(--muted)'>RNFLT visualizations will appear here when you upload an RNFLT .npz file.</div>", unsafe_allow_html=True)

    # record to session history
    history_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classification": classification_label,
        "severity": float(severity_val) if severity_val is not None else None,
        "rnflt_mean": float(rnflt_metrics['mean']) if not np.isnan(rnflt_metrics['mean']) else None
    }
    update_history(history_entry)

# Render history box
if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    history_box.dataframe(df_hist, hide_index=True, width=320)
else:
    history_box.markdown("<div style='color:var(--muted)'>No analyses yet — run Predict to create history entries.</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted)'>Built for research/demo only — not for clinical use.</div>", unsafe_allow_html=True)
