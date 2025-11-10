# app.py
# OCULAIRE — "Assistant's Dream" UI
# Drop-in replacement. Keeps your model/data filenames unchanged.
# Run: streamlit run app.py

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import io, os, time, base64
import cv2
import matplotlib.pyplot as plt

# Try to import plotly (interactive RNFLT heatmap). If not available, fallback to matplotlib.
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ---------------------------
# Page and Matplotlib style
# ---------------------------
st.set_page_config(page_title="OCULAIRE — Illuminated", layout="wide", page_icon="👁️")

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#071021",
    "axes.facecolor": "#071021",
    "text.color": "#EAF2FF",
    "xtick.color": "#EAF2FF",
    "ytick.color": "#EAF2FF",
    "axes.labelcolor": "#EAF2FF",
})

# ---------------------------
# Elegant "midnight glass" CSS
# ---------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
      :root{
        --bg:#060812;
        --panel: rgba(255,255,255,0.03);
        --glass: rgba(255,255,255,0.025);
        --accent: #2bd4ff;
        --muted: #9fb1c9;
        --glass-2: rgba(255,255,255,0.02);
      }
      html, body, #root, .appview-container, .main {
        background: linear-gradient(180deg,var(--bg), #02040a) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      }
      .topbar {
        display:flex; align-items:center; justify-content:space-between;
        padding:12px 18px; border-radius:12px;
        background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
        margin-bottom:18px;
      }
      .brand { font-weight:800; color:#EAF2FF; font-size:20px; letter-spacing:1px;}
      .subtitle { color:var(--muted); font-size:13px; margin-top:2px;}
      .rail { background: transparent; padding-top:18px; }
      .icon-btn {
        display:flex; align-items:center; justify-content:center;
        width:56px;height:56px; margin:8px auto; border-radius:12px;
        background: var(--glass); color:#EAF2FF; border:1px solid rgba(255,255,255,0.03);
      }
      .glass-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
                    border-radius:12px; padding:14px; color:#EAF2FF; border:1px solid rgba(255,255,255,0.03);}
      .kpi { font-size:20px; font-weight:700; color:#EAF2FF; }
      .kpi-label { font-size:12px; color:var(--muted); }
      .muted { color:var(--muted); }
      .big-primary > button { background: linear-gradient(90deg,#1ec2ff,#6ad2ff); color:#021518; border-radius:10px; padding:10px 18px; font-weight:700; }
      footer { visibility:hidden; }
      .tiny { font-size:12px; color:var(--muted); }
      .action-btn { background:transparent; border:1px solid rgba(255,255,255,0.04); padding:8px 12px; border-radius:8px; color:var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utility: model loaders & preprocess
# ---------------------------
@st.cache_resource
def load_bscan_model():
    try:
        return tf.keras.models.load_model("bscan_cnn.h5", compile=False)
    except Exception:
        return None

@st.cache_resource
def load_rnflt_artifacts():
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        # map cluster identities heuristically
        thick_cluster, thin_cluster = (1, 0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0, 1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception:
        return None, None, None, None, None, None

def preprocess_bscan_image(pil_img, img_size=(224,224)):
    arr = np.array(pil_img.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(rgb, axis=0).astype(np.float32)
    return batch, arr_res

def make_gradcam_heatmap(batch, model, last_conv_name=None):
    try:
        # pick last conv if not provided
        if last_conv_name is None:
            for l in reversed(model.layers):
                if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    last_conv_name = l.name
                    break
        grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(batch)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]
        heat = conv_out @ pooled[..., tf.newaxis]
        heat = tf.squeeze(heat)
        heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-6)
        return heat.numpy()
    except Exception:
        return None

def read_npz_bytes(uploaded_file):
    try:
        b = io.BytesIO(uploaded_file.getvalue())
        arrs = np.load(b, allow_pickle=True)
        # prefer 'volume' otherwise first array
        if "volume" in arrs:
            m = arrs["volume"]
        else:
            first = arrs.files[0]
            m = arrs[first]
        if m.ndim == 3:
            m = m[0, :, :]
        return m
    except Exception:
        return None

def compute_risk_map(rnflt_map, healthy_avg, thresh=-10):
    if rnflt_map.shape != healthy_avg.shape:
        healthy_avg = cv2.resize(healthy_avg, (rnflt_map.shape[1], rnflt_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < thresh, diff, np.nan)
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky/total)*100 if total>0 else 0
    return diff, risk, severity

# ---------------------------
# Top bar
# ---------------------------
top1, top2 = st.columns([1, 9])
with top1:
    st.markdown("<div class='brand'>OCULAIRE</div><div class='subtitle tiny'>Illuminating vision with ML</div>", unsafe_allow_html=True)
with top2:
    # small right-side user area (placeholder)
    st.markdown("<div style='text-align:right;'><span class='tiny muted'>Assistant edition • midnight theme</span></div>", unsafe_allow_html=True)

st.markdown("")  # spacer

# ---------------------------
# Main layout: rail | canvas | metrics
# ---------------------------
rail, canvas, metrics = st.columns([0.85, 6.3, 2.0], gap="large")

# Left rail (icon-only)
with rail:
    st.markdown("<div class='rail'>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>🏠</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>⬆️</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>📊</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>⚙️</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='tiny muted'>For research only — not clinical</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Center canvas (hero area)
with canvas:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:4px 0 6px 0'>AI-powered Glaucoma Insights</h2>", unsafe_allow_html=True)
    st.markdown("<div class='muted tiny'>Upload an OCT RNFLT map (.npz) and/or a B-scan image (jpg/png). Then Predict.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Upload tiles (two columns)
    up1, up2 = st.columns([1,1], gap="large")
    with up1:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#EAF2FF'>B-scan (jpg / png)</b>", unsafe_allow_html=True)
        bscan_file = st.file_uploader("", type=["jpg","png","jpeg"], key="u_bscan", label_visibility="collapsed")
        st.markdown("<div class='muted tiny' style='margin-top:6px'>High contrast B-scan cropped to retina works best.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with up2:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#EAF2FF'>RNFLT (.npz)</b>", unsafe_allow_html=True)
        rnflt_file = st.file_uploader("", type=["npz"], key="u_rnflt", label_visibility="collapsed")
        st.markdown("<div class='muted tiny' style='margin-top:6px'>App will handle typical RNFLT shapes (resizing if needed).</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Controls row: threshold + predict
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        threshold = st.slider("Thinness threshold (µm)", 5, 50, 10)
    with c2:
        st.markdown("<div class='big-primary'>", unsafe_allow_html=True)
        predict = st.button("Predict")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Display area: left small Grad-CAM and right interactive RNFLT container
    left_viz, right_viz = st.columns([2,3], gap="large")
    with left_viz:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<b>Grad-CAM</b>", unsafe_allow_html=True)
        gradcam_placeholder = st.empty()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_viz:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<b>RNFLT Interactive View</b>", unsafe_allow_html=True)
        rnflt_placeholder = st.empty()
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close hero glass-card

# Right metrics column
with metrics:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<b style='color:#EAF2FF'>Summary</b>", unsafe_allow_html=True)
    kpi_status = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-key'>Severity</div>", unsafe_allow_html=True)
    kpi_sev = st.empty()
    st.markdown("<div class='metric-key'>Risk score (mean)</div>", unsafe_allow_html=True)
    kpi_risk = st.empty()
    st.markdown("<hr style='border-top:1px solid rgba(255,255,255,0.04)'/>", unsafe_allow_html=True)
    st.markdown("<b style='color:#EAF2FF'>RNFLT Metrics (µm)</b>", unsafe_allow_html=True)
    k_mean = st.empty()
    k_sup = st.empty()
    k_inf = st.empty()
    k_temp = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<button class='action-btn' onclick='window.scrollTo(0,0)'>New Analysis</button>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Load models/resources
# ---------------------------
with st.spinner("Preparing assistant layout and loading artifacts..."):
    bscan_model = load_bscan_model()
    scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_artifacts()
time.sleep(0.2)

# ---------------------------
# Helper: save matplotlib fig to bytes for download
# ---------------------------
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# ---------------------------
# Predict logic (runs when user clicks Predict)
# ---------------------------
if predict:
    # placeholders & initial values
    status_text = "No result"
    severity_val = None
    risk_mean = None
    rnflt_metrics = {"mean": np.nan, "superior": np.nan, "inferior": np.nan, "temporal": np.nan}
    gradcam_img = None
    rnflt_map = None
    rnflt_fig = None

    progress = st.progress(0)
    step = 0

    # B-scan path
    if bscan_file is not None and bscan_model is not None:
        step += 1; progress.progress(int(100 * step / 4))
        try:
            pil = Image.open(bscan_file)
            batch, proc = preprocess_bscan_image(pil)
            pred_raw = bscan_model.predict(batch, verbose=0)[0][0]
            label = "Glaucoma" if pred_raw > 0.5 else "Healthy"
            conf = pred_raw if pred_raw > 0.5 else (1 - pred_raw)
            status_text = label
            severity_val = float(conf * 100)
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
        step += 1; progress.progress(int(100 * step / 4))
        try:
            rnflt_map = read_npz_bytes(rnflt_file)
            if rnflt_map is not None:
                vals = rnflt_map.flatten().astype(float)
                metrics = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}
                # classification
                X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
                X_scaled = scaler.transform(X_new)
                cluster = int(kmeans.predict(X_scaled)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                if status_text == "No result":
                    status_text = label_r
                else:
                    status_text = f"{status_text} + {label_r}"
                # risk map & severity
                diff, risk_map, severity = compute_risk_map(rnflt_map, avg_healthy, thresh=-threshold)
                severity_val = float(severity) if severity_val is None else severity_val
                risk_mean = float(np.nanmean(np.nan_to_num(risk_map))) if np.isfinite(risk_map).any() else 0.0
                # basic RNFLT metrics & quadrants (best-effort)
                rnflt_metrics["mean"] = metrics["mean"]
                h,w = rnflt_map.shape
                rnflt_metrics["superior"] = float(np.nanmean(rnflt_map[:h//2, :]))
                rnflt_metrics["inferior"] = float(np.nanmean(rnflt_map[h//2:, :]))
                rnflt_metrics["temporal"] = float(np.nanmean(rnflt_map[:, :w//3]))
                # interactive RNFLT: try Plotly first
                if PLOTLY_AVAILABLE:
                    fig_px = px.imshow(rnflt_map, color_continuous_scale="Turbo", origin='lower')
                    fig_px.update_layout({"paper_bgcolor":"#071116","plot_bgcolor":"#071116","coloraxis_colorbar":{"title":"Thickness (µm)","ticks":"outside","tickfont":{"color":"#EAF2FF"}}})
                    rnflt_fig = fig_px
                else:
                    # fallback: matplotlib fig
                    fig, axes = plt.subplots(1,3, figsize=(13,5), constrained_layout=True)
                    im0 = axes[0].imshow(rnflt_map, cmap='turbo'); axes[0].axis('off'); axes[0].set_title("Uploaded RNFLT")
                    im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30); axes[1].axis('off'); axes[1].set_title("Difference")
                    im2 = axes[2].imshow(risk_map, cmap='hot'); axes[2].axis('off'); axes[2].set_title("Risk map")
                    for c in (im0, im1, im2):
                        cb = fig.colorbar(c, ax=axes.ravel().tolist(), shrink=0.8) if False else None
                    fig.patch.set_facecolor('#071116')
                    rnflt_fig = fig
        except Exception as e:
            st.error(f"RNFLT processing failed: {e}")

    progress.progress(100)
    time.sleep(0.15)
    progress.empty()

    # ---------------------------
    # update right column KPIs
    # ---------------------------
    kpi_status.markdown(f"<div class='kpi'>{status_text}</div>", unsafe_allow_html=True)
    kpi_sev.markdown(f"<div class='kpi'>{'%.2f' % severity_val + '%' if severity_val is not None else '—'}</div>", unsafe_allow_html=True)
    kpi_risk.markdown(f"<div class='kpi'>{'%.2f' % risk_mean if risk_mean is not None else '—'}</div>", unsafe_allow_html=True)
    k_mean.markdown(f"<div class='kpi'>{rnflt_metrics['mean']:.2f}</div>", unsafe_allow_html=True)
    k_sup.markdown(f"<div class='metric-key small'>Superior</div><div class='kpi'>{rnflt_metrics['superior']:.2f}</div>", unsafe_allow_html=True)
    k_inf.markdown(f"<div class='metric-key small'>Inferior</div><div class='kpi'>{rnflt_metrics['inferior']:.2f}</div>", unsafe_allow_html=True)
    k_temp.markdown(f"<div class='metric-key small'>Temporal</div><div class='kpi'>{rnflt_metrics['temporal']:.2f}</div>", unsafe_allow_html=True)

    # show gradcam
    if gradcam_img is not None:
        gradcam_placeholder.image(gradcam_img, use_column_width=True, caption="Grad-CAM (B-scan)")
    else:
        gradcam_placeholder.markdown("<div class='muted tiny'>Grad-CAM will appear here once B-scan + model are available.</div>", unsafe_allow_html=True)

    # show RNFLT interactive view
    if rnflt_fig is not None:
        if PLOTLY_AVAILABLE and hasattr(rnflt_fig, "to_image"):
            rnflt_placeholder.plotly_chart(rnflt_fig, use_container_width=True)
        else:
            # matplotlib fallback
            rnflt_placeholder.pyplot(rnflt_fig)

    # download buttons
    dl_col1, dl_col2 = st.columns([1,1])
    with dl_col1:
        if rnflt_fig is not None:
            # RNFLT PNG
            if PLOTLY_AVAILABLE and hasattr(rnflt_fig, "to_image"):
                img_bytes = rnflt_fig.to_image(format="png")
            else:
                img_bytes = fig_to_bytes(rnflt_fig)
            b64 = base64.b64encode(img_bytes).decode()
            st.markdown(f'<a href="data:file/png;base64,{b64}" download="rnflt_visual.png" class="action-btn">Download RNFLT PNG</a>', unsafe_allow_html=True)
    with dl_col2:
        # produce a simple PDF containing RNFLT figure and Grad-CAM (if present)
        if rnflt_fig is not None:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf_buf = io.BytesIO()
            with PdfPages(pdf_buf) as pdf:
                if PLOTLY_AVAILABLE and hasattr(rnflt_fig, "to_image"):
                    # convert plotly to PIL and then save via matplotlib
                    try:
                        img = rnflt_fig.to_image(format="png")
                        pil = Image.open(io.BytesIO(img))
                        fig2, ax2 = plt.subplots(figsize=(6,6))
                        ax2.imshow(pil); ax2.axis('off'); fig2.patch.set_facecolor('#071116')
                        pdf.savefig(fig2, bbox_inches='tight', facecolor=fig2.get_facecolor())
                        plt.close(fig2)
                    except Exception:
                        pass
                else:
                    pdf.savefig(rnflt_fig, bbox_inches='tight', facecolor=rnflt_fig.get_facecolor())
                if gradcam_img is not None:
                    pil = Image.fromarray(gradcam_img)
                    fig3, ax3 = plt.subplots(figsize=(6,6))
                    ax3.imshow(pil); ax3.axis('off'); fig3.patch.set_facecolor('#071116')
                    pdf.savefig(fig3, bbox_inches='tight', facecolor=fig3.get_facecolor())
                    plt.close(fig3)
            pdf_buf.seek(0)
            b64pdf = base64.b64encode(pdf_buf.read()).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64pdf}" download="oculaire_report.pdf" class="action-btn">Download PDF Report</a>', unsafe_allow_html=True)

# ---------------------------
# final note footer
# ---------------------------
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
st.markdown("<div class='tiny muted' style='text-align:center'>This is a demo assistant design — not for clinical use. Validate with ophthalmologists before any clinical application.</div>", unsafe_allow_html=True)
