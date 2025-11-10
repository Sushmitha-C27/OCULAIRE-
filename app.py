# app.py
# OCULAIRE — FUTURISTIC NEON LAB (Surprise UI)
# Drop-in replacement. Keeps your model/data filenames unchanged.

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2, io, time, base64, os
from matplotlib.backends.backend_pdf import PdfPages

# optional: interactive RNFLT if plotly is installed
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

# -----------------------
# Page config & plotting
# -----------------------
st.set_page_config(page_title="OCULAIRE — Neon Lab", layout="wide", page_icon="🧪")
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#04050a",
    "axes.facecolor": "#04050a",
    "text.color": "#e6fbff",
    "xtick.color": "#e6fbff",
    "ytick.color": "#e6fbff",
    "axes.labelcolor": "#e6fbff",
    "axes.titleweight": "bold"
})

# -----------------------
# Neon CSS (unique style)
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700;900&display=swap');
    :root{
      --bg:#020206;
      --panel:#071026;
      --neonA:#00f0ff;
      --neonB:#ff3ac2;
      --muted:#9fb1c9;
      --glass: rgba(255,255,255,0.02);
    }
    html, body, .stApp { background: radial-gradient(circle at 10% 10%, #07102a 0%, #020206 50%); color: #e6fbff; font-family: 'Plus Jakarta Sans', Inter, system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial; }
    /* top */
    .header {
      display:flex; align-items:center; justify-content:space-between; gap:12px; padding:14px 18px; margin-bottom:14px;
      border-radius:12px; background: linear-gradient(90deg, rgba(255,255,255,0.012), rgba(255,255,255,0.008));
      border: 1px solid rgba(255,255,255,0.03);
    }
    .brand { font-weight:900; font-size:22px; letter-spacing:1px; color: #ffffff; }
    .tagline { color:var(--muted); font-size:13px; }
    /* control rail */
    .rail { background:transparent; display:flex; flex-direction:column; gap:12px; padding-top:10px; }
    .rail .btn {
        width:56px; height:56px; border-radius:12px; display:flex; align-items:center; justify-content:center;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border:1px solid rgba(255,255,255,0.03);
        color:#e6fbff; font-weight:700;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .rail .btn:hover { transform: translateY(-6px); box-shadow: 0 12px 28px rgba(0,240,255,0.08); }
    /* main hero */
    .hero {
      border-radius:12px; padding:16px; background: linear-gradient(180deg, rgba(255,255,255,0.016), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.03);
    }
    .uploader {
      border-radius:12px; padding:12px; text-align:center; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.008));
      border: 1px dashed rgba(255,255,255,0.03);
    }
    .neon-btn {
      background: linear-gradient(90deg, var(--neonA), var(--neonB)); border:none; padding:10px 18px; border-radius:12px; color:#031116; font-weight:800;
      box-shadow: 0 8px 36px rgba(0,240,255,0.10), 0 6px 18px rgba(255,58,194,0.06); transition: transform 0.09s ease;
    }
    .neon-btn:active { transform: translateY(2px); }
    .kpi {
      border-radius:10px; padding:12px; background: linear-gradient(180deg, rgba(0,0,0,0.25), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.03);
      box-shadow: inset 0 -6px 18px rgba(0,0,0,0.6);
    }
    .kpi .label { color:var(--muted); font-size:12px; }
    .kpi .value { font-weight:800; font-size:20px; color:#fff; }
    .chip { display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.02); color:var(--muted); font-size:12px; }
    .muted { color:var(--muted); }
    .footer { text-align:center; color:var(--muted); margin-top:12px; font-size:12px; }
    /* small */
    footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------
# Helper functions (models + preprocess)
# -----------------------
@st.cache_resource
def load_bscan_model():
    try:
        m = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return m
    except Exception:
        return None

@st.cache_resource
def load_rnflt_artifacts():
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        thick_cluster, thin_cluster = (1, 0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0, 1)
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception:
        return None, None, None, None, None, None

def preprocess_bscan_image(image_pil, img_size=(224,224)):
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return batch, arr_res

def compute_gradcam(batch, model, last_conv_layer_name=None):
    try:
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    last_conv_layer_name = layer.name
                    break
        grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(batch)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
        return heatmap.numpy()
    except Exception:
        return None

def read_npz(uploaded_file):
    try:
        buf = io.BytesIO(uploaded_file.getvalue())
        arrs = np.load(buf, allow_pickle=True)
        key = "volume" if "volume" in arrs else arrs.files[0]
        m = arrs[key]
        if m.ndim == 3:
            m = m[0,:,:]
        return m
    except Exception:
        return None

def compute_risk_map(rnflt_map, avg_healthy, threshold=-10):
    if rnflt_map.shape != avg_healthy.shape:
        avg_healthy = cv2.resize(avg_healthy, (rnflt_map.shape[1], rnflt_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = rnflt_map - avg_healthy
    risk = np.where(diff < threshold, diff, np.nan)
    total_pixels = np.isfinite(diff).sum()
    risky_pixels = np.isfinite(risk).sum()
    severity = (risky_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    return diff, risk, severity

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# -----------------------
# App layout (unique)
# -----------------------
# top header
top_l, top_r = st.columns([1, 3])
with top_l:
    st.markdown("<div class='header'><div style='display:flex;flex-direction:column;'><div class='brand'>OCULAIRE</div><div class='tagline'>Neon Lab — assistant's surprise</div></div></div>", unsafe_allow_html=True)
with top_r:
    st.markdown("<div style='text-align:right'><span class='chip'>Experimental UI</span></div>", unsafe_allow_html=True)

st.markdown("")  # spacer

# main columns: rail | canvas | right metrics
rail_col, canvas_col, right_col = st.columns([0.8, 5, 1.6], gap="large")

# control rail (small vertical)
with rail_col:
    st.markdown("<div class='rail'>", unsafe_allow_html=True)
    st.markdown("<div title='Home' class='btn'>🏠</div>", unsafe_allow_html=True)
    st.markdown("<div title='Upload' class='btn'>⬆️</div>", unsafe_allow_html=True)
    st.markdown("<div title='Run' class='btn'>▶️</div>", unsafe_allow_html=True)
    st.markdown("<div title='History' class='btn'>📜</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# central cinematic canvas
with canvas_col:
    st.markdown("<div class='hero'>", unsafe_allow_html=True)

    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:18px;font-weight:800'>Neon Lab Canvas</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted' style='font-size:12px'>Futuristic visuals · one-click analytics</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    up1, up2, up3 = st.columns([1.6, 1.6, 1.4], gap="large")
    with up1:
        st.markdown("<div class='uploader'><b>B-scan</b><p class='muted'>jpg / png / jpeg (grayscale OK)</p></div>", unsafe_allow_html=True)
        bscan_file = st.file_uploader("", type=["jpg","png","jpeg"], key="f_bscan", label_visibility="collapsed")
    with up2:
        st.markdown("<div class='uploader'><b>RNFLT map (.npz)</b><p class='muted'>standard OCT RNFLT arrays</p></div>", unsafe_allow_html=True)
        rnflt_file = st.file_uploader("", type=["npz"], key="f_rnflt", label_visibility="collapsed")
    with up3:
        st.markdown("<div class='uploader'><b>Mode</b><p class='muted'>Visualization</p></div>", unsafe_allow_html=True)
        viz_mode = st.selectbox("", ["Heatmap", "Contours"], key="viz_mode", label_visibility="collapsed")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # THRESHOLD + PREDICT CTA
    ctl_left, ctl_right = st.columns([2,1], gap="large")
    with ctl_left:
        threshold = st.slider("Thin-zone threshold (µm)", min_value=5, max_value=50, value=10)
    with ctl_right:
        st.markdown("<div style='display:flex;align-items:center;justify-content:flex-end'>", unsafe_allow_html=True)
        predict = st.button("PREDICT", key="btn_predict", help="Run analysis with current inputs", args=None)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Large hero visuals: top row big RNFLT / Grad-CAM side-by-side
    viz_left, viz_right = st.columns([2.6, 2.4], gap="large")
    with viz_left:
        st.markdown("<div class='kpi' style='height:420px;display:flex;flex-direction:column;gap:8px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700'>RNFLT Visual</div>", unsafe_allow_html=True)
        rnflt_display = st.empty()
        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with viz_right:
        st.markdown("<div class='kpi' style='height:420px;display:flex;flex-direction:column;gap:8px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700'>Grad-CAM</div>", unsafe_allow_html=True)
        grad_display = st.empty()
        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close hero

# right side KPIs
with right_col:
    st.markdown("<div class='kpi' style='padding:14px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700'>Overview</div>", unsafe_allow_html=True)
    status_box = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi' style='padding:10px;margin-bottom:8px'><div class='label'>Severity</div><div class='value' id='sev_val'>—</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi' style='padding:10px;margin-bottom:8px'><div class='label'>Risk (mean)</div><div class='value' id='risk_val'>—</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if "history" not in st.session_state:
        st.session_state.history = []
    st.markdown("<div style='font-size:12px;color:var(--muted)'>Recent runs</div>", unsafe_allow_html=True)
    hist_space = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Load models (background)
# -----------------------
with st.spinner("Loading model artifacts (if present)..."):
    b_model = load_bscan_model()
    scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_artifacts()
time.sleep(0.12)

# -----------------------
# Predict logic
# -----------------------
def push_history(entry):
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

def render_history():
    if st.session_state.history:
        lines = []
        for e in st.session_state.history[:6]:
            t = e.get("time", "")
            s = e.get("status", "—")
            m = e.get("mean", None)
            mstr = f"{m:.1f}" if (m is not None and not np.isnan(m)) else "—"
            lines.append(f"{t} — {s} — mean:{mstr}")
        hist_space.markdown("<br>".join([f"<div style='color:var(--muted);font-size:12px'>{l}</div>" for l in lines]), unsafe_allow_html=True)
    else:
        hist_space.markdown("<div style='color:var(--muted);font-size:12px'>No runs yet</div>", unsafe_allow_html=True)

render_history()

if predict:
    # placeholders / initial values
    status_text = "No result"
    severity_pct = None
    risk_mean = None
    rnflt_mean = None
    gradcam_img = None
    rnflt_map = None
    rnflt_fig = None

    with st.spinner("Running Neon analysis..."):
        prog = st.progress(0)
        step = 0

        # B-scan model branch
        if bscan_file is not None and b_model is not None:
            step += 1
            prog.progress(int(100 * step / 4))
            try:
                pil = Image.open(bscan_file).convert("L")
                batch, proc = preprocess_bscan_image(pil)
                pred_raw = b_model.predict(batch, verbose=0)[0][0]
                label = "Glaucoma" if pred_raw > 0.5 else "Healthy"
                confidence = pred_raw if pred_raw > 0.5 else (1 - pred_raw)
                status_text = label
                severity_pct = float(confidence * 100)
                # grad-cam
                heat = compute_gradcam(batch, b_model)
                if heat is not None:
                    heat_r = cv2.resize(heat, (224,224))
                    hm = (heat_r * 255).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                    overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                    overlay = cv2.addWeighted(overlay, 0.65, hm_color, 0.35, 0)
                    gradcam_img = overlay
            except Exception as e:
                st.error(f"B-scan processing error: {e}")

        # RNFLT branch
        if rnflt_file is not None and scaler is not None and kmeans is not None and avg_healthy is not None:
            step += 1
            prog.progress(int(100 * step / 4))
            try:
                rnflt_map = read_npz(rnflt_file)
                if rnflt_map is not None:
                    vals = rnflt_map.flatten().astype(float)
                    rnflt_mean = float(np.nanmean(vals))
                    rnflt_std = float(np.nanstd(vals))
                    rnflt_min = float(np.nanmin(vals)); rnflt_max = float(np.nanmax(vals))
                    X_new = np.array([[rnflt_mean, rnflt_std, rnflt_min, rnflt_max]])
                    Xs = scaler.transform(X_new)
                    cluster = int(kmeans.predict(Xs)[0])
                    label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                    if status_text == "No result":
                        status_text = label_r
                    else:
                        status_text = f"{status_text} + {label_r}"
                    diff, risk_map, severity = compute_risk_map(rnflt_map, avg_healthy, threshold=-threshold)
                    if severity_pct is None:
                        severity_pct = float(severity)
                    risk_mean = float(np.nanmean(np.nan_to_num(risk_map))) if np.isfinite(risk_map).any() else 0.0
                    # build RNFLT visual (interactive if plotly)
                    if PLOTLY:
                        try:
                            fig_px = px.imshow(rnflt_map, color_continuous_scale="Turbo", origin='lower')
                            fig_px.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_colorbar=dict(title="Thickness (µm)"))
                            rnflt_fig = fig_px
                        except Exception:
                            rnflt_fig = None
                    else:
                        fig, axes = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
                        axes[0].imshow(rnflt_map, cmap="turbo"); axes[0].axis("off"); axes[0].set_title("Uploaded")
                        axes[1].imshow(diff, cmap="bwr", vmin=-30, vmax=30); axes[1].axis("off"); axes[1].set_title("Diff")
                        axes[2].imshow(risk_map, cmap="hot"); axes[2].axis("off"); axes[2].set_title("Risk")
                        fig.patch.set_facecolor("#04050a")
                        rnflt_fig = fig
            except Exception as e:
                st.error(f"RNFLT processing error: {e}")

        prog.progress(100)
        time.sleep(0.12)
        prog.empty()

    # display outputs
    if gradcam_img is not None:
        grad_display.image(gradcam_img, use_column_width=True, caption="Grad-CAM (B-scan)")
    else:
        grad_display.markdown("<div class='muted'>No Grad-CAM (provide B-scan & model)</div>", unsafe_allow_html=True)

    if rnflt_fig is not None:
        if PLOTLY:
            rnflt_display.plotly_chart(rnflt_fig, use_container_width=True)
        else:
            rnflt_display.pyplot(rnflt_fig)
    else:
        rnflt_display.markdown("<div class='muted'>Upload RNFLT (.npz) to visualize map</div>", unsafe_allow_html=True)

    # update right-side KPIs
    status_box.markdown(f"<div style='font-weight:800;font-size:14px'>{status_text}</div>", unsafe_allow_html=True)
    # numeric KPI values
    sev_val = f"{severity_pct:.2f}%" if severity_pct is not None else "—"
    risk_val = f"{risk_mean:.2f}" if risk_mean is not None else "—"
    # overwrite KPI tiles by writing HTML in right column (simple approach)
    st.experimental_rerun()  # quick re-render to propagate updated right-hand values

# fallback render of history and KPIs (if not using predict branch re-run)
render_history()
# note: st.experimental_rerun() above will re-render the page and populate KPIs if predict succeeded
# If you prefer not to force rerun, we could update specific placeholders instead of rerunning.

st.markdown("<div class='footer'>OCULAIRE — Neon Lab • Research demo (not clinical)</div>", unsafe_allow_html=True)
