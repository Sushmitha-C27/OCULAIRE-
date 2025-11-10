# app.py
# OCULAIRE — Neon Lab v3 (centered technical eye + immediate RNFLT captions)
# Drop-in replacement. Keeps model/data filenames unchanged.

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io, time, base64, os, cv2
from matplotlib.backends.backend_pdf import PdfPages

# Optional interactive RNFLT if plotly installed
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

# -----------------------
# Page config & plotting defaults
# -----------------------
st.set_page_config(page_title="OCULAIRE — Neon Lab v3", layout="wide", page_icon="👁️")
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
# CSS (neon theme + centered technical eye)
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700;900&display=swap');
    :root{ --bg:#020206; --panel:#071026; --neonA:#00f0ff; --neonB:#ff3ac2; --muted:#9fb1c9; }
    html, body, .stApp { background: radial-gradient(circle at 10% 10%, #07102a 0%, #020206 50%); color: #e6fbff; font-family: 'Plus Jakarta Sans', Inter, system-ui; }
    .title-block { display:flex; flex-direction:column; align-items:center; gap:8px; margin:14px 0 18px 0; }
    .tech-eye-svg { width:170px; height:auto; filter: drop-shadow(0 12px 40px rgba(0,240,255,0.08)); cursor:pointer; }
    .brand-title { font-weight:900; font-size:36px; letter-spacing:2px;
        background: linear-gradient(90deg, #00f0ff, #ff3ac2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-sub { color:var(--muted); font-size:13px; }
    .muted { color:var(--muted); }
    .hero { border-radius:12px; padding:16px; background: linear-gradient(180deg, rgba(255,255,255,0.016), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); }
    .uploader { border-radius:12px; padding:12px; text-align:center; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.008)); border: 1px dashed rgba(255,255,255,0.03); }
    .kpi { border-radius:10px; padding:12px; background: linear-gradient(180deg, rgba(0,0,0,0.25), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.03); }
    footer { visibility:hidden; }
    .large-title { font-size:28px; font-weight:900; margin-bottom:8px; }
    .thumb-caption { text-align:center; color:var(--muted); font-size:13px; margin-top:6px; }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------
# Technical eye SVG header (centered)
# -----------------------
header_html = """
<div class="title-block">
  <svg onclick="window.location.search='?page=dashboard'"
       class="tech-eye-svg"
       viewBox="0 0 512 512"
       xmlns="http://www.w3.org/2000/svg"
       role="img"
       aria-label="OCULAIRE tech eye (click to open dashboard)"
       style="max-width:420px; width:100%; height:auto;">
    <defs>
      <radialGradient id="g1" cx="50%" cy="40%">
        <stop offset="0%" stop-color="#66f0ff" stop-opacity="1"/>
        <stop offset="60%" stop-color="#5a7fff" stop-opacity="0.9"/>
        <stop offset="100%" stop-color="#2f0f3a" stop-opacity="0.3"/>
      </radialGradient>
      <linearGradient id="g2" x1="0" x2="1">
        <stop offset="0" stop-color="#00f0ff"/>
        <stop offset="1" stop-color="#ff3ac2"/>
      </linearGradient>
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>

    <!-- outer eyelid ring -->
    <path d="M32 256 C96 112, 416 112, 480 256 C416 400, 96 400, 32 256 Z"
          fill="url(#g1)" stroke="url(#g2)" stroke-width="6" filter="url(#glow)" opacity="0.98"/>
    <!-- iris network -->
    <g transform="translate(256,256)">
      <circle r="80" fill="#03161b" />
      <circle r="66" fill="url(#g2)" opacity="0.12" />
      <!-- tech mesh -->
      <g stroke="#7ee9ff" stroke-opacity="0.12" stroke-width="1">
        <path d="M-50,-10 L50,10 M-30,-50 L30,50 M-60,0 L60,0 M-10,-60 L10,60" />
      </g>
      <!-- pupil -->
      <circle r="30" fill="#001118" stroke="#66f0ff" stroke-opacity="0.22" stroke-width="2"/>
      <circle r="10" cx="-8" cy="-6" fill="#bff8ff" opacity="0.9"/>
    </g>
  </svg>

  <div class="brand-title">OCULAIRE</div>
  <div class="brand-sub">AI-Powered Glaucoma Detection Dashboard</div>
</div>

<script>
  (function() {
    const svg = document.querySelector('.tech-eye-svg');
    if (svg) {
      svg.setAttribute('tabindex', '0');
      svg.setAttribute('role', 'button');
      svg.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          window.location.search = '?page=dashboard';
        }
      });
    }
  })();
</script>
"""
st.markdown(header_html, unsafe_allow_html=True)

# -----------------------
# Helpers: models, preprocess, compute maps
# -----------------------
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
        thick_cluster, thin_cluster = (1,0) if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma) else (0,1)
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

def compute_gradcam(batch, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv = layer.name
                break
        if last_conv is None:
            return None
        grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(batch)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., tf.newaxis]
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
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky / total) * 100 if total > 0 else 0
    return diff, risk, severity

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# -----------------------
# Layout: uploaders + visuals (main area)
# -----------------------
rail_col, canvas_col, right_col = st.columns([0.7, 5, 1.6], gap="large")

with rail_col:
    st.markdown("<div class='muted' style='padding-top:10px; text-align:center'>◼︎</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with canvas_col:
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><div style='font-weight:800;font-size:18px'>Neon Lab Canvas</div><div class='muted tiny'>Futuristic visuals · dynamic panel</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    u1, u2, u3 = st.columns([1.6, 1.6, 1.2], gap="large")
    with u1:
        st.markdown("<div class='uploader'><b>B-scan</b><p class='muted'>jpg / png / jpeg</p></div>", unsafe_allow_html=True)
        bscan_file = st.file_uploader("", type=["jpg","png","jpeg"], key="bscan", label_visibility="collapsed")
    with u2:
        st.markdown("<div class='uploader'><b>RNFLT map (.npz)</b><p class='muted'>standard OCT RNFLT arrays</p></div>", unsafe_allow_html=True)
        rnflt_file = st.file_uploader("", type=["npz"], key="rnflt", label_visibility="collapsed")
    with u3:
        st.markdown("<div class='uploader'><b>Mode</b><p class='muted'>Visualization</p></div>", unsafe_allow_html=True)
        viz_mode = st.selectbox("", ["Heatmap", "Contours"], key="viz_mode", label_visibility="collapsed")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2,1], gap="large")
    with c1:
        threshold = st.slider("Thin-zone threshold (µm)", min_value=5, max_value=50, value=10)
    with c2:
        predict = st.button("PREDICT", key="predict")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    viz_left, viz_right = st.columns([2.6, 2.4], gap="large")
    with viz_left:
        st.markdown("<div class='kpi' style='min-height:420px; display:flex; flex-direction:column; padding:18px;'>", unsafe_allow_html=True)
        st.markdown("<div class='large-title'>RNFLT Visual</div>", unsafe_allow_html=True)
        rnflt_display = st.empty()
        thumb_area = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with viz_right:
        st.markdown("<div class='kpi' style='min-height:420px; display:flex; flex-direction:column; padding:18px;'>", unsafe_allow_html=True)
        st.markdown("<div class='large-title'>Grad-CAM</div>", unsafe_allow_html=True)
        grad_display = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end hero

with right_col:
    st.markdown("<div class='kpi' style='padding:14px'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700'>Overview</div>", unsafe_allow_html=True)
    status_ph = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi' style='padding:10px;margin-bottom:8px'><div class='label'>Severity</div>", unsafe_allow_html=True)
    severity_bar_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi' style='padding:10px;margin-bottom:8px'><div class='label'>Risk (mean)</div>", unsafe_allow_html=True)
    risk_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;margin-top:6px'>RNFLT Quadrants</div>", unsafe_allow_html=True)
    mean_ph = st.empty()
    sup_ph = st.empty()
    inf_ph = st.empty()
    temp_ph = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    hist_ph = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Load models & artifacts (cached)
# -----------------------
with st.spinner("Loading models and artifacts (if available)..."):
    b_model = load_bscan_model()
    scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_artifacts()
time.sleep(0.12)

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Predict (main)
# -----------------------
if predict:
    status_text = "No result"; severity_pct = None; risk_mean = None; rnflt_mean = None
    grad_img = None; rnflt_fig = None; label_r = None

    prog = st.progress(0); step = 0

    # B-scan branch
    if bscan_file is not None and b_model is not None:
        step += 1; prog.progress(int(100 * step / 4))
        try:
            pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan_image(pil)
            pred_raw = b_model.predict(batch, verbose=0)[0][0]
            label = "Glaucoma" if pred_raw > 0.5 else "Healthy"
            confidence = pred_raw if pred_raw > 0.5 else (1 - pred_raw)
            status_text = label
            severity_pct = float(confidence * 100)
            heat = compute_gradcam(batch, b_model)
            if heat is not None:
                heat_r = cv2.resize(heat, (224,224))
                hm = (heat_r * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.65, hm_color, 0.35, 0)
                grad_img = overlay
        except Exception as e:
            st.error(f"B-scan processing error: {e}")

    # RNFLT branch
    if rnflt_file is not None and scaler is not None and kmeans is not None and avg_healthy is not None:
        step += 1; prog.progress(int(100 * step / 4))
        try:
            rnflt_map = read_npz(rnflt_file)
            if rnflt_map is not None:
                vals = rnflt_map.flatten().astype(float)
                rnflt_mean = float(np.nanmean(vals)); rnflt_std = float(np.nanstd(vals))
                mn = float(np.nanmin(vals)); mx = float(np.nanmax(vals))
                X_new = np.array([[rnflt_mean, rnflt_std, mn, mx]])
                Xs = scaler.transform(X_new)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                if status_text == "No result": status_text = label_r
                else: status_text = f"{status_text} + {label_r}"

                diff, risk_map, severity_calc = compute_risk_map(rnflt_map, avg_healthy, threshold=-threshold)
                if severity_pct is None: severity_pct = float(severity_calc)
                risk_mean = float(np.nanmean(np.nan_to_num(risk_map))) if np.isfinite(risk_map).any() else 0.0

                h, w = rnflt_map.shape
                sup = float(np.nanmean(rnflt_map[:h//2, :])) if h>1 else np.nan
                inf = float(np.nanmean(rnflt_map[h//2:, :])) if h>1 else np.nan
                temp = float(np.nanmean(rnflt_map[:, :w//3])) if w>2 else np.nan

                plot_title_base = f"Uploaded RNFLT Map — {label_r}" if label_r is not None else "Uploaded RNFLT Map"

                if PLOTLY:
                    try:
                        fig0 = px.imshow(rnflt_map, color_continuous_scale="Turbo", origin='lower')
                        fig0.update_layout(title={"text": plot_title_base, "x":0.5}, margin=dict(l=10,r=10,t=34,b=10))
                        fig0.update_traces(colorbar=dict(title="Thickness (µm)"))

                        fig1 = px.imshow(diff, color_continuous_scale="RdBu", origin='lower')
                        fig1.update_layout(title={"text": "Difference Map (vs. Healthy)", "x":0.5}, margin=dict(l=10,r=10,t=34,b=10))
                        fig1.update_traces(colorbar=dict(title="Δ Thickness (µm)"))

                        fig2 = px.imshow(risk_map, color_continuous_scale="hot", origin='lower')
                        fig2.update_layout(title={"text": "Risk Map (Thinner Zones)", "x":0.5}, margin=dict(l=10,r=10,t=34,b=10))
                        fig2.update_traces(colorbar=dict(title="Δ Thickness (µm)"))

                        rnflt_fig = (fig0, fig1, fig2)
                    except Exception:
                        rnflt_fig = None
                else:
                    fig, axes = plt.subplots(1, 3, figsize=(15,6), constrained_layout=True)
                    im0 = axes[0].imshow(rnflt_map, cmap='turbo')
                    axes[0].set_title(plot_title_base, fontsize=18, color='white')
                    axes[0].axis('off')
                    c0 = plt.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.02); c0.set_label("Thickness (µm)")

                    im1 = axes[1].imshow(diff, cmap='bwr', vmin=-30, vmax=30)
                    axes[1].set_title("Difference Map (vs. Healthy)", fontsize=18, color='white'); axes[1].axis('off')
                    c1 = plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02); c1.set_label("Δ Thickness (µm)")

                    im2 = axes[2].imshow(risk_map, cmap='hot')
                    axes[2].set_title("Risk Map (Thinner Zones)", fontsize=18, color='white'); axes[2].axis('off')
                    c2 = plt.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.02); c2.set_label("Δ Thickness (µm)")

                    fig.patch.set_facecolor("#04050a")
                    for ax in axes: ax.set_facecolor("#04050a")
                    rnflt_fig = fig
        except Exception as e:
            st.error(f"RNFLT processing error: {e}")

    prog.progress(100); time.sleep(0.08); prog.empty()

    # Update right-hand placeholders
    status_ph.markdown(f"<div style='font-weight:800;font-size:14px'>{status_text}</div>", unsafe_allow_html=True)
    if severity_pct is not None:
        severity_bar_ph.markdown(f"<div style='font-weight:800;color:#ffffff'>{severity_pct:.1f}%</div>", unsafe_allow_html=True)
        st.progress(min(max(severity_pct/100.0,0),1))
    else:
        severity_bar_ph.markdown("<div class='muted'>—</div>", unsafe_allow_html=True)
    risk_ph.markdown(f"<div style='font-weight:700'>{risk_mean:.2f}</div>" if risk_mean is not None else "<div class='muted'>—</div>", unsafe_allow_html=True)
    mean_ph.markdown(f"<div class='muted'>Mean RNFLT</div><div style='font-weight:800'>{rnflt_mean:.2f} µm</div>" if rnflt_mean is not None else "<div class='muted'>Mean RNFLT: —</div>", unsafe_allow_html=True)

    sup_ph.markdown(f"<div class='muted'>Superior</div><div style='font-weight:700'>{sup:.2f}</div>" if 'sup' in locals() else "<div class='muted'>Superior —</div>", unsafe_allow_html=True)
    inf_ph.markdown(f"<div class='muted'>Inferior</div><div style='font-weight:700'>{inf:.2f}</div>" if 'inf' in locals() else "<div class='muted'>Inferior —</div>", unsafe_allow_html=True)
    temp_ph.markdown(f"<div class='muted'>Temporal</div><div style='font-weight:700'>{temp:.2f}</div>" if 'temp' in locals() else "<div class='muted'>Temporal —</div>", unsafe_allow_html=True)

    # Render RNFLT visuals immediately under the main RNFLT container
    if rnflt_fig is not None:
        if PLOTLY and isinstance(rnflt_fig, tuple):
            main_fig, fig_diff, fig_risk = rnflt_fig
            rnflt_display.plotly_chart(main_fig.update_layout(height=520), use_container_width=True)
            cols = thumb_area.columns([1,1,1], gap="small")
            with cols[0]:
                st.plotly_chart(main_fig.update_layout(height=260), use_container_width=True)
                st.markdown("<div class='thumb-caption'>Uploaded RNFLT Map — {}</div>".format(label_r), unsafe_allow_html=True)
            with cols[1]:
                st.plotly_chart(fig_diff.update_layout(height=260), use_container_width=True)
                st.markdown("<div class='thumb-caption'>Difference Map (vs. Healthy)</div>", unsafe_allow_html=True)
            with cols[2]:
                st.plotly_chart(fig_risk.update_layout(height=260), use_container_width=True)
                st.markdown("<div class='thumb-caption'>Risk Map (Thinner Zones)</div>", unsafe_allow_html=True)
        else:
            rnflt_display.pyplot(rnflt_fig)
            try:
                c0, c1, c2 = thumb_area.columns([1,1,1], gap="small")
                with c0:
                    figa, axa = plt.subplots(figsize=(3,3)); axa.imshow(rnflt_map, cmap='turbo'); axa.axis('off'); plt.tight_layout()
                    st.pyplot(figa); plt.close(figa)
                    st.markdown(f"<div class='thumb-caption'>Uploaded RNFLT Map — {label_r}</div>", unsafe_allow_html=True)
                with c1:
                    figb, axb = plt.subplots(figsize=(3,3)); axb.imshow(diff, cmap='bwr', vmin=-30, vmax=30); axb.axis('off'); plt.tight_layout()
                    st.pyplot(figb); plt.close(figb)
                    st.markdown("<div class='thumb-caption'>Difference Map (vs. Healthy)</div>", unsafe_allow_html=True)
                with c2:
                    figc, axc = plt.subplots(figsize=(3,3)); axc.imshow(risk_map, cmap='hot'); axc.axis('off'); plt.tight_layout()
                    st.pyplot(figc); plt.close(figc)
                    st.markdown("<div class='thumb-caption'>Risk Map (Thinner Zones)</div>", unsafe_allow_html=True)
            except Exception:
                pass
    else:
        rnflt_display.markdown("<div class='muted'>No RNFLT visualization (upload .npz)</div>", unsafe_allow_html=True)

    # Grad-CAM
    if grad_img is not None:
        grad_display.image(grad_img, use_column_width=True, caption="Grad-CAM (B-scan)")
    else:
        grad_display.markdown("<div class='muted'>No Grad-CAM (need B-scan + model)</div>", unsafe_allow_html=True)

    # history
    hist_entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status_text, "mean": rnflt_mean}
    st.session_state.history.insert(0, hist_entry)
    if len(st.session_state.history) > 20: st.session_state.history = st.session_state.history[:20]
    lines = []
    for e in st.session_state.history[:6]:
        t = e.get("time", ""); s = e.get("status", "—"); m = e.get("mean", None)
        mstr = f"{m:.1f}" if (m is not None and not np.isnan(m)) else "—"
        lines.append(f"{t} — {s} — mean:{mstr}")
    hist_ph.markdown("<br>".join([f"<div style='color:var(--muted);font-size:12px'>{l}</div>" for l in lines]), unsafe_allow_html=True)

    # downloads (png/pdf)
    if rnflt_fig is not None:
        try:
            if PLOTLY and isinstance(rnflt_fig, tuple):
                png_bytes = rnflt_fig[0].to_image(format="png")
            else:
                png_bytes = fig_to_bytes(rnflt_fig)
            if png_bytes:
                b64 = base64.b64encode(png_bytes).decode()
                st.markdown(f'<a href="data:file/png;base64,{b64}" download="rnflt_visual.png" class="muted">Download RNFLT PNG</a>', unsafe_allow_html=True)
        except Exception:
            pass

        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            try:
                if not PLOTLY:
                    pdf.savefig(rnflt_fig, bbox_inches='tight', facecolor=rnflt_fig.get_facecolor())
                else:
                    img = rnflt_fig[0].to_image(format="png")
                    pil_img = Image.open(io.BytesIO(img))
                    fig2, ax2 = plt.subplots(figsize=(6,6)); ax2.imshow(pil_img); ax2.axis('off'); pdf.savefig(fig2, bbox_inches='tight'); plt.close(fig2)
            except Exception:
                pass
        pdf_buf.seek(0)
        b64pdf = base64.b64encode(pdf_buf.read()).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64pdf}" download="oculaire_report.pdf" class="muted">Download PDF Report</a>', unsafe_allow_html=True)

# -----------------------
# If not predicting, show history summary
# -----------------------
if not predict:
    lines = []
    for e in st.session_state.history[:6]:
        t = e.get("time", ""); s = e.get("status", "—"); m = e.get("mean", None)
        mstr = f"{m:.1f}" if (m is not None and not np.isnan(m)) else "—"
        lines.append(f"{t} — {s} — mean:{mstr}")
    if lines:
        hist_ph.markdown("<br>".join([f"<div style='color:var(--muted);font-size:12px'>{l}</div>" for l in lines]), unsafe_allow_html=True)
    else:
        hist_ph.markdown("<div class='muted'>No runs yet</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted)'>OCULAIRE — Neon Lab v3. Research demo only; not for clinical use.</div>", unsafe_allow_html=True)
