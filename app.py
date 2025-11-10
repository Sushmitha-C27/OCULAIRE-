# app.py — OCULAIRE: Dream UI (neumorphic / elegant dark)
# Drop-in replacement. Keeps your model/data filenames unchanged.

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import io, os, time, base64
import cv2
import matplotlib.pyplot as plt

# Try plotly for interactive RNFLT. If not available, fallback to matplotlib.
try:
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

# -------------------------
# Page config + matplotlib
# -------------------------
st.set_page_config(page_title="OCULAIRE — Vision Assistant", layout="wide", page_icon="👁️")
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0a0f1a",
    "axes.facecolor": "#0a0f1a",
    "xtick.color": "#f1f6fb",
    "ytick.color": "#f1f6fb",
    "text.color": "#f1f6fb",
    "axes.labelcolor": "#f1f6fb",
})

# -------------------------
# Aesthetic CSS (neumorphic + glow)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
:root{
  --bg1: #020617;
  --bg2: #08112a;
  --card: rgba(255,255,255,0.02);
  --glass: rgba(255,255,255,0.03);
  --muted: #9fb1c9;
  --accentA: #7ee8d1;
  --accentB: #6b5cff;
  --accentGradient: linear-gradient(135deg, #7ee8d1 0%, #6b5cff 100%);
}

/* base */
html,body,.stApp { background: radial-gradient(circle at 10% 10%, var(--bg2) 0%, var(--bg1) 60%); font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; color:#f1f6fb; }

/* top header */
.header {
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:18px 18px; margin-bottom:16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.008));
  border-radius:12px;
  border: 1px solid rgba(255,255,255,0.03);
}

/* brand */
.brand-large { font-weight:800; font-size:24px; letter-spacing:1px; color: #ffffff; }
.subtitle { color: var(--muted); font-size:13px; margin-top:2px; }

/* left rail */
.left-rail { background: transparent; padding-top:18px; min-height:calc(100vh - 80px); }
.icon-rail {
  display:flex; flex-direction:column; gap:10px; align-items:center;
}
.icon-btn {
  width:56px; height:56px; border-radius:12px;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(180deg, rgba(255,255,255,0.016), rgba(255,255,255,0.01));
  border:1px solid rgba(255,255,255,0.03);
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.icon-btn:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(107,92,255,0.12); }

/* glass card */
.glass {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius:14px; padding:16px; border:1px solid rgba(255,255,255,0.03);
  box-shadow: 0 10px 30px rgba(2,6,23,0.6);
}

/* uploader tile */
.uploader {
  border-radius:12px; padding:12px; background: rgba(255,255,255,0.012);
  border: 1px dashed rgba(255,255,255,0.03); text-align:center;
}
.uploader h4 { margin:6px 0 4px 0; color:#eef9f5; }
.uploader p { margin:0; color: var(--muted); font-size:13px; }

/* CTA */
.cta { display:inline-block; padding:10px 20px; border-radius:12px; font-weight:700; color:#021418; background: var(--accentGradient); border: none; cursor:pointer; box-shadow: 0 8px 30px rgba(107,92,255,0.14); transition: transform 0.12s ease; }
.cta:active { transform: translateY(2px); }

/* KPI tiles (neumorphic) */
.kpi-row { display:flex; gap:12px; }
.kpi {
  flex:1; padding:12px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.03); text-align:left;
  transition: transform 0.12s ease;
}
.kpi:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(107,92,255,0.08); }
.kpi .label { color: var(--muted); font-size:12px; }
.kpi .value { font-weight:800; font-size:20px; margin-top:6px; color: #fff; }

/* small action buttons */
.small-btn { background:transparent; border:1px solid rgba(255,255,255,0.04); padding:8px 12px; color:var(--muted); border-radius:8px; }

/* footer */
.footer { text-align:center; color:var(--muted); margin-top:14px; font-size:12px; }

/* responsive tweaks */
@media (max-width:900px) {
  .kpi-row { flex-direction:column; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers: model loaders & preprocess
# -------------------------
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

def preprocess_bscan_image(pil_img, img_size=(224,224)):
    arr = np.array(pil_img.convert("L"))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_res = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat(arr_res[..., None], 3, axis=-1)
    batch = np.expand_dims(rgb, axis=0).astype(np.float32)
    return batch, arr_res

def gradcam(batch, model, last_conv_name=None):
    try:
        if last_conv_name is None:
            for l in reversed(model.layers):
                if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    last_conv_name = l.name; break
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

def read_npz(uploaded_file):
    try:
        b = io.BytesIO(uploaded_file.getvalue())
        arrs = np.load(b, allow_pickle=True)
        key = "volume" if "volume" in arrs else arrs.files[0]
        m = arrs[key]
        if m.ndim == 3: m = m[0,:,:]
        return m
    except Exception:
        return None

def compute_risk_map(rnflt_map, healthy_avg, threshold=-10):
    if rnflt_map.shape != healthy_avg.shape:
        healthy_avg = cv2.resize(healthy_avg, (rnflt_map.shape[1], rnflt_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < threshold, diff, np.nan)
    total = np.isfinite(diff).sum()
    risky = np.isfinite(risk).sum()
    severity = (risky/total)*100 if total>0 else 0
    return diff, risk, severity

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Top header
# -------------------------
col_l, col_r = st.columns([1,4])
with col_l:
    st.markdown("<div class='brand-large'>OCULAIRE</div>", unsafe_allow_html=True)
with col_r:
    st.markdown("<div style='text-align:right'><div class='subtitle'>Automated glaucoma insights — beautiful by design</div></div>", unsafe_allow_html=True)

st.markdown("")  # spacer

# -------------------------
# Layout: left rail, main canvas, right panel
# -------------------------
rail, main, right = st.columns([0.9, 5.2, 1.8], gap="large")

# left rail icons
with rail:
    st.markdown("<div class='left-rail'>", unsafe_allow_html=True)
    st.markdown("<div class='icon-rail'>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>🏠</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>⬆️</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>📈</div>", unsafe_allow_html=True)
    st.markdown("<div class='icon-btn'>⚙️</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# main content
with main:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin:0;color:#f8fbff'>Vision Assistant — OCULAIRE</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted);margin-top:6px'>Upload RNFLT map or B-scan, tune threshold, and press Predict. Results appear instantly with interactive visuals.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    up1, up2 = st.columns([1,1], gap="large")
    with up1:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<h4>B-Scan Image</h4>", unsafe_allow_html=True)
        bscan_file = st.file_uploader("Choose a B-scan image (jpg/png)", type=["jpg","png","jpeg"], key="bscan", label_visibility="collapsed")
        st.markdown("<p>Best if cropped to retina. Grayscale or color accepted.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with up2:
        st.markdown("<div class='uploader'>", unsafe_allow_html=True)
        st.markdown("<h4>RNFLT Map (.npz)</h4>", unsafe_allow_html=True)
        rnflt_file = st.file_uploader("Upload RNFLT map (.npz)", type=["npz"], key="rnflt", label_visibility="collapsed")
        st.markdown("<p>Supports typical OCT RNFLT shape arrays; auto-resize applied.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # controls
    ctl1, ctl2, ctl3 = st.columns([2,1,1], gap="large")
    with ctl1:
        thresh = st.slider("Thinness threshold (µm)", min_value=5, max_value=50, value=10)
    with ctl2:
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        predict = st.button("Predict", key="predict", help="Run model(s) on uploaded inputs", args=None)
        st.markdown("</div>", unsafe_allow_html=True)
    with ctl3:
        if st.button("Reset UI"):
            st.experimental_rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # visualization zone: gradcam (left) + rnflt interactive (right)
    vL, vR = st.columns([1.6, 2.8], gap="large")
    with vL:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<b>Grad-CAM</b>", unsafe_allow_html=True)
        grad_area = st.empty()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with vR:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<b>RNFLT Interactive</b>", unsafe_allow_html=True)
        rnflt_area = st.empty()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close main glass

# right panel: KPI tiles
with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<b style='color:#fff'>Summary</b>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # KPI row 1
    st.markdown("<div class='kpi-row'>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        st.markdown("<div class='kpi'><div class='label'>Status</div><div class='value' id='stat'>—</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='kpi'><div class='label'>Mean RNFLT</div><div class='value' id='mean'>—</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='kpi'><div class='label'>Severity %</div><div class='value' id='sev'>—</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='small-btn' style='display:block;text-align:center'>Download PDF Report</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='tiny' style='color:var(--muted)'>History</div>", unsafe_allow_html=True)
    if "history" not in st.session_state: st.session_state.history = []
    if st.session_state.history:
        for h in st.session_state.history[:6]:
            st.markdown(f"<div style='font-size:12px;color:var(--muted)'>{h['time']} — {h['status']} — mean:{h.get('mean','—'):.1f if h.get('mean') is not None else '—'}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:12px;color:var(--muted)'>No runs yet</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Load models briefly
# -------------------------
with st.spinner("Loading models and artifacts..."):
    bscan_model = load_bscan_model()
    scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_artifacts()
time.sleep(0.15)

# -------------------------
# Predict logic
# -------------------------
def add_history(entry):
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

if predict:
    # state resets for run
    status = "No result"
    severity = None
    risk_mean = None
    mean_val = None
    grad_img = None
    rnflt_fig = None

    prog = st.progress(0)
    step = 0

    # B-scan branch
    if bscan_file is not None and bscan_model is not None:
        step += 1
        prog.progress(int(100 * step / 4))
        try:
            pil = Image.open(bscan_file).convert("L")
            batch, proc = preprocess_bscan_image(pil)
            pred_raw = bscan_model.predict(batch, verbose=0)[0][0]
            label = "Glaucoma" if pred_raw > 0.5 else "Healthy"
            conf = pred_raw if pred_raw > 0.5 else (1-pred_raw)
            status = label
            severity = float(conf * 100)
            # gradcam
            hm = gradcam(batch, bscan_model)
            if hm is not None:
                hm_res = cv2.resize(hm, (224,224))
                hm_img = (hm_res * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
                overlay = (np.stack([proc]*3, axis=-1) * 255).astype(np.uint8)
                overlay = cv2.addWeighted(overlay, 0.65, hm_color, 0.35, 0)
                grad_img = overlay
        except Exception as e:
            st.error(f"B-scan failed: {e}")

    # RNFLT branch
    if rnflt_file is not None and scaler is not None and kmeans is not None and avg_healthy is not None:
        step += 1
        prog.progress(int(100 * step / 4))
        try:
            rnflt_map = read_npz(rnflt_file)
            if rnflt_map is not None:
                vals = rnflt_map.flatten().astype(float)
                mean_val = float(np.nanmean(vals))
                sd = float(np.nanstd(vals))
                mn = float(np.nanmin(vals)); mx = float(np.nanmax(vals))
                X_new = np.array([[mean_val, sd, mn, mx]])
                Xs = scaler.transform(X_new)
                cluster = int(kmeans.predict(Xs)[0])
                label_r = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"
                if status == "No result": status = label_r
                else: status = f"{status} + {label_r}"
                diff, risk_map, severity_r = compute_risk_map(rnflt_map, avg_healthy, threshold=-thresh)
                if severity is None: severity = float(severity_r)
                risk_mean = float(np.nanmean(np.nan_to_num(risk_map))) if np.isfinite(risk_map).any() else 0.0
                # quadrants simple approx:
                h,w = rnflt_map.shape
                sup = float(np.nanmean(rnflt_map[:h//2,:])); inf = float(np.nanmean(rnflt_map[h//2:,:])); temp = float(np.nanmean(rnflt_map[:, :w//3]))
                # interactive RNFLT
                if PLOTLY:
                    fig_px = px.imshow(rnflt_map, color_continuous_scale="Turbo", origin='lower')
                    fig_px.update_layout(coloraxis_colorbar=dict(title="Thickness (µm)", tickfont=dict(color="#f1f6fb")), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    rnflt_fig = fig_px
                else:
                    fig, axes = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
                    axes[0].imshow(rnflt_map, cmap="turbo"); axes[0].axis("off"); axes[0].set_title("Uploaded RNFLT")
                    axes[1].imshow(diff, cmap="bwr", vmin=-30, vmax=30); axes[1].axis("off"); axes[1].set_title("Difference")
                    axes[2].imshow(risk_map, cmap="hot"); axes[2].axis("off"); axes[2].set_title("Risk")
                    fig.patch.set_facecolor("#0a0f1a")
                    rnflt_fig = fig
        except Exception as e:
            st.error(f"RNFLT failed: {e}")

    prog.progress(100)
    time.sleep(0.12)
    prog.empty()

    # display results
    # Grad-CAM
    if grad_img is not None:
        grad_area.image(grad_img, use_column_width=True, caption="Grad-CAM (B-scan)")
    else:
        grad_area.markdown("<div style='color:var(--muted)'>Grad-CAM preview (requires B-scan & model)</div>", unsafe_allow_html=True)

    # RNFLT interactive
    if rnflt_fig is not None:
        if PLOTLY:
            rnflt_area.plotly_chart(rnflt_fig, use_container_width=True)
        else:
            rnflt_area.pyplot(rnflt_fig)
    else:
        rnflt_area.markdown("<div style='color:var(--muted)'>Upload RNFLT (.npz) to view interactive heatmap</div>", unsafe_allow_html=True)

    # update KPI panel (right)
    jsstatus = status or "—"
    kpi_html = f"<script>document.getElementById('stat')?.innerText = '{jsstatus}';</script>"
    st.markdown(kpi_html, unsafe_allow_html=True)
    # simple direct writes for other kpis (since id injection is not reliable across Streamlit versions)
    st.markdown(f"<div style='display:none'></div>", unsafe_allow_html=True)
    # fallback: just write into right panel values via re-render
    right_kpis = right
    with right_kpis:
        # replace placeholders by re-render (simple approach)
        right.markdown("")  # no-op; keep UI responsive

    # For consistent presentation, we print numeric KPIs in the right card using st.write updates:
    # (update by directly writing into the panel)
    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<b style='color:#fff'>Summary</b>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Status</div><div class='value'>{status}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Mean RNFLT</div><div class='value'>{mean_val if mean_val is not None else '—'}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='label'>Severity %</div><div class='value'>{severity:.2f if severity is not None else '—'}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # add to history
    st.session_state.history.insert(0, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "mean": mean_val if mean_val is not None else None})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

    # downloads (RNFLT + Grad-CAM into PDF)
    if rnflt_fig is not None:
        # get PNG bytes
        if PLOTLY:
            try:
                png_bytes = rnflt_fig.to_image(format="png")
            except Exception:
                png_bytes = None
        else:
            png_bytes = fig_to_bytes(rnflt_fig)
        if png_bytes:
            b64 = base64.b64encode(png_bytes).decode()
            st.markdown(f'<a href="data:file/png;base64,{b64}" download="rnflt.png" class="small-btn">Download RNFLT PNG</a>', unsafe_allow_html=True)

    if rnflt_fig is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            if not PLOTLY:
                pdf.savefig(rnflt_fig, bbox_inches='tight', facecolor=rnflt_fig.get_facecolor())
            else:
                # convert plotly to png then to PIL and save
                try:
                    img = rnflt_fig.to_image(format="png")
                    pil = Image.open(io.BytesIO(img))
                    fig2, ax2 = plt.subplots(figsize=(6,6)); ax2.imshow(pil); ax2.axis('off'); fig2.patch.set_facecolor("#0a0f1a")
                    pdf.savefig(fig2, bbox_inches='tight', facecolor=fig2.get_facecolor()); plt.close(fig2)
                except Exception:
                    pass
            if grad_img is not None:
                pilg = Image.fromarray(grad_img)
                fig3, ax3 = plt.subplots(figsize=(6,6)); ax3.imshow(pilg); ax3.axis('off'); fig3.patch.set_facecolor("#0a0f1a")
                pdf.savefig(fig3, bbox_inches='tight', facecolor=fig3.get_facecolor()); plt.close(fig3)
        pdf_buf.seek(0)
        fpdf = base64.b64encode(pdf_buf.read()).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{fpdf}" download="oculaire_report.pdf" class="small-btn">Download PDF Report</a>', unsafe_allow_html=True)

# footer
st.markdown("<div class='footer'>Designed for clarity & beauty — research demo only.</div>", unsafe_allow_html=True)
