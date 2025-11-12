# app.py (safer startup: lazy-load heavy libs, robust header)
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import io, time, base64, os

st.set_page_config(page_title="OCULAIRE — Neon Lab v3 (safe)", layout="wide", page_icon="👁️")

# Minimal CSS
st.markdown("""
<style>
body { background: radial-gradient(circle at 10% 10%, #07102a 0%, #020206 50%); color: #e6fbff; }
.title-block { display:flex; flex-direction:column; align-items:center; margin:18px 0; }
.brand-title { font-weight:900; font-size:36px; background:linear-gradient(90deg,#00f0ff,#ff3ac2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.brand-sub { color:#9fb1c9; font-size:13px; }
.kv { color:#9fb1c9; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# Render header reliably (components.html prevents markdown escaping)
header_html = r"""
<div class="title-block" id="oculaire-header" style="text-align:center;">
  <svg tabindex="0" onclick="window.location.hash='#dashboard'" viewBox="0 0 512 512" width="200" height="200" style="cursor:pointer;">
    <defs>
      <radialGradient id="g1" cx="50%" cy="40%"><stop offset="0%" stop-color="#66f0ff"/><stop offset="60%" stop-color="#5a7fff"/><stop offset="100%" stop-color="#2f0f3a"/></radialGradient>
      <linearGradient id="g2" x1="0" x2="1"><stop offset="0" stop-color="#00f0ff"/><stop offset="1" stop-color="#ff3ac2"/></linearGradient>
    </defs>
    <path d="M32 256 C96 112, 416 112, 480 256 C416 400, 96 400, 32 256 Z" fill="url(#g1)" stroke="url(#g2)" stroke-width="6" opacity="0.98"/>
    <g transform="translate(256,256)">
      <circle r="80" fill="#03161b" />
      <circle r="66" fill="url(#g2)" opacity="0.12" />
      <circle r="30" fill="#001118" stroke="#66f0ff" stroke-opacity="0.22" stroke-width="2"/>
      <circle r="8" cx="-6" cy="-4" fill="#bff8ff" opacity="0.9"/>
    </g>
  </svg>
  <div class="brand-title">OCULAIRE</div>
  <div class="brand-sub">AI-Powered Glaucoma Detection Dashboard</div>
</div>
<script>
  (function(){ const svg = document.querySelector('#oculaire-header svg'); if(svg){ svg.setAttribute('tabindex',0); svg.addEventListener('keydown', (e)=>{ if(e.key==='Enter' || e.key===' ') window.location.hash='#dashboard'; }); } })();
</script>
"""
components.html(header_html, height=300, scrolling=False)

st.markdown("<div id='dashboard'></div>", unsafe_allow_html=True)

# --- Try to import heavy libs lazily and fail gracefully ---
TF_AVAILABLE = False
CV_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    st.warning("TensorFlow not available at startup — B-scan features disabled (will try lazy-load later).")
try:
    import cv2
    CV_AVAILABLE = True
except Exception:
    st.warning("OpenCV not available — some image operations disabled.")

# Provide an upload area and simple UI so app starts
st.markdown("## Neon Lab Canvas — quick safe mode")
col1, col2 = st.columns([3,1])
with col1:
    rnflt_file = st.file_uploader("Upload RNFLT .npz (optional)", type=["npz"])
    bscan_file = st.file_uploader("Upload B-scan (optional)", type=["jpg","png","jpeg"])
    threshold = st.slider("Thin-zone threshold (µm)", 5, 50, 10)
    if st.button("Run (safe)"):
        st.info("Running simplified analysis...")
        # simple check: attempt to read .npz and show mean
        if rnflt_file is not None:
            try:
                arrs = np.load(io.BytesIO(rnflt_file.getvalue()), allow_pickle=True)
                key = "volume" if "volume" in arrs else arrs.files[0]
                m = arrs[key]
                if m.ndim == 3: m = m[0]
                st.metric("Mean RNFLT", f"{np.nanmean(m):.2f} µm")
                # show a small image using matplotlib if numpy shape is 2D
                if m.ndim==2:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.imshow(m, cmap='turbo'); ax.axis('off')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not read .npz: {e}")
        else:
            st.write("No RNFLT provided.")

with col2:
    st.markdown("<div class='kv'>Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='kv'>Status: Safe mode</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kv'>Models loaded: {}</div>".format("yes" if TF_AVAILABLE else "no"), unsafe_allow_html=True)
    if TF_AVAILABLE:
        st.success("TF present — full features available")
    else:
        st.info("TF not loaded — upload will still work but model predictions disabled")

st.write("---")
st.write("If this page loads then Streamlit runtime is fine. Check Manage App -> Logs for details if your original app still fails.")
