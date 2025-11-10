import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io

# --- App Configuration & Initial Load ---
st.set_page_config(
    page_title="OCULAIRE Glaucoma Detection", 
    layout="wide", 
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# --- App Title and Sidebar ---
st.title("👁️ OCULAIRE: Glaucoma Detection Dashboard")

with st.sidebar:
    st.header("Analysis Controls")
    analysis_type = st.radio(
        "Select Analysis Type",
        ("🩺 RNFLT Map Analysis (.npz)", "👁️ B-Scan Slice Analysis (Image)"),
    )
    st.markdown("---")
    st.info("Upload your files and select the analysis type here.")


# --- Model Loading (Cached) ---

@st.cache_resource
def load_bscan_model():
    """Loads the Supervised B-Scan CNN model."""
    try:
        model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading B-Scan CNN model: {e}")
        return None

def load_rnflt_models_safe():
    """Loads Unsupervised RNFLT artifacts safely."""
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        
        # Determine thin/thick cluster based on mean thickness
        if np.nanmean(avg_healthy) > np.nanmean(avg_glaucoma):
            thick_cluster = 1
            thin_cluster = 0
        else:
            thick_cluster = 0
            thin_cluster = 1
            
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception as e:
        st.error(f"Error loading RNFLT artifacts: {e}")
        st.warning("Ensure all RNFLT model/data files are in the same directory.")
        return None, None, None, None, None, None

# ==============================================================================
# 1. RNFLT (PHASE D) HELPER FUNCTIONS
# ==============================================================================
# ... (Keep process_uploaded_npz and compute_risk_map functions the same) ...

def process_uploaded_npz(uploaded_file):
    """Loads NPZ file from Streamlit's uploader and extracts metrics."""
    try:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        npz = np.load(file_bytes, allow_pickle=True)
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
    """Generates difference and risk maps."""
    if rnflt_map.shape != healthy_avg.shape:
         healthy_avg = cv2.resize(healthy_avg, (rnflt_map.shape[1], rnflt_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < threshold, diff, np.nan)
    
    total_pixels = np.isfinite(diff).sum()
    risky_pixels = np.isfinite(risk).sum()
    severity = (risky_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    return diff, risk, severity

# ==============================================================================
# 2. B-SCAN (PHASE S) HELPER FUNCTIONS
# ==============================================================================
# ... (Keep preprocess_bscan_image and make_gradcam_heatmap functions the same) ...

def preprocess_bscan_image(image_pil, img_size=(224, 224)):
    """Preprocesses a PIL Image for the B-Scan model."""
    arr = np.array(image_pil.convert('L'))
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr_resized = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    arr_rgb = np.repeat(arr_resized[..., None], 3, axis=-1)
    img_batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return img_batch, arr_resized

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Generates Grad-CAM heatmap."""
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        st.error("Could not find a Conv2D layer for Grad-CAM.")
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
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


# ==============================================================================
# --- MAIN APPLICATION LOGIC ---
# ==============================================================================

if "RNFLT" in analysis_type:
    # --- RNFLT ANALYSIS (PHASE D) ---
    st.header("RNFLT Map Analysis (Unsupervised)")
    
    # Use a container for the upload widget to keep it clean
    with st.container(border=True):
        scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models_safe()
        
        if scaler is None:
            st.stop()
            
        uploaded_file = st.file_uploader("Upload an RNFLT .npz file", type=["npz"])
    
    if uploaded_file is not None:
        rnflt_map, metrics = process_uploaded_npz(uploaded_file)
        
        if rnflt_map is not None:
            # 1. Predict Cluster
            X_new = np.array([[metrics["mean"], metrics["std"], metrics["min"], metrics["max"]]])
            X_scaled = scaler.transform(X_new)
            cluster = int(kmeans.predict(X_scaled)[0])
            label = "Glaucoma-like" if cluster == thin_cluster else "Healthy-like"

            # 2. Compute Risk
            diff, risk, severity = compute_risk_map(rnflt_map, avg_healthy)

            # 3. Display Results in clean metric boxes
            st.markdown("### Diagnosis Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            # Use color for emphasis
            status_emoji = "🚨" if label == "Glaucoma-like" else "✅"
            
            col1.metric("Predicted Status", f"{status_emoji} **{label}**")
            col2.metric("Mean RNFLT", f"{metrics['mean']:.2f} µm")
            col3.metric("Severity Score", f"{severity:.2f}%", help="Percentage of area significantly thinner than healthy average.")
            col4.metric("K-Means Cluster", cluster)
            
            st.markdown("---")
            
            # Use an expander for the large plots
            with st.expander("🔬 Detailed RNFLT Visualization", expanded=True):
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Plot 1: Uploaded Map
                im1 = axes[0].imshow(rnflt_map, cmap='turbo')
                axes[0].set_title(f"Uploaded RNFLT Map ({label})")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], shrink=0.8, label="Thickness (µm)")
                
                # Plot 2: Difference Map
                im2 = axes[1].imshow(diff, cmap='bwr', vmin=-25, vmax=25)
                axes[1].set_title("Difference Map (vs. Healthy)")
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], shrink=0.8, label="Δ Thickness (µm)")
                
                # Plot 3: Risk Map
                im3 = axes[2].imshow(risk, cmap='hot') 
                axes[2].set_title("Risk Map (Thinner Zones)")
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2], shrink=0.8, label="Δ Thickness (µm)")
                
                plt.tight_layout()
                st.pyplot(fig)


elif "B-Scan" in analysis_type:
    # --- B-SCAN ANALYSIS (PHASE S) ---
    st.header("B-Scan Slice Analysis (Supervised CNN)")
    
    with st.container(border=True):
        model = load_bscan_model()
        if model is None:
            st.stop()
        uploaded_file = st.file_uploader("Upload a B-Scan image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        # Preprocess the image for the model
        img_batch, processed_img_display = preprocess_bscan_image(image_pil)
        
        # Run prediction
        pred_raw = model.predict(img_batch, verbose=0)[0][0]
        label = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
        confidence = pred_raw * 100 if label == "Glaucoma-like" else (1 - pred_raw) * 100
        
        # Display Results
        status_emoji = "🚨" if label == "Glaucoma-like" else "✅"
        st.markdown("---")
        
        st.metric(
            label="Prediction", 
            value=f"{status_emoji} {label}", 
            delta=f"{confidence:.2f}% Confidence", 
            delta_color="inverse" if label == "Glaucoma-like" else "normal"
        )
        
        # Generate Grad-CAM and display images in two columns for better look
        col_img, col_cam = st.columns([1, 2])
        
        with col_img:
            st.subheader("Original Image")
            st.image(image_pil, caption="Uploaded B-Scan Image", use_column_width=True)

        with col_cam:
            st.subheader("Model Interpretation (Grad-CAM)")
            heatmap = make_gradcam_heatmap(img_batch, model)
            
            if heatmap is not None:
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                superimposed_img = (np.stack([processed_img_display]*3, axis=-1) * 255).astype(np.uint8)
                superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap_color, 0.4, 0)
                
                col_c1, col_c2 = st.columns(2)
                col_c1.image(heatmap_color, caption="Heatmap", use_column_width=True)
                col_c2.image(superimposed_img, caption="Overlay: Areas of Focus", use_column_width=True)
            else:
                st.warning("Could not generate Grad-CAM visualization.")
