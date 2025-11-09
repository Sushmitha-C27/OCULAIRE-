import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io

# --- App Configuration ---
st.set_page_config(page_title="OCULAIRE Glaucoma Detection", layout="wide")
st.title("👁️ OCULAIRE: Glaucoma Detection Dashboard")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_bscan_model():
    try:
        model = tf.keras.models.load_model("bscan_cnn.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading B-Scan CNN model: {e}")
        return None

@st.cache_resource
def load_rnflt_models():
    try:
        scaler = joblib.load("rnflt_scaler.joblib")
        kmeans = joblib.load("rnflt_kmeans.joblib")
        avg_healthy = np.load("avg_map_healthy.npy")
        avg_glaucoma = np.load("avg_map_glaucoma.npy")
        
        # Re-create thin/thick logic from your notebook
        # This assumes cluster 0 is thin (glaucoma) and cluster 1 is thick (healthy)
        # You may need to adjust this logic based on your saved kmeans model
        # A simple check:
        if np.mean(avg_healthy) > np.mean(avg_glaucoma):
            thick_cluster = 1
            thin_cluster = 0
        else:
            thick_cluster = 0
            thin_cluster = 1
            
        return scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster
    except Exception as e:
        st.error(f"Error loading RNFLT artifacts: {e}")
        st.info("Please make sure 'rnflt_scaler.joblib', 'rnflt_kmeans.joblib', 'avg_map_healthy.npy', and 'avg_map_glaucoma.npy' are in the same directory.")
        return None, None, None, None, None, None

# ==============================================================================
# 1. RNFLT (PHASE D) HELPER FUNCTIONS
# ==============================================================================
def process_uploaded_npz(uploaded_file):
    """Loads NPZ file from Streamlit's uploader and extracts metrics."""
    try:
        # Load NPZ from bytes
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        npz = np.load(file_bytes, allow_pickle=True)
        rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
        
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
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < threshold, diff, np.nan)
    
    total_pixels = np.isfinite(diff).sum()
    risky_pixels = np.isfinite(risk).sum()
    severity = (risky_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    return diff, risk, severity

# ==============================================================================
# 2. B-SCAN (PHASE S) HELPER FUNCTIONS
# ==============================================================================
def preprocess_bscan_image(image_pil, img_size=(224, 224)):
    """
    Replicates the *exact* preprocessing from your BscanDataGenerator.
    Takes a PIL Image -> returns a processed numpy array for the model.
    """
    # Convert PIL to numpy array (grayscale)
    arr = np.array(image_pil.convert('L'))
    
    # 1. Clip to 99th percentile
    arr = np.clip(arr, 0, np.percentile(arr, 99))
    
    # 2. Min-Max Normalize
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    
    # 3. Resize (using OpenCV to match tf.image.resize)
    arr_resized = cv2.resize(arr, img_size, interpolation=cv2.INTER_NEAREST)
    
    # 4. Stack grayscale to 3-channel RGB
    arr_rgb = np.repeat(arr_resized[..., None], 3, axis=-1)
    
    # 5. Add batch dimension
    img_batch = np.expand_dims(arr_rgb, axis=0).astype(np.float32)
    return img_batch, arr_resized

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Generates Grad-CAM heatmap (copied from your S2+ script)."""
    if last_conv_layer_name is None:
        # Auto-detect last Conv2D layer
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
# --- MAIN APPLICATION UI ---
# ==============================================================================

analysis_type = st.radio(
    "Select Analysis Type",
    ("🩺 RNFLT Map Analysis (.npz)", "👁️ B-Scan Slice Analysis (Image)"),
    horizontal=True
)

st.markdown("---")

if "RNFLT" in analysis_type:
    # --- RNFLT ANALYSIS (PHASE D) ---
    st.header("RNFLT Map Analysis (Unsupervised)")
    
    # Load models
    scaler, kmeans, avg_healthy, avg_glaucoma, thin_cluster, thick_cluster = load_rnflt_models()
    
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
            label = "Healthy-like" if cluster == thick_cluster else "Glaucoma-like"

            # 2. Compute Risk
            diff, risk, severity = compute_risk_map(rnflt_map, avg_healthy)

            # 3. Display Results
            st.subheader(f"🩺 Classification Result: **{label}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Cluster", cluster)
            col2.metric("Mean RNFLT", f"{metrics['mean']:.2f} µm")
            col3.metric("Severity Score", f"{severity:.2f}%", help="Percentage of area thinner than healthy average")

            st.markdown("---")
            st.subheader("RNFLT Visualization")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(rnflt_map, cmap='turbo')
            axes[0].set_title(f"Uploaded RNFLT Map ({label})")
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], shrink=0.6, label="Thickness (µm)")
            
            im2 = axes[1].imshow(diff, cmap='bwr', vmin=-20, vmax=20)
            axes[1].set_title("Difference Map (vs. Healthy)")
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], shrink=0.6, label="Δ Thickness (µm)")
            
            im3 = axes[2].imshow(risk, cmap='hot')
            axes[2].set_title("Risk Map (Thinner Zones)")
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], shrink=0.6, label="Δ Thickness (µm)")
            
            plt.tight_layout()
            st.pyplot(fig)


elif "B-Scan" in analysis_type:
    # --- B-SCAN ANALYSIS (PHASE S) ---
    st.header("B-Scan Slice Analysis (Supervised CNN)")
    
    model = load_bscan_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload a B-Scan image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        
        # Preprocess the image for the model
        img_batch, processed_img_display = preprocess_bscan_image(image_pil)
        
        # Run prediction
        pred_raw = model.predict(img_batch)[0][0]
        label = "Glaucoma-like" if pred_raw > 0.5 else "Healthy-like"
        confidence = pred_raw * 100 if label == "Glaucoma-like" else (1 - pred_raw) * 100
        
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_batch, model)
        
        # Display Results
        st.subheader(f"🩺 Classification Result: **{label}** (Confidence: {confidence:.2f}%)")
        
        # Create overlay
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Use the *processed* grayscale image for overlay
        superimposed_img = (np.stack([processed_img_display]*3, axis=-1) * 255).astype(np.uint8)
        superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap_color, 0.4, 0)
        
        col1, col2, col3 = st.columns(3)
        col1.image(image_pil, caption="Original Uploaded Image", use_column_width=True)
        col2.image(heatmap_color, caption="Grad-CAM Heatmap", use_column_width=True)
        col3.image(superimposed_img, caption="Heatmap Overlay", use_column_width=True)