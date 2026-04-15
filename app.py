import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Vision Detector", layout="centered")

# --- UI Header ---
st.title("🔍 AI Object Detector")
st.write("Apni image upload krein aur AI automatically objects detect kr k dikhaye ga.")
st.markdown("---")

# --- Load Pre-trained Model ---
# Ye model free hai aur pehli dafa run krny pr automatically download ho jaye ga
@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')  # Nano version: Fast aur lightweight
    return model

model = load_model()

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Image select krein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    
    # UI Columns for Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_container_width=True)

    # Perform Detection
    with st.spinner('AI analysis kr rha hai...'):
        # Model inference
        results = model(image)
        
        # Result image (with boxes)
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1]) # Convert BGR to RGB

    with col2:
        st.success("AI Detection")
        st.image(res_image, use_container_width=True)

    # --- Details Section ---
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    # Get detected objects count
    names = model.names
    detected_objects = results[0].boxes.cls.tolist()
    
    if len(detected_objects) > 0:
        counts = {}
        for obj_id in detected_objects:
            obj_name = names[int(obj_id)]
            counts[obj_name] = counts.get(obj_name, 0) + 1
        
        for item, count in counts.items():
            st.write(f"- **{item.capitalize()}**: {count}")
    else:
        st.write("Koi object detect nahi hua.")

else:
    st.warning("Shuru krny k liye koi image upload krein.")
