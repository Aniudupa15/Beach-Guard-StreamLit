import streamlit as st
import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO
import onnxruntime as ort
import tempfile
import os

# ========================
# CONFIG
# ========================
IMG_SIZE = 640

# ========================
# Page Config
# ========================
st.set_page_config(
    page_title="Beach Safety Monitor",
    page_icon="🏖️",
    layout="wide"
)

st.title("🏖️ Beach Safety Monitoring System")
st.markdown("Upload a video to detect people on the beach and assess their distance from the shoreline.")

# ========================
# Sidebar Configuration
# ========================
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.2, 0.05)
safe_distance = st.sidebar.number_input("Safe Distance (px)", 10, 100, 50, 5)
danger_distance = st.sidebar.number_input("Danger Distance (px)", 50, 200, 100, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Distance Classification")
st.sidebar.markdown(f"🟩 **SAFE**: ≤ {safe_distance}px")
st.sidebar.markdown(f"🟨 **MODERATE**: {safe_distance}-{danger_distance}px")
st.sidebar.markdown(f"🟥 **DANGER**: > {danger_distance}px")

# ========================
# Model Loading
# ========================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO("yolov8m.pt").to(device)
    session = ort.InferenceSession("unet_beach.onnx")
    input_name = session.get_inputs()[0].name
    return yolo_model, session, input_name, device

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================
# Video Processing Function
# ========================
def process_video(input_path, yolo_model, session, input_name, device, safe_dist, danger_dist, conf_thresh):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay = frame.copy()
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # --- YOLO Person Detection ---
        results = yolo_model.predict(frame, device=device, classes=[0], conf=conf_thresh, verbose=False)
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                detections.append((x1, y1, x2, y2, conf))
        
        # --- UNet Segmentation ---
        input_tensor = preprocess_frame(frame)
        output = session.run(None, {input_name: input_tensor})
        mask = np.argmax(output[0], axis=1).squeeze()
        
        # --- Extract Shoreline Points ---
        water_mask = (mask == 2).astype(np.uint8) * 255
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shoreline_points = []
        for cnt in contours:
            for pt in cnt:
                x, y = pt[0]
                if y > IMG_SIZE - 5 or x < 5 or x > IMG_SIZE - 5:
                    continue
                x_mapped = int(x * frame.shape[1] / IMG_SIZE)
                y_mapped = int(y * frame.shape[0] / IMG_SIZE)
                shoreline_points.append((x_mapped, y_mapped))
                cv2.circle(overlay, (x_mapped, y_mapped), 2, (255, 0, 0), -1)
        
        # --- Distance Calculation ---
        label_offset = 0
        for (x1, y1, x2, y2, conf) in detections:
            person_feet = ((x1 + x2) // 2, y2)
            
            feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
            feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])
            if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
                continue
            
            region_class = mask[feet_y, feet_x]
            
            # Only process people in sand (class 3)
            if region_class != 3:
                continue
            
            if len(shoreline_points) > 0:
                nearest_point = min(
                    shoreline_points,
                    key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1])
                )
                min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])
                
                # Three-level classification
                if min_dist <= safe_dist:
                    color = (0, 255, 0)
                    text = f"SAFE ({int(min_dist)}px)"
                elif safe_dist < min_dist < danger_dist:
                    color = (0, 255, 255)
                    text = f"MODERATE ({int(min_dist)}px)"
                else:
                    color = (0, 0, 255)
                    text = f"DANGER ({int(min_dist)}px)"
            else:
                continue
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Text background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_bg_x1 = x1
            text_bg_y1 = y1 - 10 - (label_offset * 20)
            text_bg_x2 = x1 + tw + 6
            text_bg_y2 = y1 + 5 - (label_offset * 20)
            cv2.rectangle(overlay, (text_bg_x1, text_bg_y1 - th - 5), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(overlay, text, (x1 + 3, y1 - 10 - (label_offset * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            label_offset = (label_offset + 1) % 3
        
        out.write(overlay)
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    return output_path, None

# ========================
# Main App
# ========================
uploaded_file = st.file_uploader("📹 Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.video(uploaded_file)
    
    if st.button("🚀 Process Video", type="primary"):
        with st.spinner("Loading models..."):
            try:
                yolo_model, session, input_name, device = load_models()
                st.success(f"✅ Models loaded successfully! Using device: {device}")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.stop()
        
        with st.spinner("Processing video..."):
            output_path, error = process_video(
                tfile.name, 
                yolo_model, 
                session, 
                input_name, 
                device,
                safe_distance,
                danger_distance,
                conf_threshold
            )
            
            if error:
                st.error(error)
            else:
                st.success("✅ Video processed successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Video")
                    st.video(tfile.name)
                
                with col2:
                    st.subheader("Processed Video")
                    st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=f,
                        file_name="beach_safety_output.mp4",
                        mime="video/mp4"
                    )
                
                # Cleanup
                os.unlink(output_path)
        
        # Cleanup uploaded file
        os.unlink(tfile.name)

else:
    st.info("👆 Please upload a video file to get started")
    
    # Display sample information
    st.markdown("---")
    st.markdown("### 📋 How it works:")
    st.markdown("""
    1. **Person Detection**: YOLOv8 detects people in the video
    2. **Segmentation**: UNet model segments the beach into sky, water, and sand regions
    3. **Shoreline Detection**: Extracts water-sand boundary points
    4. **Distance Calculation**: Measures distance from each person to the nearest shoreline point
    5. **Risk Assessment**: Classifies safety level based on distance thresholds
    """)
    
    st.markdown("### 🎯 Requirements:")
    st.markdown("""
    - `yolov8m.pt` - YOLOv8 model weights
    - `unet_beach.onnx` - Beach segmentation model
    - Both files should be in the same directory as this script
    """)