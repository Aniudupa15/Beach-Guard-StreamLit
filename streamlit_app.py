import streamlit as st
import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO
import onnxruntime as ort
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import time
import threading
import gdown

# ========================
# CONFIG
# ========================
IMG_SIZE = 640
ONNX_MODEL_PATH = "unet_beach.onnx"
YOLO_MODEL_PATH = "yolov8m.pt"

# FIXED: Extract proper file IDs from Google Drive URLs
ONNX_FILE_ID = "1KOPbrwUtYJ0yGI8tc2qjBLVLfZ-PY5Ji"  # Extract ID from URL
YOLO_FILE_ID = "1yDj15y9cBl16fdraY0zNBHwfaMpuJMQq"  # Extract ID from URL

DEFAULT_SAFE_DIST = 50
DEFAULT_MODERATE_DIST = 100
DEFAULT_PASSWORD = "admin123"

# ========================
# FIXED: Improved download with validation
# ========================
def download_model(path, file_id, model_name, min_size_mb=1):
    """Download model from Google Drive with validation"""
    min_size_bytes = min_size_mb * 1024 * 1024
    
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        if file_size < min_size_bytes:
            st.warning(f"{model_name} exists but appears corrupted ({file_size/1024:.2f} KB). Re-downloading...")
            os.remove(path)
        else:
            st.info(f"✅ {model_name} already exists ({file_size/(1024*1024):.2f} MB)")
            return True
    
    try:
        st.info(f"📥 Downloading {model_name}, please wait...")
        
        # Force gdown to use direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Try download with multiple methods
        try:
            gdown.download(url, path, quiet=False, fuzzy=True)
        except:
            # Fallback: try without fuzzy
            st.warning("Retrying download with different method...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
        
        # Validate download
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            if file_size >= min_size_bytes:
                st.success(f"✅ {model_name} downloaded successfully! ({file_size/(1024*1024):.2f} MB)")
                return True
            else:
                st.error(f"❌ {model_name} download failed - file too small ({file_size/1024:.2f} KB)")
                st.error("This usually means the Google Drive file is not publicly accessible.")
                st.info(f"Please ensure the file is shared as 'Anyone with the link can view'")
                return False
        else:
            st.error(f"❌ {model_name} file not created after download")
            return False
    except Exception as e:
        st.error(f"❌ Error downloading {model_name}: {str(e)}")
        st.info("Try: 1) Check file sharing permissions 2) Use alternative hosting")
        return False

# FORCE CLEANUP: Remove corrupted files
for path in [ONNX_MODEL_PATH, YOLO_MODEL_PATH]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        if size_mb < 1:  # Less than 1MB is corrupted
            st.warning(f"🗑️ Removing corrupted file: {path} ({size_mb:.3f} MB)")
            os.remove(path)

# Download models before loading
with st.spinner("Initializing models..."):
    onnx_ready = download_model(ONNX_MODEL_PATH, ONNX_FILE_ID, "ONNX Model", min_size_mb=5)
    yolo_ready = download_model(YOLO_MODEL_PATH, YOLO_FILE_ID, "YOLO Model", min_size_mb=20)
    
    if not (onnx_ready and yolo_ready):
        st.error("❌ Model download failed. Please check your Google Drive links and permissions.")
        st.stop()

# ========================
# Load models globally with error handling
# ========================
@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load YOLO with error handling
        st.info("Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
        
        # Load ONNX with validation
        st.info("Loading ONNX model...")
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
        
        file_size = os.path.getsize(ONNX_MODEL_PATH)
        st.info(f"ONNX model size: {file_size / (1024*1024):.2f} MB")
        
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_name = session.get_inputs()[0].name
        
        st.success("✅ All models loaded successfully!")
        return yolo_model, session, input_name, device
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please check:\n1. Model files are valid\n2. Google Drive links have public access\n3. Files downloaded completely")
        raise e

yolo_model, session, input_name, device = load_models()

# ========================
# Utility Functions
# ========================
def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def process_frame(frame, safe_dist, moderate_dist):
    overlay = frame.copy()
    detections = []
    shoreline_points = []

    # YOLO Person Detection
    results = yolo_model.predict(frame, device=device, classes=[0], conf=0.2, verbose=False)
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append((x1, y1, x2, y2, conf))

    # UNet Segmentation
    input_tensor = preprocess_frame(frame)
    output = session.run(None, {input_name: input_tensor})
    mask = np.argmax(output[0], axis=1).squeeze()

    # Extract Shoreline Points
    water_mask = (mask == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            if y > IMG_SIZE - 5 or x < 5 or x > IMG_SIZE - 5:
                continue
            x_mapped = int(x * frame.shape[1] / IMG_SIZE)
            y_mapped = int(y * frame.shape[0] / IMG_SIZE)
            shoreline_points.append((x_mapped, y_mapped))
            cv2.circle(overlay, (x_mapped, y_mapped), 2, (255, 0, 0), -1)

    # Distance Calculation
    people_counts = {"safe": 0, "moderate": 0, "danger": 0}
    label_offset = 0
    for (x1, y1, x2, y2, conf) in detections:
        person_feet = ((x1 + x2) // 2, y2)
        feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
        feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])
        if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
            continue

        region_class = mask[feet_y, feet_x]
        if region_class != 3:  # Only sand
            continue

        if len(shoreline_points) > 0:
            nearest_point = min(shoreline_points, key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1]))
            min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])

            if min_dist <= safe_dist:
                color = (0, 255, 0)
                text = f"SAFE ({int(min_dist)}px)"
                people_counts["safe"] += 1
            elif safe_dist < min_dist < moderate_dist:
                color = (0, 255, 255)
                text = f"MODERATE ({int(min_dist)}px)"
                people_counts["moderate"] += 1
            else:
                color = (0, 0, 255)
                text = f"DANGER ({int(min_dist)}px)"
                people_counts["danger"] += 1
        else:
            continue

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_bg_x1 = x1
        text_bg_y1 = y1 - 10 - (label_offset * 20)
        text_bg_x2 = x1 + tw + 6
        text_bg_y2 = y1 + 5 - (label_offset * 20)
        cv2.rectangle(overlay, (text_bg_x1, text_bg_y1 - th - 5), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
        cv2.putText(overlay, text, (x1 + 3, y1 - 10 - (label_offset * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        label_offset = (label_offset + 1) % 3

    return overlay, people_counts

# ========================
# Session State
# ========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "safe_dist" not in st.session_state:
    st.session_state.safe_dist = DEFAULT_SAFE_DIST
if "moderate_dist" not in st.session_state:
    st.session_state.moderate_dist = DEFAULT_MODERATE_DIST
if "input_type" not in st.session_state:
    st.session_state.input_type = "upload"
if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False
if "people_counts" not in st.session_state:
    st.session_state.people_counts = {"safe": 0, "moderate": 0, "danger": 0}

# ========================
# CSS Styling
# ========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #87CEEB 0%, #FFD700 100%);
    background-attachment: fixed;
}
.stButton>button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.card {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
}
.alarm {
    background: linear-gradient(45deg, #FF0000, #FF4500);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    animation: blink 1s infinite;
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.title {
    text-align: center;
    color: #2E8B57;
    font-size: 3em;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ========================
# Pages
# ========================
def landing_page():
    st.markdown('<h1 class="title">🏖 Beach Guard AI 🏖</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em;">Intelligent shoreline monitoring and safety system</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📹 Upload Video", key="upload_btn"):
            st.session_state.input_type = "upload"
            st.rerun()
    with col2:
        if st.button("📷 Camera Feed", key="camera_btn"):
            st.session_state.input_type = "camera"
            st.rerun()

    if st.session_state.input_type == "upload":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, counts = process_frame(frame, st.session_state.safe_dist, st.session_state.moderate_dist)
                st.session_state.people_counts = counts
                stframe.image(processed_frame, channels="BGR", use_container_width=True)
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                time.sleep(0.03)

            cap.release()
            os.unlink(video_path)
            st.success("Video processing complete! 🎉")

    elif st.session_state.input_type == "camera":
        st.markdown("### 📷 Real-time Camera Feed")
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class VideoProcessor:
            def __init__(self):
                self.frame_lock = threading.Lock()
                self.out_image = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                processed_img, counts = process_frame(img, st.session_state.safe_dist, st.session_state.moderate_dist)
                st.session_state.people_counts = counts
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

def dashboard_page():
    st.markdown('<h1 class="title">📊 Dashboard 📊</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h2>🟢 Safe</h2><h1>{st.session_state.people_counts["safe"]}</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h2>🟡 Moderate</h2><h1>{st.session_state.people_counts["moderate"]}</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h2>🔴 Danger</h2><h1>{st.session_state.people_counts["danger"]}</h1></div>', unsafe_allow_html=True)

    if st.session_state.people_counts["danger"] > 0:
        st.session_state.alarm_active = True
        st.markdown('<div class="alarm"><h2>🚨 DANGER ALERT! 🚨</h2><p>People detected in danger zone!</p></div>', unsafe_allow_html=True)
        if st.button("🔔 Stop Alarm", key="stop_alarm"):
            st.session_state.alarm_active = False
            st.success("Alarm stopped.")
    else:
        st.session_state.alarm_active = False

def settings_page():
    st.markdown('<h1 class="title">⚙ Settings ⚙</h1>', unsafe_allow_html=True)

    if not st.session_state.authenticated:
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if password == DEFAULT_PASSWORD:
                st.session_state.authenticated = True
                st.success("Authenticated! ✅")
                st.rerun()
            else:
                st.error("Incorrect password ❌")
        return

    st.markdown("### Distance Thresholds (pixels)")
    col1, col2 = st.columns(2)
    with col1:
        safe = st.slider("Safe Distance", 10, 100, st.session_state.safe_dist)
    with col2:
        moderate = st.slider("Moderate Distance", 50, 200, st.session_state.moderate_dist)

    if st.button("Save Settings"):
        st.session_state.safe_dist = safe
        st.session_state.moderate_dist = moderate
        st.success("Settings saved! 💾")

    st.markdown("### Input Type")
    input_choice = st.radio("Default Input", ["upload", "camera"], index=0 if st.session_state.input_type == "upload" else 1)
    if st.button("Set Input Type"):
        st.session_state.input_type = input_choice
        st.success("Input type updated! 🔄")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ========================
# Main App
# ========================
def main():
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Landing", "📊 Dashboard", "⚙ Settings"])

    if page == "🏠 Landing":
        landing_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "⚙ Settings":
        settings_page()

if __name__ == "__main__":
    main()import streamlit as st
import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO
import onnxruntime as ort
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import time
import threading
import gdown

# ========================
# CONFIG
# ========================
IMG_SIZE = 640
ONNX_MODEL_PATH = "unet_beach.onnx"
YOLO_MODEL_PATH = "yolov8m.pt"

# FIXED: Extract proper file IDs from Google Drive URLs
ONNX_FILE_ID = "1KOPbrwUtYJ0yGI8tc2qjBLVLfZ-PY5Ji"  # Extract ID from URL
YOLO_FILE_ID = "1yDj15y9cBl16fdraY0zNBHwfaMpuJMQq"  # Extract ID from URL

DEFAULT_SAFE_DIST = 50
DEFAULT_MODERATE_DIST = 100
DEFAULT_PASSWORD = "admin123"

# ========================
# FIXED: Improved download with validation
# ========================
def download_model(path, file_id, model_name, min_size_mb=1):
    """Download model from Google Drive with validation"""
    min_size_bytes = min_size_mb * 1024 * 1024
    
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        if file_size < min_size_bytes:
            st.warning(f"{model_name} exists but appears corrupted ({file_size/1024:.2f} KB). Re-downloading...")
            os.remove(path)
        else:
            st.info(f"✅ {model_name} already exists ({file_size/(1024*1024):.2f} MB)")
            return True
    
    try:
        st.info(f"📥 Downloading {model_name}, please wait...")
        
        # Force gdown to use direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Try download with multiple methods
        try:
            gdown.download(url, path, quiet=False, fuzzy=True)
        except:
            # Fallback: try without fuzzy
            st.warning("Retrying download with different method...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
        
        # Validate download
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            if file_size >= min_size_bytes:
                st.success(f"✅ {model_name} downloaded successfully! ({file_size/(1024*1024):.2f} MB)")
                return True
            else:
                st.error(f"❌ {model_name} download failed - file too small ({file_size/1024:.2f} KB)")
                st.error("This usually means the Google Drive file is not publicly accessible.")
                st.info(f"Please ensure the file is shared as 'Anyone with the link can view'")
                return False
        else:
            st.error(f"❌ {model_name} file not created after download")
            return False
    except Exception as e:
        st.error(f"❌ Error downloading {model_name}: {str(e)}")
        st.info("Try: 1) Check file sharing permissions 2) Use alternative hosting")
        return False

# FORCE CLEANUP: Remove corrupted files
for path in [ONNX_MODEL_PATH, YOLO_MODEL_PATH]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        if size_mb < 1:  # Less than 1MB is corrupted
            st.warning(f"🗑️ Removing corrupted file: {path} ({size_mb:.3f} MB)")
            os.remove(path)

# Download models before loading
with st.spinner("Initializing models..."):
    onnx_ready = download_model(ONNX_MODEL_PATH, ONNX_FILE_ID, "ONNX Model", min_size_mb=5)
    yolo_ready = download_model(YOLO_MODEL_PATH, YOLO_FILE_ID, "YOLO Model", min_size_mb=20)
    
    if not (onnx_ready and yolo_ready):
        st.error("❌ Model download failed. Please check your Google Drive links and permissions.")
        st.stop()

# ========================
# Load models globally with error handling
# ========================
@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load YOLO with error handling
        st.info("Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
        
        # Load ONNX with validation
        st.info("Loading ONNX model...")
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")
        
        file_size = os.path.getsize(ONNX_MODEL_PATH)
        st.info(f"ONNX model size: {file_size / (1024*1024):.2f} MB")
        
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_name = session.get_inputs()[0].name
        
        st.success("✅ All models loaded successfully!")
        return yolo_model, session, input_name, device
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please check:\n1. Model files are valid\n2. Google Drive links have public access\n3. Files downloaded completely")
        raise e

yolo_model, session, input_name, device = load_models()

# ========================
# Utility Functions
# ========================
def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def process_frame(frame, safe_dist, moderate_dist):
    overlay = frame.copy()
    detections = []
    shoreline_points = []

    # YOLO Person Detection
    results = yolo_model.predict(frame, device=device, classes=[0], conf=0.2, verbose=False)
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append((x1, y1, x2, y2, conf))

    # UNet Segmentation
    input_tensor = preprocess_frame(frame)
    output = session.run(None, {input_name: input_tensor})
    mask = np.argmax(output[0], axis=1).squeeze()

    # Extract Shoreline Points
    water_mask = (mask == 2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for pt in cnt:
            x, y = pt[0]
            if y > IMG_SIZE - 5 or x < 5 or x > IMG_SIZE - 5:
                continue
            x_mapped = int(x * frame.shape[1] / IMG_SIZE)
            y_mapped = int(y * frame.shape[0] / IMG_SIZE)
            shoreline_points.append((x_mapped, y_mapped))
            cv2.circle(overlay, (x_mapped, y_mapped), 2, (255, 0, 0), -1)

    # Distance Calculation
    people_counts = {"safe": 0, "moderate": 0, "danger": 0}
    label_offset = 0
    for (x1, y1, x2, y2, conf) in detections:
        person_feet = ((x1 + x2) // 2, y2)
        feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
        feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])
        if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
            continue

        region_class = mask[feet_y, feet_x]
        if region_class != 3:  # Only sand
            continue

        if len(shoreline_points) > 0:
            nearest_point = min(shoreline_points, key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1]))
            min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])

            if min_dist <= safe_dist:
                color = (0, 255, 0)
                text = f"SAFE ({int(min_dist)}px)"
                people_counts["safe"] += 1
            elif safe_dist < min_dist < moderate_dist:
                color = (0, 255, 255)
                text = f"MODERATE ({int(min_dist)}px)"
                people_counts["moderate"] += 1
            else:
                color = (0, 0, 255)
                text = f"DANGER ({int(min_dist)}px)"
                people_counts["danger"] += 1
        else:
            continue

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_bg_x1 = x1
        text_bg_y1 = y1 - 10 - (label_offset * 20)
        text_bg_x2 = x1 + tw + 6
        text_bg_y2 = y1 + 5 - (label_offset * 20)
        cv2.rectangle(overlay, (text_bg_x1, text_bg_y1 - th - 5), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
        cv2.putText(overlay, text, (x1 + 3, y1 - 10 - (label_offset * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        label_offset = (label_offset + 1) % 3

    return overlay, people_counts

# ========================
# Session State
# ========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "safe_dist" not in st.session_state:
    st.session_state.safe_dist = DEFAULT_SAFE_DIST
if "moderate_dist" not in st.session_state:
    st.session_state.moderate_dist = DEFAULT_MODERATE_DIST
if "input_type" not in st.session_state:
    st.session_state.input_type = "upload"
if "alarm_active" not in st.session_state:
    st.session_state.alarm_active = False
if "people_counts" not in st.session_state:
    st.session_state.people_counts = {"safe": 0, "moderate": 0, "danger": 0}

# ========================
# CSS Styling
# ========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #87CEEB 0%, #FFD700 100%);
    background-attachment: fixed;
}
.stButton>button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.card {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
}
.alarm {
    background: linear-gradient(45deg, #FF0000, #FF4500);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    animation: blink 1s infinite;
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.title {
    text-align: center;
    color: #2E8B57;
    font-size: 3em;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ========================
# Pages
# ========================
def landing_page():
    st.markdown('<h1 class="title">🏖 Beach Guard AI 🏖</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em;">Intelligent shoreline monitoring and safety system</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📹 Upload Video", key="upload_btn"):
            st.session_state.input_type = "upload"
            st.rerun()
    with col2:
        if st.button("📷 Camera Feed", key="camera_btn"):
            st.session_state.input_type = "camera"
            st.rerun()

    if st.session_state.input_type == "upload":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, counts = process_frame(frame, st.session_state.safe_dist, st.session_state.moderate_dist)
                st.session_state.people_counts = counts
                stframe.image(processed_frame, channels="BGR", use_container_width=True)
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                time.sleep(0.03)

            cap.release()
            os.unlink(video_path)
            st.success("Video processing complete! 🎉")

    elif st.session_state.input_type == "camera":
        st.markdown("### 📷 Real-time Camera Feed")
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class VideoProcessor:
            def __init__(self):
                self.frame_lock = threading.Lock()
                self.out_image = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                processed_img, counts = process_frame(img, st.session_state.safe_dist, st.session_state.moderate_dist)
                st.session_state.people_counts = counts
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

def dashboard_page():
    st.markdown('<h1 class="title">📊 Dashboard 📊</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h2>🟢 Safe</h2><h1>{st.session_state.people_counts["safe"]}</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h2>🟡 Moderate</h2><h1>{st.session_state.people_counts["moderate"]}</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h2>🔴 Danger</h2><h1>{st.session_state.people_counts["danger"]}</h1></div>', unsafe_allow_html=True)

    if st.session_state.people_counts["danger"] > 0:
        st.session_state.alarm_active = True
        st.markdown('<div class="alarm"><h2>🚨 DANGER ALERT! 🚨</h2><p>People detected in danger zone!</p></div>', unsafe_allow_html=True)
        if st.button("🔔 Stop Alarm", key="stop_alarm"):
            st.session_state.alarm_active = False
            st.success("Alarm stopped.")
    else:
        st.session_state.alarm_active = False

def settings_page():
    st.markdown('<h1 class="title">⚙ Settings ⚙</h1>', unsafe_allow_html=True)

    if not st.session_state.authenticated:
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if password == DEFAULT_PASSWORD:
                st.session_state.authenticated = True
                st.success("Authenticated! ✅")
                st.rerun()
            else:
                st.error("Incorrect password ❌")
        return

    st.markdown("### Distance Thresholds (pixels)")
    col1, col2 = st.columns(2)
    with col1:
        safe = st.slider("Safe Distance", 10, 100, st.session_state.safe_dist)
    with col2:
        moderate = st.slider("Moderate Distance", 50, 200, st.session_state.moderate_dist)

    if st.button("Save Settings"):
        st.session_state.safe_dist = safe
        st.session_state.moderate_dist = moderate
        st.success("Settings saved! 💾")

    st.markdown("### Input Type")
    input_choice = st.radio("Default Input", ["upload", "camera"], index=0 if st.session_state.input_type == "upload" else 1)
    if st.button("Set Input Type"):
        st.session_state.input_type = input_choice
        st.success("Input type updated! 🔄")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ========================
# Main App
# ========================
def main():
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Landing", "📊 Dashboard", "⚙ Settings"])

    if page == "🏠 Landing":
        landing_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "⚙ Settings":
        settings_page()

if __name__ == "__main__":
    main()
