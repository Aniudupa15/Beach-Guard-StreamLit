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
ONNX_MODEL_PATH = "unet_beach.onnx"

# ========================
# Global models
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = None
session = None
input_name = None

def load_models():
    global yolo_model, session, input_name
    if yolo_model is None:
        yolo_model = YOLO("yolov8m.pt").to(device)
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        input_name = session.get_inputs()[0].name

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def process_video(input_path: str, output_path: str, safe_thresh=50, moderate_thresh=100):
    """Process video and return counts and alarm flags"""
    load_models()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    safe_count = 0
    moderate_count = 0
    danger_count = 0
    alarm_triggered = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay = frame.copy()
        frame_count += 1
        
        # YOLO Person Detection
        results = yolo_model.predict(frame, device=device, classes=[0], conf=0.2, verbose=False)
        detections = []
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
        
        # Distance Calculation
        label_offset = 0
        for (x1, y1, x2, y2, conf) in detections:
            person_feet = ((x1 + x2) // 2, y2)
            
            feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
            feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])
            if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
                continue
            
            region_class = mask[feet_y, feet_x]
            if region_class != 3:
                continue
            
            if len(shoreline_points) > 0:
                nearest_point = min(
                    shoreline_points,
                    key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1])
                )
                min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])
                
                if min_dist <= safe_thresh:
                    color = (0, 255, 0)
                    text = f"SAFE ({int(min_dist)}px)"
                    safe_count += 1
                elif safe_thresh < min_dist < moderate_thresh:
                    color = (0, 255, 255)
                    text = f"MODERATE ({int(min_dist)}px)"
                    moderate_count += 1
                else:
                    color = (0, 0, 255)
                    text = f"DANGER ({int(min_dist)}px)"
                    danger_count += 1
                    alarm_triggered = True
            else:
                continue
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_bg_x1 = x1
            text_bg_y1 = y1 - 10 - (label_offset * 20)
            text_bg_x2 = x1 + tw + 6
            text_bg_y2 = y1 + 5 - (label_offset * 20)
            cv2.rectangle(overlay, (text_bg_x1, text_bg_y1 - th - 5), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
            
            cv2.putText(overlay, text, (x1 + 3, y1 - 10 - (label_offset * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            label_offset = (label_offset + 1) % 3
        
        out.write(overlay)
    
    cap.release()
    out.release()
    
    return {
        "output_path": output_path,
        "safe_count": safe_count,
        "moderate_count": moderate_count,
        "danger_count": danger_count,
        "alarm_triggered": alarm_triggered
    }
