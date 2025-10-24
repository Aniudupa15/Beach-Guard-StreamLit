# pip install ultralytics
# pip install onnxruntime
# pip install opencv-python-headless

import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO
import onnxruntime as ort

# ========================
# CONFIG
# ========================
IMG_SIZE = 640
input_video = "vid5.mp4"   # <-- your video path
output_video = "final_with_distance.mp4"
onnx_model_path = "unet_beach.onnx"

# ========================
# Load models
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8m.pt").to(device)
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_array = image.astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================
# Video Setup
# ========================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise FileNotFoundError("❌ Video not found!")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
print("⏳ Processing video...")

# ========================
# MAIN LOOP
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    overlay = frame.copy()
    frame_count += 1

    # --- YOLO Person Detection ---
    results = yolo_model.predict(frame, device=device, classes=[0], conf=0.2, verbose=False)
    detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append((x1, y1, x2, y2, conf))

    # --- UNet Segmentation ---
    input_tensor = preprocess_frame(frame)
    output = session.run(None, {input_name: input_tensor})
    mask = np.argmax(output[0], axis=1).squeeze()  # 0=background,1=sky,2=water,3=sand

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
            cv2.circle(overlay, (x_mapped, y_mapped), 2, (255, 0, 0), -1)  # blue shoreline dots

    # --- Distance Calculation ---
    label_offset = 0
    for (x1, y1, x2, y2, conf) in detections:
        person_feet = ((x1 + x2) // 2, y2)

        # Map feet to mask coordinates
        feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
        feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])
        if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
            continue

        region_class = mask[feet_y, feet_x]

        # ✅ Only process people in sand (class 3)
        if region_class != 3:
            continue

        if len(shoreline_points) > 0:
            nearest_point = min(
                shoreline_points,
                key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1])
            )
            min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])

            # 🟩🟨🟥 Three-level classification
            if min_dist <= 50:
                color = (0, 255, 0)
                text = f"SAFE ({int(min_dist)}px)"
            elif 50 < min_dist < 100:
                color = (0, 255, 255)
                text = f"MODERATE ({int(min_dist)}px)"
            else:
                color = (0, 0, 255)
                text = f"DANGER ({int(min_dist)}px)"
        else:
            continue

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Text background for readability
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

    # --- Write frame ---
    out.write(overlay)

    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"✅ Video saved as {output_video}")



# pip install fastapi uvicorn python-multipart
# pip install ultralytics onnxruntime opencv-python-headless

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse
# import cv2
# import torch
# import numpy as np
# import math
# from ultralytics import YOLO
# import onnxruntime as ort
# import tempfile
# import os
# from pathlib import Path
# import shutil

# # ========================
# # CONFIG
# # ========================
# IMG_SIZE = 640
# ONNX_MODEL_PATH = "unet_beach.onnx"

# # ========================
# # Initialize FastAPI
# # ========================
# app = FastAPI(title="Beach Safety Video Processing API")

# # ========================
# # Load models at startup
# # ========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# yolo_model = None
# session = None
# input_name = None

# @app.on_event("startup")
# async def load_models():
#     global yolo_model, session, input_name
#     print("🔄 Loading models...")
#     yolo_model = YOLO("yolov8m.pt").to(device)
#     session = ort.InferenceSession(ONNX_MODEL_PATH)
#     input_name = session.get_inputs()[0].name
#     print("✅ Models loaded successfully!")

# def preprocess_frame(frame):
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     img_array = image.astype(np.float32) / 255.0
#     img_array = np.transpose(img_array, (2, 0, 1))
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def process_video(input_path: str, output_path: str):
#     """Process video with YOLO detection and UNet segmentation"""
    
#     # Open video
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         raise ValueError("Cannot open video file")
    
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Create output writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     frame_count = 0
#     print("⏳ Processing video...")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         overlay = frame.copy()
#         frame_count += 1
        
#         # --- YOLO Person Detection ---
#         results = yolo_model.predict(frame, device=device, classes=[0], conf=0.2, verbose=False)
#         detections = []
#         if len(results[0].boxes) > 0:
#             for box in results[0].boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#                 conf = float(box.conf[0].cpu().numpy())
#                 detections.append((x1, y1, x2, y2, conf))
        
#         # --- UNet Segmentation ---
#         input_tensor = preprocess_frame(frame)
#         output = session.run(None, {input_name: input_tensor})
#         mask = np.argmax(output[0], axis=1).squeeze()  # 0=background,1=sky,2=water,3=sand
        
#         # --- Extract Shoreline Points ---
#         water_mask = (mask == 2).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         shoreline_points = []
#         for cnt in contours:
#             for pt in cnt:
#                 x, y = pt[0]
#                 if y > IMG_SIZE - 5 or x < 5 or x > IMG_SIZE - 5:
#                     continue
#                 x_mapped = int(x * frame.shape[1] / IMG_SIZE)
#                 y_mapped = int(y * frame.shape[0] / IMG_SIZE)
#                 shoreline_points.append((x_mapped, y_mapped))
#                 cv2.circle(overlay, (x_mapped, y_mapped), 2, (255, 0, 0), -1)  # blue shoreline dots
        
#         # --- Distance Calculation ---
#         label_offset = 0
#         for (x1, y1, x2, y2, conf) in detections:
#             person_feet = ((x1 + x2) // 2, y2)
            
#             # Map feet to mask coordinates
#             feet_x = int(person_feet[0] * IMG_SIZE / frame.shape[1])
#             feet_y = int(person_feet[1] * IMG_SIZE / frame.shape[0])
#             if not (0 <= feet_x < IMG_SIZE and 0 <= feet_y < IMG_SIZE):
#                 continue
            
#             region_class = mask[feet_y, feet_x]
            
#             # Only process people in sand (class 3)
#             if region_class != 3:
#                 continue
            
#             if len(shoreline_points) > 0:
#                 nearest_point = min(
#                     shoreline_points,
#                     key=lambda p: math.hypot(p[0] - person_feet[0], p[1] - person_feet[1])
#                 )
#                 min_dist = math.hypot(nearest_point[0] - person_feet[0], nearest_point[1] - person_feet[1])
                
#                 # Three-level classification
#                 if min_dist <= 50:
#                     color = (0, 255, 0)
#                     text = f"SAFE ({int(min_dist)}px)"
#                 elif 50 < min_dist < 100:
#                     color = (0, 255, 255)
#                     text = f"MODERATE ({int(min_dist)}px)"
#                 else:
#                     color = (0, 0, 255)
#                     text = f"DANGER ({int(min_dist)}px)"
#             else:
#                 continue
            
#             # Draw bounding box
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
#             # Text background for readability
#             (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#             text_bg_x1 = x1
#             text_bg_y1 = y1 - 10 - (label_offset * 20)
#             text_bg_x2 = x1 + tw + 6
#             text_bg_y2 = y1 + 5 - (label_offset * 20)
#             cv2.rectangle(overlay, (text_bg_x1, text_bg_y1 - th - 5), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
            
#             # Draw text
#             cv2.putText(overlay, text, (x1 + 3, y1 - 10 - (label_offset * 20)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
#             label_offset = (label_offset + 1) % 3
        
#         # Write frame
#         out.write(overlay)
        
#         if frame_count % 10 == 0:
#             print(f"Processed {frame_count} frames...")
    
#     cap.release()
#     out.release()
#     print(f"✅ Video processing complete! Total frames: {frame_count}")

# # ========================
# # API Endpoints
# # ========================

# @app.get("/")
# async def root():
#     return {
#         "message": "Beach Safety Video Processing API",
#         "endpoints": {
#             "/process-video": "POST - Upload video for processing",
#             "/health": "GET - Check API health"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "models_loaded": yolo_model is not None and session is not None,
#         "device": device
#     }

# @app.post("/process-video")
# async def process_video_endpoint(video: UploadFile = File(...)):
#     """
#     Upload a video file and receive a processed video with distance annotations
#     """
    
#     # Validate file type
#     if not video.content_type.startswith('video/'):
#         raise HTTPException(status_code=400, detail="File must be a video")
    
#     # Create temporary files
#     temp_dir = tempfile.mkdtemp()
#     input_path = os.path.join(temp_dir, f"input_{video.filename}")
#     output_path = os.path.join(temp_dir, f"processed_{video.filename}")
    
#     try:
#         # Save uploaded video
#         with open(input_path, "wb") as buffer:
#             shutil.copyfileobj(video.file, buffer)
        
#         # Process video
#         process_video(input_path, output_path)
        
#         # Return processed video
#         return FileResponse(
#             output_path,
#             media_type="video/mp4",
#             filename=f"processed_{video.filename}",
#             background=None  # Keep file until response is sent
#         )
    
#     except Exception as e:
#         # Cleanup on error
#         shutil.rmtree(temp_dir, ignore_errors=True)
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
#     finally:
#         # Schedule cleanup after response (optional - may want to keep for debugging)
#         # Note: FileResponse will handle file deletion if background task is set
#         pass

# # ========================
# # Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# # ========================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)