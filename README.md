# Beach Guard AI

An intelligent shoreline monitoring and safety system built with Streamlit, YOLOv8, and U-Net segmentation. This application detects people on the beach and classifies them as safe, moderate, or in danger based on their distance from the shoreline, helping prevent accidents and enhance beach safety.

## Features

- **Real-time Person Detection**: Uses YOLOv8 to detect people in video feeds or uploaded videos.
- **Shoreline Segmentation**: Employs a U-Net model to identify water and sand regions for accurate distance calculations.
- **Distance Classification**: Categorizes individuals into three zones:
  - 🟢 Safe (within 50 pixels of shoreline)
  - 🟡 Moderate (50-100 pixels)
  - 🔴 Danger (beyond 100 pixels)
- **Multi-Input Support**: Process uploaded video files or use real-time camera feed via WebRTC.
- **Interactive Dashboard**: View live counts of people in each zone with visual alerts.
- **Customizable Settings**: Adjust distance thresholds and input preferences (password-protected).
- **Automatic Model Download**: Downloads required AI models (YOLO and U-Net) on first run if not present.

## Prerequisites

- Python 3.8 or higher
- Webcam (for camera feed functionality)
- Internet connection (for initial model downloads)

## Installation

1. **Clone or Download the Repository**:
   - Ensure you have the `Beach-Guard-StreamLit` folder containing the project files.

2. **Navigate to the Project Directory**:
   ```
   cd Beach-Guard-StreamLit
   ```

3. **Install Dependencies**:
   - Run the following command to install all required packages:
     ```
     pip install -r requirements.txt
     ```
   - This will install libraries such as Streamlit, OpenCV, PyTorch, Ultralytics, ONNX Runtime, and others.

4. **Verify Installation**:
   - Ensure Python and pip are properly installed by running:
     ```
     python --version
     pip --version
     ```

## Running the Application

1. **Start the Streamlit App**:
   - From the project directory, run:
     ```
     streamlit run streamlit_app.py
     ```
   - This will launch the app in your default web browser at `http://localhost:8501`.

2. **First Run**:
   - On the first run, the app will automatically download the required AI models (`yolov8m.pt` and `unet_beach.onnx`) from Google Drive. This may take a few minutes depending on your internet speed.
   - A progress indicator will show the download status.

3. **Alternative Script**:
   - For direct video processing without the Streamlit interface, you can run:
     ```
     python app.py
     ```
     - Note: This script processes a hardcoded video file (`vid5.mp4`) and outputs `final_with_distance.mp4`. Modify the script for different inputs if needed.

## Usage

### Landing Page
- Choose between uploading a video file (MP4, AVI, MOV) or switching to camera feed.
- Upload a video to process it frame-by-frame with annotations.
- Switch to camera mode for real-time processing.

### Dashboard
- View live counts of people in safe, moderate, and danger zones.
- Monitor for danger alerts, which trigger a visual alarm.
- Dismiss alarms as needed.

### Settings
- Access is password-protected (default: `admin123`).
- Adjust safe and moderate distance thresholds (in pixels).
- Change default input type (upload or camera).

### Video Processing
- Processed videos show bounding boxes around detected people, shoreline points (blue dots), and distance labels.
- Outputs are saved as MP4 files.

## Models and Data

- **YOLOv8 Model**: `yolov8m.pt` - Pre-trained for person detection. Downloaded automatically.
- **U-Net Model**: `unet_beach.onnx` - Custom-trained for beach segmentation (background, sky, water, sand). Downloaded automatically.
- Models are cached after first download for faster subsequent runs.
- If downloads fail, check your internet connection or manually download from the provided Google Drive links in the code.

## Troubleshooting

- **Model Download Issues**: Ensure stable internet. If blocked, download models manually and place in the project directory.
- **Camera Not Working**: Check webcam permissions and ensure WebRTC is supported in your browser.
- **Video Processing Errors**: Verify video format and ensure OpenCV can read the file.
- **Performance**: For better speed, use a GPU-enabled environment (CUDA). Processing may be slow on CPU.
- **Dependencies**: If installation fails, try creating a virtual environment:
  ```
  python -m venv venv
  venv\Scripts\activate  # On Windows
  pip install -r requirements.txt
  ```
- **Port Issues**: If port 8501 is busy, Streamlit will suggest an alternative port.

## Contributing

Feel free to submit issues or pull requests for improvements. Ensure to test changes thoroughly.

## License

This project is for educational and safety purposes. Use responsibly.
