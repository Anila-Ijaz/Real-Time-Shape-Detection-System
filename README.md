# Real-Time-Shape and Object Detection-System (YOLOv8 + OpenCV)
Real-Time Object &amp; Shape Detection (YOLOv8 + OpenCV) – Developed a computer vision system that integrates deep learning (YOLOv8) with classical contour analysis to detect and classify objects and their geometric shapes in real time from a webcam feed.

**Language**: Python  
**Topics:** Computer Vision, Real-Time Object Detection, OpenCV, YOLOv8

This mini project uses **YOLOv8** (Ultralytics) for real-time object detection from a webcam and then runs a simple **shape detection** (triangle / rectangle / circle) on each detected object using OpenCV contours. The result is displayed live with bounding boxes and labels.

https://github.com/<your-username>/<your-repo-name>

---

## Features
- 🎥 Live webcam inference with YOLOv8 (`yolov8n.pt`)
- 🔲 Bounding boxes with class labels & confidence
- 🔺 Simple shape detection on the cropped object (triangle/rectangle/circle)
- ⌨️ Press **Q** to quit

---

## Requirements
- Python 3.9+ (3.10/3.11 recommended)
- Packages:
  - `ultralytics` (pulls in `torch`)
  - `opencv-python`
  - `numpy`
  - `pillow` (PIL)

If you have a CUDA-capable GPU and the right PyTorch build installed, Ultralytics/YOLO will automatically use it. Otherwise it runs on CPU.

---

## Installation

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
