# 🔥 Fire & Smoke Detection using YOLO

A computer vision and machine learning project for detecting **fire and smoke from visual input** using a YOLO-based object detection model.

This project was developed as part of a hackathon and focuses on applying deep learning and computer vision techniques to the problem of early fire and smoke detection.

---

## 📌 Project Overview

Fire incidents can develop rapidly, and detecting them at an early stage can help reduce potential damage and improve response time.

The goal of this project is to develop an automated vision-based detection system that can identify **fire** and **smoke** from images or video input.

Instead of relying only on traditional sensors, the system uses a computer vision model to analyze visual information and detect relevant objects.

The model produces:

- Detected object classes
- Bounding boxes around detected regions
- Confidence scores for each prediction

The project also considers visually confusing examples so that the model can learn to distinguish actual fire and smoke from non-fire visual patterns.

---

# 🎯 Objectives

The main objectives of this project are:

- Detect fire from images and video.
- Detect smoke from images and video.
- Use a YOLO-based object detection approach.
- Identify the location of detected fire or smoke using bounding boxes.
- Provide confidence scores for predictions.
- Reduce false detections caused by visually similar objects or artificial fire-like visuals.
- Build a foundation that can later be extended into a real-time monitoring and alert system.

---

# ✨ Key Features

### 🔥 Fire Detection

The system identifies visible fire regions in an input image or video frame.

### 💨 Smoke Detection

The model detects visible smoke and identifies the corresponding region.

### 🎯 Bounding Box Detection

Detected fire and smoke regions are highlighted using bounding boxes.

### 📊 Confidence Scores

Each detection is associated with a confidence score indicating how strongly the model predicts the detected class.

### 🖼️ Image Detection

The model can be used to analyze individual images for fire and smoke.

### 🎥 Video Detection

The detection pipeline can be extended to process video frames and identify fire or smoke over time.

### 🧠 False-Positive Reduction

The dataset includes visually similar examples to help the model learn the difference between actual fire and misleading visual patterns.

---

# 🧠 Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| YOLO | Object detection |
| PyTorch | Deep learning framework |
| OpenCV | Image and video processing |
| NumPy | Numerical and array operations |
| Computer Vision | Visual detection and analysis |
| Machine Learning | Model training and prediction |

---

# 🏗️ System Architecture

The overall workflow of the project can be represented as:

```text
              Input
                │
                ▼
       Image / Video Frame
                │
                ▼
        Image Preprocessing
                │
                ▼
        YOLO Detection Model
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
      Fire             Smoke
        │                │
        └───────┬────────┘
                ▼
       Bounding Boxes
                │
                ▼
       Confidence Scores
                │
                ▼
        Detection Output
