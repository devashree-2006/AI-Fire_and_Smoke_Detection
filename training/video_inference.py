import cv2
from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = Path("runs/detect/runs/smoke_fire_yolov8n/weights/best.pt")
VIDEO_PATH = Path("dataset/raw_videos/Video Project 8.mp4")
OUTPUT_PATH = Path("outputs/Video_Project_8_detected.mp4")

# Create output folder if needed
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
print("Loading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))

# -----------------------------
# Open input video
# -----------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {VIDEO_PATH.name}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")

# -----------------------------
# Create output video
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

# -----------------------------
# Process video
# -----------------------------
frame_number = 0
fire_count = 0
smoke_count = 0

print("\nStarting detection...\n")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO inference
    results = model.predict(
        source=frame,
        conf=0.25,
        device=0,
        verbose=False
    )

    result = results[0]

    # Draw detections
    annotated_frame = result.plot()

    # Count detections
    if result.boxes is not None:
        for cls in result.boxes.cls:
            class_id = int(cls.item())

            if class_id == 0:
                smoke_count += 1
            elif class_id == 1:
                fire_count += 1

    # Write processed frame
    out.write(annotated_frame)

    # Progress
    if frame_number % 30 == 0:
        progress = (frame_number / total_frames) * 100
        print(f"Progress: {progress:.1f}%")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

print("\nDetection completed!")
print(f"Fire detections: {fire_count}")
print(f"Smoke detections: {smoke_count}")
print(f"\nOutput saved to:")
print(OUTPUT_PATH)