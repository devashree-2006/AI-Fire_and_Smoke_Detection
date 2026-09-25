import streamlit as st
import cv2
import tempfile
from pathlib import Path
from ultralytics import YOLO


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "runs" / "smoke_fire_yolov8n" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="FlameGuard AI",
    page_icon="🔥",
    layout="wide"
)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# -----------------------------
# Title
# -----------------------------
st.title("🔥 FlameGuard AI")
st.subheader("AI-Powered Fire & Smoke Detection")

st.write(
    "Upload a video and let the YOLOv8 model detect fire and smoke "
    "frame-by-frame."
)


# -----------------------------
# Upload Video
# -----------------------------
uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)


# -----------------------------
# Detection
# -----------------------------
if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("🚨 Start Detection"):

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(uploaded_video.read())
            input_path = temp_file.name

        output_path = OUTPUT_DIR / "streamlit_detected.mp4"

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        fire_count = 0
        smoke_count = 0

        progress = st.progress(0)
        status = st.empty()

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame, verbose=False)

            result = results[0]

            annotated_frame = result.plot()

            writer.write(annotated_frame)

            # Count detections
            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_id = int(cls)

                    if class_id == 0:
                        smoke_count += 1

                    elif class_id == 1:
                        fire_count += 1

            frame_number += 1

            if total_frames > 0:
                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            status.text(
                f"Processing frame {frame_number}/{total_frames}"
            )

        cap.release()
        writer.release()

        progress.progress(1.0)
        status.text("Detection completed successfully! ✅")

        # -----------------------------
        # Results
        # -----------------------------
        st.success("Fire and smoke detection completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🔥 Fire Detections",
                fire_count
            )

        with col2:
            st.metric(
                "💨 Smoke Detections",
                smoke_count
            )

        st.subheader("🎥 Detected Video")

        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)

        st.download_button(
            label="⬇️ Download Detected Video",
            data=video_bytes,
            file_name="fire_smoke_detected.mp4",
            mime="video/mp4"
        )