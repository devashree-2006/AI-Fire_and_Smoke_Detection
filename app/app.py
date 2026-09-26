import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "smoke_fire_yolov8n"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FlameGuard AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       MAIN BACKGROUND
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(255, 85, 0, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(255, 170, 0, 0.08),
                transparent 25%
            ),
            #090b10;

        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* =========================
       HIDE SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stDecoration"] {
        display: none;
    }


    /* =========================
       HEADINGS
       ========================= */

    h1,
    h2,
    h3 {
        color: #ffffff !important;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        font-size: 1.7rem !important;
        font-weight: 750 !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }


    /* =========================
       TEXT
       ========================= */

    p,
    label,
    .stMarkdown {
        color: #c8ccd4;
    }


    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.025);
        border: 1px dashed rgba(255,110,30,0.45);
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(255,105,20,0.55);

        background:
            linear-gradient(
                135deg,
                #ff5a16,
                #ff8a00
            );

        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #ffb066;

        box-shadow:
            0 8px 25px
            rgba(255,90,22,0.20);
    }


    /* =========================
       DOWNLOAD BUTTON
       ========================= */

    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;

        background: #171b23;

        color: white;

        border: 1px solid #343945;

        font-weight: 650;
    }

    .stDownloadButton > button:hover {
        border-color: #ff7225;
        color: #ff9b5d;
    }


    /* =========================
       METRICS
       ========================= */

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius: 16px;

        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #aeb4bf !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* =========================
       SELECT BOX
       ========================= */

    div[data-baseweb="select"] > div {
        background-color: #151820;

        border-color: #343945;

        border-radius: 10px;
    }


    /* =========================
       DATAFRAME
       ========================= */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }


    /* =========================
       ALERTS
       ========================= */

    .stAlert {
        border-radius: 12px;
    }


    /* =========================
       DIVIDER
       ========================= */

    hr {
        border-color:
            rgba(255,255,255,0.08);
    }


    /* =========================
       EXPANDER
       ========================= */

    [data-testid="stExpander"] {
        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("🔥 FlameGuard AI")

st.markdown(
    """
    ### AI-powered Fire & Smoke Detection

    Upload a video and let the trained **YOLOv8 model**
    analyze it frame-by-frame for **fire** and **smoke**.
    """
)

st.divider()


# ============================================================
# MODEL STATUS
# ============================================================

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    if model is not None:
        st.success("● Model loaded")
    else:
        st.error("● Model not found")


with status_col2:

    st.info("YOLOv8 Nano")


with status_col3:

    st.info("Classes: Fire + Smoke")


# ============================================================
# STOP IF MODEL IS MISSING
# ============================================================

if model is None:

    st.error(
        "The trained model could not be found.\n\n"
        f"Expected location:\n`{MODEL_PATH}`"
    )

    st.stop()


st.divider()


# ============================================================
# DETECTION SETTINGS
# ============================================================

st.subheader("⚙️ Detection Settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)


with settings_col1:

    confidence = st.slider(
        "Confidence Threshold",

        min_value=0.10,
        max_value=0.95,

        value=0.40,
        step=0.05,

        help=(
            "Higher values reduce "
            "low-confidence detections."
        ),
    )


with settings_col2:

    iou_threshold = st.slider(
        "IoU Threshold",

        min_value=0.10,
        max_value=0.90,

        value=0.45,
        step=0.05,

        help=(
            "Controls overlapping "
            "bounding boxes."
        ),
    )


with settings_col3:

    detection_mode = st.selectbox(
        "Detection Mode",

        [
            "Fire + Smoke",
            "Fire Only",
            "Smoke Only",
        ],
    )


st.divider()


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.subheader("🎥 Upload Video")

uploaded_video = st.file_uploader(
    "Choose a video file",

    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],

    help=(
        "Upload a fire or smoke video "
        "for analysis."
    ),
)


# ============================================================
# PROCESS VIDEO FUNCTION
# ============================================================

def process_video(
    input_path,
    output_path,
    confidence_threshold,
    iou_threshold,
    mode,
):

    # --------------------------------------------------------
    # OPEN INPUT VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        return None


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),

        fourcc,

        fps,

        (
            width,
            height,
        ),
    )


    if not writer.isOpened():

        cap.release()

        return None


    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    fire_count = 0

    smoke_count = 0

    total_detections = 0

    processed_frames = 0


    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    start_time = time.time()


    # --------------------------------------------------------
    # STREAMLIT UI
    # --------------------------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()

    preview_placeholder = st.empty()


    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()


        if not success:
            break


        processed_frames += 1


        # ----------------------------------------------------
        # YOLO PREDICTION
        # ----------------------------------------------------

        results = model.predict(

            source=frame,

            conf=confidence_threshold,

            iou=iou_threshold,

            verbose=False,
        )


        result = results[0]

        boxes = result.boxes


        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        if boxes is not None:

            for box in boxes:

                cls_id = int(
                    box.cls[0].item()
                )

                conf = float(
                    box.conf[0].item()
                )


                # --------------------------------------------
                # CLASS
                #
                # 0 = smoke
                # 1 = fire
                # --------------------------------------------

                if cls_id == 0:

                    class_name = "SMOKE"

                elif cls_id == 1:

                    class_name = "FIRE"

                else:

                    continue


                # --------------------------------------------
                # DETECTION MODE
                # --------------------------------------------

                if (
                    mode == "Fire Only"
                    and class_name != "FIRE"
                ):

                    continue


                if (
                    mode == "Smoke Only"
                    and class_name != "SMOKE"
                ):

                    continue


                # --------------------------------------------
                # COUNTS
                # --------------------------------------------

                if class_name == "FIRE":

                    fire_count += 1

                elif class_name == "SMOKE":

                    smoke_count += 1


                total_detections += 1


                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                # Orange box
                box_color = (
                    0,
                    140,
                    255,
                )


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    box_color,

                    3,
                )


                # --------------------------------------------
                # LABEL
                # --------------------------------------------

                label = (
                    f"{class_name} "
                    f"{conf:.2f}"
                )


                (
                    text_width,
                    text_height,
                ), baseline = cv2.getTextSize(

                    label,

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    2,
                )


                label_y = max(
                    y1 - 10,
                    text_height + 10,
                )


                # Label background

                cv2.rectangle(

                    frame,

                    (
                        x1,

                        label_y
                        - text_height
                        - baseline
                        - 5,
                    ),

                    (
                        x1
                        + text_width
                        + 10,

                        label_y + 5,
                    ),

                    box_color,

                    -1,
                )


                # Label text

                cv2.putText(

                    frame,

                    label,

                    (
                        x1 + 5,
                        label_y,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (
                        255,
                        255,
                        255,
                    ),

                    2,

                    cv2.LINE_AA,
                )


        # ----------------------------------------------------
        # SAVE PROCESSED FRAME
        # ----------------------------------------------------

        writer.write(frame)


        # ----------------------------------------------------
        # LIVE PREVIEW
        # ----------------------------------------------------

        if (
            processed_frames == 1
            or processed_frames % 10 == 0
        ):

            preview_frame = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB,
            )


            preview_placeholder.image(

                preview_frame,

                caption=(
                    "Live detection preview"
                ),

                use_container_width=True,
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if total_frames > 0:

            progress = (
                processed_frames
                / total_frames
            )


            progress = min(
                progress,
                1.0,
            )


            progress_bar.progress(
                progress
            )


            status_text.write(

                f"Processing frame "
                f"{processed_frames:,} "
                f"of "
                f"{total_frames:,} "
                f"("
                f"{progress * 100:.1f}%"
                f")"
            )


    # --------------------------------------------------------
    # RELEASE VIDEO
    # --------------------------------------------------------

    cap.release()

    writer.release()


    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    processing_time = (
        time.time()
        - start_time
    )


    progress_bar.progress(1.0)


    status_text.success(

        "Processing completed in "
        f"{processing_time:.1f} seconds."
    )

    st.toast(
    "🔥 Detection results are ready!",
    icon="✅"
    )
    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "fire": fire_count,

        "smoke": smoke_count,

        "total": total_detections,

        "frames": processed_frames,

        "fps": fps,

        "width": width,

        "height": height,

        "processing_time": (
            processing_time
        ),
    }


# ============================================================
# PROCESS BUTTON
# ============================================================

if uploaded_video is not None:

    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    st.success(
        f"Video selected: "
        f"**{uploaded_video.name}**"
    )


    file_size_mb = (
        uploaded_video.size
        / (1024 * 1024)
    )


    info_col1, info_col2, info_col3 = (
        st.columns(3)
    )


    with info_col1:

        st.metric(
            "File Size",
            f"{file_size_mb:.2f} MB",
        )


    with info_col2:

        file_format = (
            Path(
                uploaded_video.name
            )
            .suffix
            .upper()
            .replace(
                ".",
                "",
            )
        )

        st.metric(
            "Format",
            file_format,
        )


    with info_col3:

        st.metric(
            "Model",
            "YOLOv8",
        )


    st.write("")


    # --------------------------------------------------------
    # PROCESS BUTTON
    # --------------------------------------------------------

    process_button = st.button(

        "🔥 Start Fire & Smoke Detection",

        type="primary",

        use_container_width=True,
    )


    if process_button:

        # ----------------------------------------------------
        # TEMPORARY INPUT FILE
        # ----------------------------------------------------

        suffix = Path(
            uploaded_video.name
        ).suffix


        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=suffix,

        ) as temp_file:

            temp_file.write(
                uploaded_video.read()
            )

            input_path = Path(
                temp_file.name
            )


        # ----------------------------------------------------
        # OUTPUT FILE
        # ----------------------------------------------------

        original_name = Path(
            uploaded_video.name
        ).stem


        output_path = (
            OUTPUT_DIR
            / (
                f"{original_name}"
                "_FlameGuard_detected.mp4"
            )
        )


        # ----------------------------------------------------
        # DELETE OLD OUTPUT
        # ----------------------------------------------------

        if output_path.exists():

            try:
                if output_path.exists():
                    output_path.unlink()
            except PermissionError:
                pass


        st.divider()


        st.subheader(
            "🔍 Detection in Progress"
        )


        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        stats = process_video(

            input_path=input_path,

            output_path=output_path,

            confidence_threshold=confidence,

            iou_threshold=iou_threshold,

            mode=detection_mode,
        )


        # ----------------------------------------------------
        # REMOVE TEMPORARY INPUT
        # ----------------------------------------------------

        try:

            input_path.unlink()

        except Exception:

            pass


        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        if stats is None:

            st.error(
                "Could not process "
                "the uploaded video."
            )

        else:

            st.session_state[
                "last_output"
            ] = str(output_path)


            st.session_state[
                "last_stats"
            ] = stats


# ============================================================
# RESULTS
# ============================================================

if (
    "last_output" in st.session_state
    and "last_stats" in st.session_state
):

    output_path = Path(
        st.session_state[
            "last_output"
        ]
    )


    stats = st.session_state[
        "last_stats"
    ]


    if output_path.exists():

        st.divider()


        st.subheader(
            "📊 Detection Results"
        )


        # ----------------------------------------------------
        # DETECTION ALERT
        # ----------------------------------------------------

        if stats["fire"] > 0:

            st.error(

                "🚨 FIRE DETECTED — "
                "Review the processed video "
                "immediately."
            )


        elif stats["smoke"] > 0:

            st.warning(

                "⚠️ SMOKE DETECTED — "
                "Potential fire-related "
                "activity found."
            )


        else:

            st.success(

                "✅ No fire or smoke "
                "detections were found."
            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )


        with metric1:

            st.metric(

                "🔥 Fire Detections",

                f"{stats['fire']:,}",
            )


        with metric2:

            st.metric(

                "💨 Smoke Detections",

                f"{stats['smoke']:,}",
            )


        with metric3:

            st.metric(

                "Total Detections",

                f"{stats['total']:,}",
            )


        with metric4:

            st.metric(

                "Processing Time",

                f"{stats['processing_time']:.1f}s",
            )


        st.write("")


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.subheader(
            "⬇️ Download Result"
        )


        try:

            with open(
                output_path,
                "rb",
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )


            st.download_button(

                label=(
                    "⬇️ Download "
                    "Processed Video"
                ),

                data=video_bytes,

                file_name=(
                    output_path.name
                ),

                mime="video/mp4",

                use_container_width=True,
            )


        except Exception:

            st.error(
                "Could not prepare "
                "the processed video."
            )


        # ----------------------------------------------------
        # DETECTION REPORT
        # ----------------------------------------------------

        st.write("")


        with st.expander(
            "📋 View Detection Report",
            expanded=True,
        ):

            report_data = {

                "Metric": [

                    "Fire detections",

                    "Smoke detections",

                    "Total detections",

                    "Frames processed",

                    "Video FPS",

                    "Video resolution",

                    "Confidence threshold",

                    "IoU threshold",

                    "Detection mode",

                    "Processing time",
                ],


                "Value": [

                    f"{stats['fire']:,}",

                    f"{stats['smoke']:,}",

                    f"{stats['total']:,}",

                    f"{stats['frames']:,}",

                    f"{stats['fps']:.2f}",

                    (
                        f"{stats['width']} × "
                        f"{stats['height']}"
                    ),

                    f"{confidence:.2f}",

                    f"{iou_threshold:.2f}",

                    detection_mode,

                    (
                        f"{stats['processing_time']:.2f} "
                        "seconds"
                    ),
                ],
            }


            st.dataframe(

                report_data,

                use_container_width=True,

                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


footer_col1, footer_col2 = (
    st.columns(2)
)


with footer_col1:

    st.caption(
        "🔥 FlameGuard AI — "
        "Fire & Smoke Detection"
    )


with footer_col2:

    st.caption(
        "YOLOv8 • OpenCV • Streamlit"
    )