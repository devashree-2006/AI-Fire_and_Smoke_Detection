import cv2
from pathlib import Path

RAW_DIR = Path("dataset/raw_videos")
OUTPUT_DIR = Path("dataset/frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SECONDS = 2.5

for video_path in RAW_DIR.glob("*.mp4"):
    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("Could not read FPS.")
        continue

    duration = total_frames / fps
    interval_frames = int(fps * INTERVAL_SECONDS)

    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.1f} seconds")

    frame_number = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % interval_frames == 0:
            filename = (
                f"{video_path.stem}_frame_{saved_count:04d}.jpg"
            )

            output_path = OUTPUT_DIR / filename
            cv2.imwrite(str(output_path), frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    print(f"Saved: {saved_count} frames")

print("\nFrame extraction complete.")