import cv2
from pathlib import Path

IMAGE_DIR = Path("dataset/images/val")
LABEL_DIR = Path("dataset/labels/val")
OUTPUT_DIR = Path("outputs/annotation_check")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}

for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        lines = label_path.read_text().splitlines()

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # Convert YOLO normalized coordinates to pixels
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            # Keep coordinates inside the image
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            class_name = CLASS_NAMES.get(class_id, "unknown")

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                class_name,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

print("Annotation check images created.")
print(f"Location: {OUTPUT_DIR}")