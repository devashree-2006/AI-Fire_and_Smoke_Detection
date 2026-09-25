from pathlib import Path
import shutil

FRAMES_DIR = Path("dataset/frames")
LABELS_DIR = Path("dataset/labels/train")

TRAIN_IMAGES = Path("dataset/images/train")
VAL_IMAGES = Path("dataset/images/val")
TRAIN_LABELS = Path("dataset/labels/train")
VAL_LABELS = Path("dataset/labels/val")

TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

# Get frames separately for each source video
videos = {}

for image in sorted(FRAMES_DIR.glob("*.jpg")):
    # Everything before "_frame_" identifies the source video
    source = image.name.split("_frame_")[0]
    videos.setdefault(source, []).append(image)

for source, images in videos.items():
    images.sort()

    # 80% train, 20% validation
    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"\n{source}")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")

    for image in train_images:
        label = LABELS_DIR / f"{image.stem}.txt"

        shutil.copy2(image, TRAIN_IMAGES / image.name)

        if label.exists():
            # Already in train labels, so leave it there
            pass

    for image in val_images:
        label = LABELS_DIR / f"{image.stem}.txt"

        shutil.copy2(image, VAL_IMAGES / image.name)

        if label.exists():
            shutil.move(str(label), str(VAL_LABELS / label.name))

print("\nDataset split complete.")