from pathlib import Path
import random
import shutil

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "visualized" / "dataset"

RANDOM_SEED = 42

# Số ảnh muốn visualize cho mỗi split
NUM_IMAGES = {
    "train": 10,
    "val": 10,
    "test": 10,
}

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_font(size=16):
    """
    Try to load a common font.
    If unavailable, use PIL default font.
    """
    font_candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
    ]

    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default()


def yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO normalized xywh to pixel xyxy.
    """

    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Clamp to image boundary
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))

    return x1, y1, x2, y2


def read_yolo_label(label_path):
    """
    Read YOLO label file.

    Format:
        class_id x_center y_center width height
    """

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"[WARNING] Invalid label format: "
                    f"{label_path} line {line_number}"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(
                    f"[WARNING] Cannot parse label: "
                    f"{label_path} line {line_number}"
                )
                continue

            boxes.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    return boxes


def draw_boxes(image, boxes):
    """
    Draw ground-truth bounding boxes on image.
    """

    draw = ImageDraw.Draw(image)

    font = get_font(16)

    for box in boxes:

        class_id, xc, yc, w, h = box

        x1, y1, x2, y2 = yolo_to_xyxy(
            xc,
            yc,
            w,
            h,
            image.width,
            image.height,
        )

        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"class_{class_id}"

        label = f"{class_id}: {class_name}"

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=2,
        )

        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        bbox = draw.textbbox(
            (x1, y1),
            label,
            font=font,
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        label_y1 = max(0, y1 - text_height - 4)
        label_y2 = label_y1 + text_height + 4

        draw.rectangle(
            [
                x1,
                label_y1,
                x1 + text_width + 6,
                label_y2,
            ],
            fill="red",
        )

        draw.text(
            (x1 + 3, label_y1 + 2),
            label,
            fill="white",
            font=font,
        )

    return image


def get_image_files(image_dir):
    """
    Return all supported image files.
    """

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]


# ============================================================
# VISUALIZE ONE SPLIT
# ============================================================

def visualize_split(split):
    """
    Randomly select images from one split and visualize them.
    """

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    output_dir = OUTPUT_DIR / split

    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        return

    if not label_dir.exists():
        print(f"[ERROR] Label directory not found: {label_dir}")
        return

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = get_image_files(image_dir)

    if not image_files:
        print(f"[ERROR] No images found in {image_dir}")
        return

    # --------------------------------------------------------
    # Random sampling
    # --------------------------------------------------------

    num_images = min(NUM_IMAGES[split], len(image_files))

    selected_images = random.sample(
        image_files,
        num_images,
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_boxes = 0

    class_counts = {
        class_id: 0
        for class_id in range(len(CLASS_NAMES))
    }

    # --------------------------------------------------------
    # Process selected images
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"VISUALIZING: {split.upper()}")
    print("=" * 60)

    for index, image_path in enumerate(selected_images, start=1):

        label_path = label_dir / f"{image_path.stem}.txt"

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Cannot open {image_path}: {e}")
            continue

        boxes = read_yolo_label(label_path)

        total_boxes += len(boxes)

        for box in boxes:
            class_id = box[0]

            if class_id in class_counts:
                class_counts[class_id] += 1

        # Draw boxes
        image = draw_boxes(
            image,
            boxes,
        )

        # Save
        output_path = output_dir / image_path.name

        image.save(
            output_path,
            quality=95,
        )

        print(
            f"[{index:02d}/{num_images}] "
            f"{image_path.name} | "
            f"{len(boxes)} boxes"
        )

    # --------------------------------------------------------
    # Statistics output
    # --------------------------------------------------------

    print()
    print(f"Images visualized : {num_images}")
    print(f"Boxes visualized  : {total_boxes}")

    print("Class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id}: {class_name:<10} "
            f"{class_counts[class_id]}"
        )

    print(f"Output: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DeepPCB YOLO DATASET VISUALIZATION")
    print("=" * 60)

    # Check dataset
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found:")
        print(DATASET_DIR)
        return

    print(f"Dataset : {DATASET_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Seed    : {RANDOM_SEED}")

    # Deterministic random selection
    random.seed(RANDOM_SEED)

    # Visualize each split
    for split in ["train", "val", "test"]:
        visualize_split(split)

    # Final message
    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print()
    print("Check the following folders:")
    print(f"  {OUTPUT_DIR / 'train'}")
    print(f"  {OUTPUT_DIR / 'val'}")
    print(f"  {OUTPUT_DIR / 'test'}")


if __name__ == "__main__":
    main()