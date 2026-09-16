from pathlib import Path
from collections import Counter
import random
import shutil

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEEPPPCB_ROOT = PROJECT_ROOT / "DeepPCB" / "PCBData"
OUTPUT_ROOT = PROJECT_ROOT / "dataset"

TRAINVAL_FILE = DEEPPPCB_ROOT / "trainval.txt"
TEST_FILE = DEEPPPCB_ROOT / "test.txt"

IMAGE_SIZE = 640

# 80/20 split from 1000 trainval images
TRAIN_RATIO = 0.8

# Reproducible split
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin-hole",
}

# Original DeepPCB class IDs -> YOLO class IDs
DEEPPPCB_TO_YOLO = {
    1: 0,  # open
    2: 1,  # short
    3: 2,  # mousebite
    4: 3,  # spur
    5: 4,  # copper
    6: 5,  # pin-hole
}


# ============================================================
# PATHS
# ============================================================

IMAGE_DIRS = {
    "train": OUTPUT_ROOT / "images" / "train",
    "val": OUTPUT_ROOT / "images" / "val",
    "test": OUTPUT_ROOT / "images" / "test",
}

LABEL_DIRS = {
    "train": OUTPUT_ROOT / "labels" / "train",
    "val": OUTPUT_ROOT / "labels" / "val",
    "test": OUTPUT_ROOT / "labels" / "test",
}


# ============================================================
# HELPERS
# ============================================================

def clean_output_directory():
    """
    Remove previous converted dataset.
    DeepPCB source data is NOT touched.
    """

    if OUTPUT_ROOT.exists():
        print()
        print("Removing previous dataset:")
        print(f"  {OUTPUT_ROOT}")

        shutil.rmtree(OUTPUT_ROOT)

    for directory in IMAGE_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

    for directory in LABEL_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


def resolve_image_path(image_relative):
    """
    trainval.txt/test.txt contains:

        groupXXXX/PCB/xxxxx.jpg

    Actual DeepPCB image:

        groupXXXX/PCB/xxxxx_test.jpg
    """

    actual_relative = image_relative.replace(".jpg", "_test.jpg")

    return DEEPPPCB_ROOT / actual_relative


def read_split_file(split_file):
    """
    Read entries from trainval.txt/test.txt.

    Returns list of:
        {
            "image_relative": ...,
            "annotation_relative": ...,
            "image_path": ...,
            "annotation_path": ...,
            "group": ...
        }
    """

    samples = []

    with split_file.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Invalid line {line_number} in {split_file}: {line}"
                )

            image_relative = parts[0]
            annotation_relative = parts[1]

            image_path = resolve_image_path(image_relative)
            annotation_path = DEEPPPCB_ROOT / annotation_relative

            group = Path(image_relative).parts[0]

            samples.append({
                "image_relative": image_relative,
                "annotation_relative": annotation_relative,
                "image_path": image_path,
                "annotation_path": annotation_path,
                "group": group,
            })

    return samples


def read_annotations(annotation_path):
    """
    Read DeepPCB annotation:

        x1 y1 x2 y2 class_id
    """

    annotations = []

    with annotation_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid annotation format: "
                    f"{annotation_path}:{line_number}"
                )

            x1, y1, x2, y2, class_id = map(int, parts)

            if class_id not in DEEPPPCB_TO_YOLO:
                raise ValueError(
                    f"Unknown DeepPCB class {class_id}: "
                    f"{annotation_path}:{line_number}"
                )

            if not (
                0 <= x1 < x2 <= IMAGE_SIZE
                and 0 <= y1 < y2 <= IMAGE_SIZE
            ):
                raise ValueError(
                    f"Invalid bounding box: "
                    f"{annotation_path}:{line_number}"
                )

            annotations.append(
                (x1, y1, x2, y2, class_id)
            )

    return annotations


def get_sample_classes(sample):
    """
    Return the set of YOLO classes present in an image.
    """

    annotations = read_annotations(sample["annotation_path"])

    return {
        DEEPPPCB_TO_YOLO[class_id]
        for _, _, _, _, class_id in annotations
    }


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_train_val_split(samples):
    """
    Split 1000 trainval images into:

        train = 800
        val   = 200

    Uses a greedy multi-label stratification approach.

    Each image may contain multiple defect classes, so normal
    single-label stratification is not appropriate.
    """

    random.seed(RANDOM_SEED)

    total = len(samples)

    train_count = round(total * TRAIN_RATIO)
    val_count = total - train_count

    print()
    print("=" * 70)
    print("CREATING STRATIFIED TRAIN / VAL SPLIT")
    print("=" * 70)

    print()
    print(f"Total trainval images : {total}")
    print(f"Train images          : {train_count}")
    print(f"Val images            : {val_count}")
    print(f"Random seed           : {RANDOM_SEED}")

    # --------------------------------------------------------
    # Get classes for every image
    # --------------------------------------------------------

    sample_classes = []

    for sample in samples:
        classes = get_sample_classes(sample)
        sample_classes.append(classes)

    # --------------------------------------------------------
    # Global target class box/image counts
    # --------------------------------------------------------

    # Count image-level class occurrence
    global_class_images = Counter()

    for classes in sample_classes:
        for class_id in classes:
            global_class_images[class_id] += 1

    target_train_class_images = {
        class_id: global_class_images[class_id] * TRAIN_RATIO
        for class_id in CLASS_NAMES
    }

    # --------------------------------------------------------
    # Shuffle deterministically
    # --------------------------------------------------------

    indices = list(range(total))
    random.shuffle(indices)

    # Process images containing rarer classes first.
    indices.sort(
        key=lambda i: (
            len(sample_classes[i]),
            sum(
                1 / max(global_class_images[c], 1)
                for c in sample_classes[i]
            ),
        ),
        reverse=True,
    )

    train_indices = []
    val_indices = []

    train_class_images = Counter()

    # --------------------------------------------------------
    # Greedy assignment
    # --------------------------------------------------------

    for index in indices:

        classes = sample_classes[index]

        if len(train_indices) >= train_count:
            val_indices.append(index)
            continue

        if len(val_indices) >= val_count:
            train_indices.append(index)

            for class_id in classes:
                train_class_images[class_id] += 1

            continue

        # Calculate how much this sample helps train
        train_score = sum(
            max(
                target_train_class_images[class_id]
                - train_class_images[class_id],
                0
            )
            for class_id in classes
        )

        # Prefer train while it still needs class coverage.
        # Add a small capacity term so split approaches 80/20.
        train_score += 0.1 * (
            train_count - len(train_indices)
        )

        val_score = sum(
            max(
                global_class_images[class_id]
                - target_train_class_images[class_id]
                - (
                    global_class_images[class_id]
                    - train_class_images[class_id]
                ),
                0
            )
            for class_id in classes
        )

        # If train is currently closer to capacity, favor val.
        if len(train_indices) / train_count > 0.80:
            train_score -= 1.0

        if train_score >= val_score:
            train_indices.append(index)

            for class_id in classes:
                train_class_images[class_id] += 1

        else:
            val_indices.append(index)

    # --------------------------------------------------------
    # Fix exact counts if necessary
    # --------------------------------------------------------

    while len(train_indices) > train_count:

        index = train_indices.pop()
        val_indices.append(index)

    while len(val_indices) > val_count:

        index = val_indices.pop()
        train_indices.append(index)

    train_samples = [
        samples[i]
        for i in train_indices
    ]

    val_samples = [
        samples[i]
        for i in val_indices
    ]

    print()
    print("Split completed:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val  : {len(val_samples)}")

    return train_samples, val_samples


# ============================================================
# YOLO CONVERSION
# ============================================================

def convert_annotation(annotation_path, output_label_path):
    """
    Convert:

        x1 y1 x2 y2 class_id

    to:

        class x_center y_center width height

    normalized to [0, 1].
    """

    annotations = read_annotations(annotation_path)

    with output_label_path.open("w", encoding="utf-8") as f:

        for x1, y1, x2, y2, deep_class_id in annotations:

            yolo_class_id = DEEPPPCB_TO_YOLO[deep_class_id]

            box_width = x2 - x1
            box_height = y2 - y1

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            x_center /= IMAGE_SIZE
            y_center /= IMAGE_SIZE
            box_width /= IMAGE_SIZE
            box_height /= IMAGE_SIZE

            f.write(
                f"{yolo_class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )


def copy_and_convert_sample(sample, split_name):
    """
    Copy image and convert annotation.
    """

    image_path = sample["image_path"]
    annotation_path = sample["annotation_path"]

    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing image: {image_path}"
        )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Missing annotation: {annotation_path}"
        )

    # Verify image
    with Image.open(image_path) as img:

        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Unexpected image size: "
                f"{image_path} -> {img.size}"
            )

    # Use original sample ID without "_test"
    output_stem = image_path.stem.replace("_test", "")

    output_image = (
        IMAGE_DIRS[split_name]
        / f"{output_stem}.jpg"
    )

    output_label = (
        LABEL_DIRS[split_name]
        / f"{output_stem}.txt"
    )

    shutil.copy2(
        image_path,
        output_image
    )

    convert_annotation(
        annotation_path,
        output_label
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_samples(samples, split_name):
    """
    Convert a list of samples.
    """

    print()
    print("=" * 70)
    print(f"CONVERTING: {split_name.upper()}")
    print("=" * 70)

    total = len(samples)

    for index, sample in enumerate(samples, start=1):

        copy_and_convert_sample(
            sample,
            split_name
        )

        if index % 100 == 0 or index == total:
            print(f"Progress: {index}/{total}")


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_split(split_name):
    """
    Verify that every image has a label and vice versa.
    Also calculate class distribution.
    """

    image_files = sorted(
        IMAGE_DIRS[split_name].glob("*.jpg")
    )

    label_files = sorted(
        LABEL_DIRS[split_name].glob("*.txt")
    )

    image_stems = {
        path.stem
        for path in image_files
    }

    label_stems = {
        path.stem
        for path in label_files
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    class_boxes = Counter()
    total_boxes = 0

    for label_file in label_files:

        with label_file.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO label: {label_file}"
                    )

                class_id = int(parts[0])

                if class_id not in CLASS_NAMES:
                    raise ValueError(
                        f"Invalid YOLO class {class_id}: "
                        f"{label_file}"
                    )

                values = list(
                    map(float, parts[1:])
                )

                # YOLO values must be [0, 1]
                if not all(
                    0 <= value <= 1
                    for value in values
                ):
                    raise ValueError(
                        f"Invalid normalized values: "
                        f"{label_file}"
                    )

                class_boxes[class_id] += 1
                total_boxes += 1

    print()
    print(f"Verification: {split_name.upper()}")
    print("-" * 50)

    print(f"Images       : {len(image_files)}")
    print(f"Labels       : {len(label_files)}")
    print(f"Boxes        : {total_boxes}")
    print(f"Missing label: {len(missing_labels)}")
    print(f"Missing image: {len(missing_images)}")

    print()
    print("Class distribution:")

    for class_id in range(6):

        count = class_boxes[class_id]

        percentage = (
            count / total_boxes * 100
            if total_boxes > 0
            else 0
        )

        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:<10} "
            f"{count:>5} boxes "
            f"({percentage:>6.2f}%)"
        )

    if missing_labels:
        raise RuntimeError(
            f"Missing labels in {split_name}: "
            f"{sorted(missing_labels)}"
        )

    if missing_images:
        raise RuntimeError(
            f"Missing images in {split_name}: "
            f"{sorted(missing_images)}"
        )

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "boxes": total_boxes,
        "class_boxes": class_boxes,
    }


# ============================================================
# DATA.YAML
# ============================================================

def create_data_yaml():
    """
    Create Ultralytics YOLO data.yaml.
    """

    yaml_path = OUTPUT_ROOT / "data.yaml"

    content = """path: C:/Users/Admin/VSCode-workspace/PCB-YOLO/dataset

train: images/train
val: images/val
test: images/test

names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: copper
  5: pin-hole
"""

    yaml_path.write_text(
        content,
        encoding="utf-8"
    )

    print()
    print("Created:")
    print(f"  {yaml_path}")


# ============================================================
# FINAL REPORT
# ============================================================

def print_final_report(results):

    print()
    print("=" * 70)
    print("FINAL DATASET REPORT")
    print("=" * 70)

    total_images = 0
    total_boxes = 0

    for split_name in ["train", "val", "test"]:

        result = results[split_name]

        total_images += result["images"]
        total_boxes += result["boxes"]

        print()
        print(
            f"{split_name.upper():<8}: "
            f"{result['images']:>4} images | "
            f"{result['boxes']:>5} boxes"
        )

    print()
    print("-" * 70)

    print(
        f"TOTAL   : "
        f"{total_images:>4} images | "
        f"{total_boxes:>5} boxes"
    )

    print()

    expected = {
        "train": 800,
        "val": 200,
        "test": 500,
    }

    success = True

    for split_name, expected_count in expected.items():

        actual = results[split_name]["images"]

        if actual != expected_count:

            print(
                f"ERROR: {split_name} expected "
                f"{expected_count}, got {actual}"
            )

            success = False

    if total_images != 1500:

        print(
            f"ERROR: Expected 1500 images, "
            f"got {total_images}"
        )

        success = False

    if success:

        print("STATUS: DATASET CONVERSION SUCCESSFUL")
        print()
        print("Train : 800 images")
        print("Val   : 200 images")
        print("Test  : 500 images")
        print("Total : 1500 images")

    else:

        print("STATUS: DATASET VERIFICATION FAILED")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DeepPCB -> YOLO Dataset Converter")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"DeepPCB root : {DEEPPPCB_ROOT}")
    print(f"Output       : {OUTPUT_ROOT}")

    # --------------------------------------------------------
    # Check source files
    # --------------------------------------------------------

    if not TRAINVAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {TRAINVAL_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {TEST_FILE}"
        )

    # --------------------------------------------------------
    # Clean previous converted dataset
    # --------------------------------------------------------

    clean_output_directory()

    # --------------------------------------------------------
    # Read source lists
    # --------------------------------------------------------

    print()
    print("Reading DeepPCB split files...")

    trainval_samples = read_split_file(
        TRAINVAL_FILE
    )

    test_samples = read_split_file(
        TEST_FILE
    )

    print(
        f"  Trainval: {len(trainval_samples)} images"
    )

    print(
        f"  Test    : {len(test_samples)} images"
    )

    if len(trainval_samples) != 1000:
        raise ValueError(
            f"Expected 1000 trainval images, "
            f"got {len(trainval_samples)}"
        )

    if len(test_samples) != 500:
        raise ValueError(
            f"Expected 500 test images, "
            f"got {len(test_samples)}"
        )

    # --------------------------------------------------------
    # Split trainval -> train / val
    # --------------------------------------------------------

    train_samples, val_samples = (
        stratified_train_val_split(
            trainval_samples
        )
    )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    process_samples(
        train_samples,
        "train"
    )

    process_samples(
        val_samples,
        "val"
    )

    # Test remains exactly from test.txt
    process_samples(
        test_samples,
        "test"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    results = {}

    for split_name in ["train", "val", "test"]:

        results[split_name] = verify_split(
            split_name
        )

    # --------------------------------------------------------
    # data.yaml
    # --------------------------------------------------------

    create_data_yaml()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print_final_report(results)


if __name__ == "__main__":
    main()