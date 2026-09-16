from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEEPPPCB_ROOT = PROJECT_ROOT / "DeepPCB" / "PCBData"

TRAINVAL_FILE = DEEPPPCB_ROOT / "trainval.txt"
TEST_FILE = DEEPPPCB_ROOT / "test.txt"

IMAGE_SIZE = 640

CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}


# ============================================================
# ANALYZE ONE SPLIT
# ============================================================

def analyze_split(split_file, split_name):

    print("\n" + "=" * 70)
    print(f"ANALYZING: {split_name}")
    print("=" * 70)

    if not split_file.exists():
        raise FileNotFoundError(
            f"Cannot find split file: {split_file}"
        )

    with open(split_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Images listed: {len(lines)}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    groups = Counter()

    class_boxes = Counter()
    class_images = Counter()

    total_boxes = 0

    invalid_annotations = []
    missing_images = []
    missing_annotations = []

    image_sizes = Counter()

    boxes_per_image = []

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for index, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) != 2:
            print(f"[WARNING] Invalid line: {line}")
            continue

        image_relative = parts[0]
        annotation_relative = parts[1]

        # ----------------------------------------------------
        # Actual DeepPCB test image
        #
        # trainval.txt/test.txt:
        #     xxx.jpg
        #
        # Actual image:
        #     xxx_test.jpg
        # ----------------------------------------------------

        image_relative = image_relative.replace(
            ".jpg",
            "_test.jpg"
        )

        image_path = DEEPPPCB_ROOT / image_relative
        annotation_path = DEEPPPCB_ROOT / annotation_relative

        # ----------------------------------------------------
        # Group
        # ----------------------------------------------------

        group_name = Path(image_relative).parts[0]
        groups[group_name] += 1

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if not image_path.exists():

            missing_images.append(
                str(image_path)
            )

            continue

        # ----------------------------------------------------
        # Check annotation
        # ----------------------------------------------------

        if not annotation_path.exists():

            missing_annotations.append(
                str(annotation_path)
            )

            continue

        # ----------------------------------------------------
        # Image size
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as image:
                image_sizes[image.size] += 1

        except Exception as e:

            print(
                f"[WARNING] Cannot read image: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        image_class_ids = set()
        image_box_count = 0

        try:

            with open(
                annotation_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line_number, line in enumerate(
                    f,
                    start=1
                ):

                    line = line.strip()

                    if not line:
                        continue

                    values = line.split()

                    if len(values) != 5:

                        invalid_annotations.append(
                            (
                                str(annotation_path),
                                line_number,
                                line
                            )
                        )

                        continue

                    try:

                        x1 = int(values[0])
                        y1 = int(values[1])
                        x2 = int(values[2])
                        y2 = int(values[3])
                        class_id = int(values[4])

                    except ValueError:

                        invalid_annotations.append(
                            (
                                str(annotation_path),
                                line_number,
                                line
                            )
                        )

                        continue

                    # ----------------------------------------
                    # Validate class
                    # ----------------------------------------

                    if class_id not in CLASS_NAMES:

                        invalid_annotations.append(
                            (
                                str(annotation_path),
                                line_number,
                                line
                            )
                        )

                        continue

                    # ----------------------------------------
                    # Validate bounding box
                    # ----------------------------------------

                    if x1 < 0 or y1 < 0:
                        invalid_annotations.append(
                            (
                                str(annotation_path),
                                line_number,
                                line
                            )
                        )
                        continue

                    if x2 <= x1 or y2 <= y1:
                        invalid_annotations.append(
                            (
                                str(annotation_path),
                                line_number,
                                line
                            )
                        )
                        continue

                    if x2 > IMAGE_SIZE or y2 > IMAGE_SIZE:

                        invalid_annotations.append(
                            (
                                str(annotation_path),
                                line_number,
                                line
                            )
                        )

                        continue

                    # ----------------------------------------
                    # Valid box
                    # ----------------------------------------

                    class_boxes[class_id] += 1
                    image_class_ids.add(class_id)

                    total_boxes += 1
                    image_box_count += 1

            boxes_per_image.append(image_box_count)

            # Count image once per class
            for class_id in image_class_ids:
                class_images[class_id] += 1

        except Exception:

            invalid_annotations.append(
                (
                    str(annotation_path),
                    0,
                    "Could not read annotation"
                )
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 100 == 0 or index == len(lines):

            print(
                f"Progress: "
                f"{index}/{len(lines)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print(f"RESULTS: {split_name}")
    print("-" * 70)

    print(f"Images                  : {len(lines)}")
    print(f"Groups                  : {len(groups)}")
    print(f"Total bounding boxes    : {total_boxes}")

    if boxes_per_image:

        print(
            f"Average boxes/image    : "
            f"{sum(boxes_per_image) / len(boxes_per_image):.2f}"
        )

        print(
            f"Max boxes/image        : "
            f"{max(boxes_per_image)}"
        )

        print(
            f"Min boxes/image        : "
            f"{min(boxes_per_image)}"
        )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    print("\nGroups:")
    for group, count in sorted(groups.items()):

        print(
            f"  {group:<15} {count:>4} images"
        )

    # --------------------------------------------------------
    # Image sizes
    # --------------------------------------------------------

    print("\nImage sizes:")

    for size, count in image_sizes.items():

        print(
            f"  {size[0]}x{size[1]} : {count} images"
        )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    print("\nDefect distribution:")

    print(
        f"{'ID':<5}"
        f"{'Class':<15}"
        f"{'Boxes':>10}"
        f"{'Images':>10}"
    )

    print("-" * 45)

    for class_id in sorted(CLASS_NAMES):

        print(
            f"{class_id:<5}"
            f"{CLASS_NAMES[class_id]:<15}"
            f"{class_boxes[class_id]:>10}"
            f"{class_images[class_id]:>10}"
        )

    # --------------------------------------------------------
    # Problems
    # --------------------------------------------------------

    print("\nValidation:")

    print(
        f"Missing images          : "
        f"{len(missing_images)}"
    )

    print(
        f"Missing annotations     : "
        f"{len(missing_annotations)}"
    )

    print(
        f"Invalid annotations     : "
        f"{len(invalid_annotations)}"
    )

    # --------------------------------------------------------
    # Print examples
    # --------------------------------------------------------

    if missing_images:

        print("\nFirst missing images:")

        for path in missing_images[:10]:
            print(f"  {path}")

    if missing_annotations:

        print("\nFirst missing annotations:")

        for path in missing_annotations[:10]:
            print(f"  {path}")

    if invalid_annotations:

        print("\nFirst invalid annotations:")

        for item in invalid_annotations[:10]:

            path, line_number, line = item

            print(
                f"  {path}:"
                f"{line_number}"
                f" -> {line}"
            )

    return {
        "images": len(lines),
        "groups": groups,
        "class_boxes": class_boxes,
        "class_images": class_images,
        "total_boxes": total_boxes,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "invalid_annotations": invalid_annotations,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DeepPCB Dataset Analyzer")
    print("=" * 70)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"DeepPCB root : {DEEPPPCB_ROOT}")

    # --------------------------------------------------------
    # Analyze trainval
    # --------------------------------------------------------

    trainval_stats = analyze_split(
        TRAINVAL_FILE,
        "TRAINVAL"
    )

    # --------------------------------------------------------
    # Analyze test
    # --------------------------------------------------------

    test_stats = analyze_split(
        TEST_FILE,
        "TEST"
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)

    total_images = (
        trainval_stats["images"]
        + test_stats["images"]
    )

    total_boxes = (
        trainval_stats["total_boxes"]
        + test_stats["total_boxes"]
    )

    total_groups = set(
        trainval_stats["groups"].keys()
    ) | set(
        test_stats["groups"].keys()
    )

    total_missing_images = (
        len(trainval_stats["missing_images"])
        + len(test_stats["missing_images"])
    )

    total_missing_annotations = (
        len(trainval_stats["missing_annotations"])
        + len(test_stats["missing_annotations"])
    )

    total_invalid = (
        len(trainval_stats["invalid_annotations"])
        + len(test_stats["invalid_annotations"])
    )

    print(f"Total images            : {total_images}")
    print(f"Total groups            : {len(total_groups)}")
    print(f"Total bounding boxes    : {total_boxes}")

    print(
        f"Missing images          : "
        f"{total_missing_images}"
    )

    print(
        f"Missing annotations     : "
        f"{total_missing_annotations}"
    )

    print(
        f"Invalid annotations     : "
        f"{total_invalid}"
    )

    # --------------------------------------------------------
    # Combined class distribution
    # --------------------------------------------------------

    combined_boxes = (
        trainval_stats["class_boxes"]
        + test_stats["class_boxes"]
    )

    combined_images = (
        trainval_stats["class_images"]
        + test_stats["class_images"]
    )

    print("\nCombined defect distribution:")

    print(
        f"{'ID':<5}"
        f"{'Class':<15}"
        f"{'Boxes':>10}"
        f"{'Images':>10}"
    )

    print("-" * 45)

    for class_id in sorted(CLASS_NAMES):

        print(
            f"{class_id:<5}"
            f"{CLASS_NAMES[class_id]:<15}"
            f"{combined_boxes[class_id]:>10}"
            f"{combined_images[class_id]:>10}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)

    if (
        total_missing_images == 0
        and total_missing_annotations == 0
        and total_invalid == 0
    ):

        print(
            "STATUS: DATASET PASSED BASIC VALIDATION"
        )

    else:

        print(
            "STATUS: DATASET HAS PROBLEMS"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()