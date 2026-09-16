from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEEPPPCB_ROOT = PROJECT_ROOT / "DeepPCB" / "PCBData"

TRAINVAL_FILE = DEEPPPCB_ROOT / "trainval.txt"
TEST_FILE = DEEPPPCB_ROOT / "test.txt"

CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

IMAGE_SIZE = 640


# ============================================================
# DATA STRUCTURES
# ============================================================

groups = defaultdict(lambda: {
    "trainval": {
        "images": 0,
        "boxes": 0,
        "class_boxes": Counter(),
        "class_images": Counter(),
        "boxes_per_image": [],
    },
    "test": {
        "images": 0,
        "boxes": 0,
        "class_boxes": Counter(),
        "class_images": Counter(),
        "boxes_per_image": [],
    }
})


# ============================================================
# HELPERS
# ============================================================

def resolve_image_path(image_relative):
    """
    DeepPCB txt files contain:
        groupXXXX/PCB/xxxxx.jpg

    Actual image:
        groupXXXX/PCB/xxxxx_test.jpg
    """
    image_relative = image_relative.replace(".jpg", "_test.jpg")
    return DEEPPPCB_ROOT / image_relative


def read_annotation(annotation_path):
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
                print(
                    f"WARNING: Invalid annotation format: "
                    f"{annotation_path}:{line_number}"
                )
                continue

            try:
                x1, y1, x2, y2, class_id = map(int, parts)
            except ValueError:
                print(
                    f"WARNING: Non-integer annotation: "
                    f"{annotation_path}:{line_number}"
                )
                continue

            annotations.append(
                (x1, y1, x2, y2, class_id)
            )

    return annotations


def process_split(split_file, split_name):
    print()
    print("=" * 70)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 70)

    with split_file.open("r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    total = len(lines)

    for index, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) < 2:
            print(f"WARNING: Invalid line {index}: {line}")
            continue

        image_relative = parts[0]
        annotation_relative = parts[1]

        image_path = resolve_image_path(image_relative)
        annotation_path = DEEPPPCB_ROOT / annotation_relative

        # group name = first directory
        group_name = Path(image_relative).parts[0]

        data = groups[group_name][split_name]

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        if not image_path.exists():
            print(f"WARNING: Missing image: {image_path}")
            continue

        data["images"] += 1

        # ----------------------------------------------------
        # Image size
        # ----------------------------------------------------

        try:
            with Image.open(image_path) as img:
                width, height = img.size

                if width != IMAGE_SIZE or height != IMAGE_SIZE:
                    print(
                        f"WARNING: Unexpected image size: "
                        f"{image_path} -> {width}x{height}"
                    )

        except Exception as e:
            print(f"WARNING: Cannot read image: {image_path} ({e})")

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        if not annotation_path.exists():
            print(f"WARNING: Missing annotation: {annotation_path}")
            continue

        annotations = read_annotation(annotation_path)

        data["boxes"] += len(annotations)
        data["boxes_per_image"].append(len(annotations))

        image_classes = set()

        for x1, y1, x2, y2, class_id in annotations:

            if class_id not in CLASS_NAMES:
                print(
                    f"WARNING: Unknown class {class_id}: "
                    f"{annotation_path}"
                )
                continue

            data["class_boxes"][class_id] += 1
            image_classes.add(class_id)

        for class_id in image_classes:
            data["class_images"][class_id] += 1

        if index % 100 == 0 or index == total:
            print(f"Progress: {index}/{total}")


# ============================================================
# PRINT GROUP SUMMARY
# ============================================================

def print_group_summary():

    print()
    print("=" * 90)
    print("GROUP SUMMARY")
    print("=" * 90)

    header = (
        f"{'Group':<15}"
        f"{'TV Img':>8}"
        f"{'TV Box':>8}"
        f"{'TV Avg':>8}"
        f"{'Test Img':>9}"
        f"{'Test Box':>9}"
        f"{'Test Avg':>9}"
    )

    print(header)
    print("-" * 90)

    for group_name in sorted(groups.keys()):

        tv = groups[group_name]["trainval"]
        test = groups[group_name]["test"]

        tv_avg = (
            tv["boxes"] / tv["images"]
            if tv["images"] > 0
            else 0
        )

        test_avg = (
            test["boxes"] / test["images"]
            if test["images"] > 0
            else 0
        )

        print(
            f"{group_name:<15}"
            f"{tv['images']:>8}"
            f"{tv['boxes']:>8}"
            f"{tv_avg:>8.2f}"
            f"{test['images']:>9}"
            f"{test['boxes']:>9}"
            f"{test_avg:>9.2f}"
        )


# ============================================================
# CLASS DISTRIBUTION BY GROUP
# ============================================================

def print_class_distribution():

    print()
    print("=" * 100)
    print("CLASS DISTRIBUTION BY GROUP")
    print("=" * 100)

    for split_name in ["trainval", "test"]:

        print()
        print(f"--- {split_name.upper()} ---")

        print(
            f"{'Group':<15}"
            f"{'open':>10}"
            f"{'short':>10}"
            f"{'mousebite':>12}"
            f"{'spur':>10}"
            f"{'copper':>10}"
            f"{'pin-hole':>10}"
        )

        print("-" * 100)

        for group_name in sorted(groups.keys()):

            data = groups[group_name][split_name]

            values = [
                data["class_boxes"][class_id]
                for class_id in range(1, 7)
            ]

            print(
                f"{group_name:<15}"
                f"{values[0]:>10}"
                f"{values[1]:>10}"
                f"{values[2]:>12}"
                f"{values[3]:>10}"
                f"{values[4]:>10}"
                f"{values[5]:>10}"
            )


# ============================================================
# CLASS PERCENTAGES BY GROUP
# ============================================================

def print_class_percentages():

    print()
    print("=" * 100)
    print("CLASS PERCENTAGE BY GROUP")
    print("=" * 100)

    for split_name in ["trainval", "test"]:

        print()
        print(f"--- {split_name.upper()} ---")

        for group_name in sorted(groups.keys()):

            data = groups[group_name][split_name]

            if data["boxes"] == 0:
                continue

            print()
            print(
                f"{group_name} "
                f"({data['images']} images, "
                f"{data['boxes']} boxes)"
            )

            for class_id in range(1, 7):

                count = data["class_boxes"][class_id]

                percentage = (
                    count / data["boxes"] * 100
                )

                print(
                    f"  {CLASS_NAMES[class_id]:<12}: "
                    f"{count:>4} boxes "
                    f"({percentage:>6.2f}%)"
                )


# ============================================================
# IMAGE CLASS COVERAGE
# ============================================================

def print_class_image_coverage():

    print()
    print("=" * 100)
    print("IMAGE COVERAGE BY CLASS")
    print("=" * 100)

    for split_name in ["trainval", "test"]:

        print()
        print(f"--- {split_name.upper()} ---")

        print(
            f"{'Group':<15}"
            f"{'open':>10}"
            f"{'short':>10}"
            f"{'mousebite':>12}"
            f"{'spur':>10}"
            f"{'copper':>10}"
            f"{'pin-hole':>10}"
        )

        print("-" * 100)

        for group_name in sorted(groups.keys()):

            data = groups[group_name][split_name]

            values = [
                data["class_images"][class_id]
                for class_id in range(1, 7)
            ]

            print(
                f"{group_name:<15}"
                f"{values[0]:>10}"
                f"{values[1]:>10}"
                f"{values[2]:>12}"
                f"{values[3]:>10}"
                f"{values[4]:>10}"
                f"{values[5]:>10}"
            )


# ============================================================
# BOXES PER IMAGE
# ============================================================

def print_boxes_per_image():

    print()
    print("=" * 100)
    print("BOXES PER IMAGE STATISTICS")
    print("=" * 100)

    for split_name in ["trainval", "test"]:

        print()
        print(f"--- {split_name.upper()} ---")

        print(
            f"{'Group':<15}"
            f"{'Min':>8}"
            f"{'Max':>8}"
            f"{'Avg':>10}"
            f"{'Images':>10}"
        )

        print("-" * 60)

        for group_name in sorted(groups.keys()):

            data = groups[group_name][split_name]
            values = data["boxes_per_image"]

            if not values:
                continue

            min_boxes = min(values)
            max_boxes = max(values)
            avg_boxes = sum(values) / len(values)

            print(
                f"{group_name:<15}"
                f"{min_boxes:>8}"
                f"{max_boxes:>8}"
                f"{avg_boxes:>10.2f}"
                f"{len(values):>10}"
            )


# ============================================================
# GROUP CLASS BALANCE
# ============================================================

def print_group_balance():

    print()
    print("=" * 100)
    print("GROUP BALANCE ANALYSIS")
    print("=" * 100)

    print()
    print(
        "Percentage of each defect class inside each TRAINVAL group:"
    )

    print()

    header = (
        f"{'Group':<15}"
        f"{'Images':>8}"
        f"{'Boxes':>8}"
        f"{'open':>9}"
        f"{'short':>9}"
        f"{'mousebite':>11}"
        f"{'spur':>9}"
        f"{'copper':>9}"
        f"{'pin-hole':>9}"
    )

    print(header)
    print("-" * 100)

    for group_name in sorted(groups.keys()):

        data = groups[group_name]["trainval"]

        if data["boxes"] == 0:
            continue

        percentages = []

        for class_id in range(1, 7):
            percentage = (
                data["class_boxes"][class_id]
                / data["boxes"]
                * 100
            )
            percentages.append(percentage)

        print(
            f"{group_name:<15}"
            f"{data['images']:>8}"
            f"{data['boxes']:>8}"
            f"{percentages[0]:>8.1f}%"
            f"{percentages[1]:>8.1f}%"
            f"{percentages[2]:>10.1f}%"
            f"{percentages[3]:>8.1f}%"
            f"{percentages[4]:>8.1f}%"
            f"{percentages[5]:>8.1f}%"
        )


# ============================================================
# GROUP OVERLAP
# ============================================================

def print_group_overlap():

    trainval_groups = {
        group
        for group in groups
        if groups[group]["trainval"]["images"] > 0
    }

    test_groups = {
        group
        for group in groups
        if groups[group]["test"]["images"] > 0
    }

    overlap = sorted(trainval_groups & test_groups)

    trainval_only = sorted(trainval_groups - test_groups)
    test_only = sorted(test_groups - trainval_groups)

    print()
    print("=" * 80)
    print("GROUP OVERLAP")
    print("=" * 80)

    print()
    print(f"Trainval groups : {len(trainval_groups)}")
    print(f"Test groups     : {len(test_groups)}")
    print(f"Overlap groups  : {len(overlap)}")

    print()
    print("Groups appearing in BOTH trainval and test:")
    print(
        "  " +
        ", ".join(overlap)
        if overlap
        else "  None"
    )

    print()
    print("Trainval-only groups:")
    print(
        "  " +
        ", ".join(trainval_only)
        if trainval_only
        else "  None"
    )

    print()
    print("Test-only groups:")
    print(
        "  " +
        ", ".join(test_only)
        if test_only
        else "  None"
    )


# ============================================================
# SMALL GROUP WARNING
# ============================================================

def print_small_groups():

    print()
    print("=" * 80)
    print("SMALL GROUP CHECK")
    print("=" * 80)

    print()
    print("Trainval groups with fewer than 100 images:")

    found = False

    for group_name in sorted(groups.keys()):

        count = groups[group_name]["trainval"]["images"]

        if 0 < count < 100:
            print(f"  {group_name}: {count} images")
            found = True

    if not found:
        print("  None")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DeepPCB Group Analyzer")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"DeepPCB root : {DEEPPPCB_ROOT}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TRAINVAL_FILE.exists():
        print(f"ERROR: Missing file: {TRAINVAL_FILE}")
        return

    if not TEST_FILE.exists():
        print(f"ERROR: Missing file: {TEST_FILE}")
        return

    # --------------------------------------------------------
    # Process both splits
    # --------------------------------------------------------

    process_split(TRAINVAL_FILE, "trainval")
    process_split(TEST_FILE, "test")

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    print_group_summary()
    print_class_distribution()
    print_class_percentages()
    print_class_image_coverage()
    print_boxes_per_image()
    print_group_balance()
    print_group_overlap()
    print_small_groups()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GROUP ANALYSIS COMPLETED")
    print("=" * 70)

    print()
    print("No files were modified.")
    print("Use this report to decide the train/val split.")


if __name__ == "__main__":
    main()