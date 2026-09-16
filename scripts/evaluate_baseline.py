from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "yolo11n_baseline"
    / "weights"
    / "best.pt"
)

PROJECT_NAME = PROJECT_ROOT / "runs"

EXPERIMENT_NAME = "yolo11n_baseline_test"

IMAGE_SIZE = 640

BATCH_SIZE = 16

DEVICE = 0

WORKERS = 4

CONF_THRESHOLD = 0.001

IOU_THRESHOLD = 0.7

SAVE_PLOTS = True

SAVE_JSON = True

VERBOSE = True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLO11n BASELINE - FINAL TEST EVALUATION")
    print("=" * 60)

    print(f"Model       : {MODEL_PATH}")
    print(f"Dataset     : {DATA_YAML}")
    print(f"Split       : test")
    print(f"Images      : 500")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Device      : {DEVICE}")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("Loading best.pt...")
    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Evaluate on TEST
    # --------------------------------------------------------

    print()
    print("Evaluating on TEST split...")
    print()

    metrics = model.val(
        data=str(DATA_YAML),

        split="test",

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        project=str(PROJECT_NAME),

        name=EXPERIMENT_NAME,

        plots=SAVE_PLOTS,

        save_json=SAVE_JSON,

        verbose=VERBOSE,
    )

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    box_metrics = metrics.box

    precision = box_metrics.mp
    recall = box_metrics.mr
    map50 = box_metrics.map50
    map5095 = box_metrics.map

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"mAP@50          : {map50:.4f}")
    print(f"mAP@50-95       : {map5095:.4f}")

    print()
    print("Percentage:")
    print(f"Precision       : {precision * 100:.2f}%")
    print(f"Recall          : {recall * 100:.2f}%")
    print(f"mAP@50          : {map50 * 100:.2f}%")
    print(f"mAP@50-95       : {map5095 * 100:.2f}%")

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    class_names = model.names

    print()
    print("=" * 60)
    print("PER-CLASS TEST RESULTS")
    print("=" * 60)

    print(
        f"{'Class':<12}"
        f"{'P':>10}"
        f"{'R':>10}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
    )

    print("-" * 54)

    for class_id, class_name in class_names.items():

        p = box_metrics.p[class_id]
        r = box_metrics.r[class_id]
        ap50 = box_metrics.ap50[class_id]
        ap = box_metrics.ap[class_id]

        print(
            f"{class_name:<12}"
            f"{p:>10.4f}"
            f"{r:>10.4f}"
            f"{ap50:>10.4f}"
            f"{ap:>12.4f}"
        )

    # --------------------------------------------------------
    # Output location
    # --------------------------------------------------------

    output_dir = PROJECT_NAME / EXPERIMENT_NAME

    print()
    print("=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)

    print(f"Results saved to:")
    print(output_dir)

    print()
    print("Important files:")

    important_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    ]

    for filename in important_files:

        path = output_dir / filename

        if path.exists():
            print(f"  [OK] {filename}")
        else:
            print(f"  [--] {filename}")


if __name__ == "__main__":
    main()