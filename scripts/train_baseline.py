from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"

MODEL_NAME = "yolo11n.pt"

PROJECT_NAME = PROJECT_ROOT / "runs"

EXPERIMENT_NAME = "yolo11n_baseline"

# Training configuration
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16

DEVICE = 0

WORKERS = 4

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLOv11 BASELINE TRAINING")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATA_YAML}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Device       : {DEVICE}")
    print(f"Seed         : {SEED}")
    print()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    # --------------------------------------------------------
    # Load pretrained YOLOv11
    # --------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS,

        seed=SEED,

        project=str(PROJECT_NAME),
        name=EXPERIMENT_NAME,

        pretrained=True,

        # Save checkpoints
        save=True,
        save_period=10,

        # Validation during training
        val=True,

        # Deterministic experiment
        deterministic=True,

        # Cache disabled to avoid unnecessary RAM usage
        cache=False,

        # Early stopping
        patience=20,

        # Standard augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=0.0,
        translate=0.1,
        scale=0.5,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.0,

        # Close mosaic near the end
        close_mosaic=10,
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Results saved to:")
    print(PROJECT_NAME / EXPERIMENT_NAME)

    best_model = (
        PROJECT_NAME
        / EXPERIMENT_NAME
        / "weights"
        / "best.pt"
    )

    last_model = (
        PROJECT_NAME
        / EXPERIMENT_NAME
        / "weights"
        / "last.pt"
    )

    print()
    print(f"Best model : {best_model}")
    print(f"Last model : {last_model}")

    print()
    print("Training metrics and plots should be available in:")
    print(PROJECT_NAME / EXPERIMENT_NAME)


if __name__ == "__main__":
    main()