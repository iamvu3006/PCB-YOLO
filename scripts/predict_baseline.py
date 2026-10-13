from pathlib import Path
import torch
from ultralytics import YOLO

# Xác định đường dẫn gốc của project độc lập với vị trí thực thi terminal
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo11n_baseline" / "weights" / "best.pt"
SOURCE_PATH = PROJECT_ROOT / "dataset" / "images" / "test"
OUTPUT_PROJECT = PROJECT_ROOT / "runs"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {MODEL_PATH}")

if not SOURCE_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh nguồn tại: {SOURCE_PATH}")

device = 0 if torch.cuda.is_available() else "cpu"

print(f"Loading model  : {MODEL_PATH}")
print(f"Predict source : {SOURCE_PATH}")
print(f"Device         : {device}")

model = YOLO(str(MODEL_PATH))

model.predict(
    source=str(SOURCE_PATH),
    imgsz=640,
    conf=0.25,        # ngưỡng thường dùng để xem trực quan
    iou=0.7,
    device=device,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(OUTPUT_PROJECT),
    name="yolo11n_baseline_predict",
    exist_ok=True,
)