"""
yolo_model.py — YOLOv8 Model Loader (Singleton)
Loads best.pt once and reuses across the entire system.
"""

import os
from ultralytics import YOLO

_model = None  # Singleton instance


def load_model(model_path: str = None) -> YOLO:
    """
    Load YOLOv8 model once and cache it.
    Falls back to 'best.pt' in the script's directory if no path given.
    """
    global _model

    if _model is not None:
        return _model

    if model_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        # Try common locations
        candidates = [
            os.path.join(base, "best.pt"),
            os.path.join(base, "runs/detect/train2/weights/best.pt"),
            os.path.join(base, "yolov8_plate.pt"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[YOLO] Model file not found. Checked:\n"
            + "\n".join(f"  - {p}" for p in candidates)
            + "\nPlace your best.pt in the project root."
        )

    print(f"[YOLO] Loading model from: {model_path}")
    _model = YOLO(model_path)
    print("[✓] YOLOv8 model loaded successfully")
    return _model


def is_loaded() -> bool:
    return _model is not None
