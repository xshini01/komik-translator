from ultralytics import YOLO
from typing import List

# Cache loaded models to avoid reloading on every image
_model_cache: dict[str, YOLO] = {}


def get_model(model_path: str) -> YOLO:
    """Load and cache a YOLO model by path."""
    if model_path not in _model_cache:
        print(f"[detect_bubbles] Loading YOLO model: {model_path}")
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def detect_bubbles(model_path: str, image_path: str, conf: float = 0.25) -> List[list]:
    """
    Detects speech bubbles in an image using a YOLOv8 model.

    Args:
        model_path (str): Path to the YOLO model file.
        image_path (str): Path to the input image.
        conf (float): Confidence threshold for detection (default: 0.25).

    Returns:
        list: Detected boxes as [x1, y1, x2, y2, score, class_id].
    """
    model = get_model(model_path)
    result = model.predict(image_path, conf=conf, verbose=False)[0]
    return result.boxes.data.tolist()


def clear_model_cache() -> None:
    """Clear the model cache to free memory."""
    _model_cache.clear()
