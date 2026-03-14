import cv2
import numpy as np
from PIL import Image


CLASS_COLORS = {
    "stomata": (0, 220, 0),
    "trichome": (255, 180, 0),
    "vein": (255, 0, 255),
    "nothing": (120, 120, 120),
}


def _color_for_class(class_name: str):
    return CLASS_COLORS.get(class_name, (0, 200, 255))


def render_detection_overlay(image_pil: Image.Image, instances: list[dict], alpha: float = 0.35):
    image = np.array(image_pil).copy()
    overlay = image.copy()

    for instance in instances:
        mask = instance["mask"]
        class_name = instance["class_name"]
        confidence = instance["confidence"]
        x1, y1, x2, y2 = [int(v) for v in instance["bbox_xyxy"]]

        color = _color_for_class(class_name)
        overlay[mask] = color

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(image, contours, -1, color, 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return Image.fromarray(blended)


def render_countable_area_overlay(image_pil: Image.Image, countable_mask: np.ndarray, alpha: float = 0.28):
    image = np.array(image_pil).copy()
    overlay = image.copy()
    highlight = np.array([90, 210, 255], dtype=np.uint8)
    overlay[countable_mask] = highlight
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return Image.fromarray(blended)
