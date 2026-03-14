import io
from collections import Counter

import cv2
import numpy as np
from PIL import Image


COUNTABLE_OVERLAP_THRESHOLD = 0.5
NON_COUNTABLE_CLASSES = {"nothing", "trichome", "vein"}


def mask_to_bbox(binary_mask: np.ndarray):
    ys, xs = np.where(binary_mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("mask_to_bbox received an empty mask")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def get_ellipse_axes(binary_mask: np.ndarray):
    ys, xs = np.where(binary_mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) < 5:
        return None, None
    (_, _), (minor_axis, major_axis), _ = cv2.fitEllipse(points)
    return float(major_axis), float(minor_axis)


def add_closest_stomata_distance(stomata_metadata: list[dict], um_per_px: float):
    if not stomata_metadata:
        return stomata_metadata

    centroids = np.array([[s["centroid_x"], s["centroid_y"]] for s in stomata_metadata], dtype=float)

    for i, stomata in enumerate(stomata_metadata):
        diff = centroids - centroids[i]
        distances = np.sqrt(np.sum(diff**2, axis=1))
        distances[i] = np.inf
        min_dist_px = distances.min()
        if np.isinf(min_dist_px):
            min_dist_px = None

        stomata["closest_stomata_distance_px"] = float(min_dist_px) if min_dist_px is not None else None
        stomata["closest_stomata_distance_um"] = float(min_dist_px * um_per_px) if min_dist_px is not None else None
        stomata["closest_stomata_distance_mm"] = float(min_dist_px * um_per_px / 1000) if min_dist_px is not None else None

    return stomata_metadata


def density_per_mm2(stomata_count: int, countable_pixels: int, um_per_px: float):
    if not um_per_px or countable_pixels == 0:
        return None
    countable_mm2 = countable_pixels * (um_per_px**2) / 1e6
    return stomata_count / countable_mm2


def resize_binary_mask(binary_mask: np.ndarray, width: int, height: int):
    if binary_mask.shape == (height, width):
        return binary_mask
    return np.array(Image.fromarray(binary_mask).resize((width, height), Image.NEAREST), dtype=bool)


def build_instance(binary_mask: np.ndarray, class_id: int, class_name: str, confidence: float):
    x1, y1, x2, y2 = mask_to_bbox(binary_mask)
    ys, xs = np.where(binary_mask)
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": float(confidence),
        "pixel_area": int(binary_mask.sum()),
        "bbox_xyxy": [x1, y1, x2, y2],
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "mask": binary_mask,
    }


def build_stomata_metadata(binary_mask: np.ndarray, instance_id: int, class_id: int, confidence: float, um_per_px: float):
    x1, y1, x2, y2 = mask_to_bbox(binary_mask)
    ys, xs = np.where(binary_mask)
    ellipse_length_px, _ = get_ellipse_axes(binary_mask)
    pixel_area = int(binary_mask.sum())

    return {
        "instance_id": instance_id,
        "class_id": class_id,
        "class_name": "stomata",
        "confidence": float(confidence),
        "pixel_area": pixel_area,
        "area_um2": float(pixel_area * um_per_px**2),
        "stomata_length_um": ellipse_length_px * um_per_px if ellipse_length_px is not None else None,
        "stomata_length_mm": (ellipse_length_px * um_per_px) / 1000 if ellipse_length_px is not None else None,
        "bbox_xyxy": [x1, y1, x2, y2],
        "width_px": float(x2 - x1),
        "height_px": float(y2 - y1),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
    }


def process_yolo_result(result, um_per_px: float):
    img_h, img_w = result.orig_shape
    total_pixels = int(img_h * img_w)

    if result.masks is None or len(result.masks.data) == 0:
        return {
            "instances": [],
            "stomata": [],
            "class_counts": {},
            "density_info": {
                "total_pixels": total_pixels,
                "non_countable_pixels": 0,
                "countable_pixels": total_pixels,
                "stomata_count": 0,
                "stomatal_density_per_px2": 0.0,
                "um_per_px": um_per_px,
                "stomatal_density_per_mm2": 0.0,
            },
            "non_countable_mask": np.zeros((img_h, img_w), dtype=bool),
            "countable_mask": np.ones((img_h, img_w), dtype=bool),
        }

    masks_float = result.masks.data.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    non_countable_mask = np.zeros((img_h, img_w), dtype=bool)
    all_instances = []
    stomata_candidates = []
    class_counter = Counter()

    for mask_float, conf, cls in zip(masks_float, confidences, class_ids):
        cls_id = int(cls)
        cls_name = result.names[cls_id]
        binary = resize_binary_mask(mask_float > 0.5, img_w, img_h)

        if int(binary.sum()) == 0:
            continue

        class_counter[cls_name] += 1
        instance = build_instance(binary, cls_id, cls_name, float(conf))
        all_instances.append(instance)

        if cls_name in NON_COUNTABLE_CLASSES:
            non_countable_mask |= binary
        elif cls_name == "stomata":
            stomata_candidates.append((binary, cls_id, float(conf)))

    countable_mask = ~non_countable_mask
    countable_pixels = int(countable_mask.sum())

    stomata_metadata = []
    kept_index = 0
    for binary, cls_id, conf in stomata_candidates:
        stomata_pixels = int(binary.sum())
        if stomata_pixels == 0:
            continue

        stomata_in_countable = int((binary & countable_mask).sum())
        overlap_fraction = stomata_in_countable / stomata_pixels

        if overlap_fraction >= COUNTABLE_OVERLAP_THRESHOLD:
            meta = build_stomata_metadata(binary, kept_index, cls_id, conf, um_per_px)
            meta["fraction_in_countable_area"] = float(overlap_fraction)
            stomata_metadata.append(meta)
            kept_index += 1

    stomata_metadata = add_closest_stomata_distance(stomata_metadata, um_per_px)
    stomata_count = len(stomata_metadata)

    density_info = {
        "total_pixels": total_pixels,
        "non_countable_pixels": int(non_countable_mask.sum()),
        "countable_pixels": countable_pixels,
        "stomata_count": stomata_count,
        "stomatal_density_per_px2": float(stomata_count / countable_pixels) if countable_pixels else None,
        "um_per_px": um_per_px,
        "stomatal_density_per_mm2": density_per_mm2(stomata_count, countable_pixels, um_per_px),
    }

    return {
        "instances": all_instances,
        "stomata": stomata_metadata,
        "class_counts": dict(class_counter),
        "density_info": density_info,
        "non_countable_mask": non_countable_mask,
        "countable_mask": countable_mask,
    }


def image_bytes_to_pil(image_bytes: bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
