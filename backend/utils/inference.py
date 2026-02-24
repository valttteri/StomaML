import io
import logging
from typing import Dict, List, Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def process_detections(result):
    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, clss):
        cls = int(cls)
        detections.append({
            "class_id": cls,
            "class_name": result.names[cls],
            "confidence": float(conf),
            "bbox": box.tolist(),
        })

    return detections



def run_inference(model, image_bytes: bytes, conf_threshold: float) -> Dict[str, Any]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
        results = model(img_array, conf=conf_threshold)
        result = results[0]
        detections = process_detections(result)
        return {
            "success": True,
            "num_detections": len(detections),
            "detections": detections,
        }

    except Exception as e:
        logger.exception(f"Error during inference: {e}")
        raise


def batch_inference(model, images_data: List[tuple], conf_threshold: float) -> List[Dict[str, Any]]:
    results_out= []

    for index, (filename, image_bytes) in enumerate(images_data):
            res = run_inference(model, image_bytes, conf_threshold)
            res["filename"] = filename
            res["index"] = index
            results_out.append(res)

    #error handling?
    return results_out
