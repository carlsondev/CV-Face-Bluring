import cv2
import numpy as np
from typing import Tuple, List, Any


_weights_path = "yolov4_dependencies/yolov4-tiny.weights"
_config_path = "yolov4_dependencies/yolov4-tiny.cfg"

person_labels_idx = 0

# (x, y, w, h)
RectType = Tuple[float, float, float, float]


# Setup YOLO model
def _setup_model() -> Tuple[Any, List[str]]:

    model = cv2.dnn.readNetFromDarknet(_config_path, _weights_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, layer_name


yolo_model, yolo_layer_name = _setup_model()


def get_person_rects(
    image: np.ndarray, NMS_THRESHOLD: float = 0.3, MIN_CONFIDENCE: float = 0.7
) -> List[RectType]:
    """
    Get body rects in the given image

    :param image: The image to detect
    :param NMS_THRESHOLD: Non-Max suppression threshold
    :param MIN_CONFIDENCE: Minimum confidence
    :return: List of body bounding boxes
    """
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    layerOutputs = yolo_model.forward(yolo_layer_name)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID != person_labels_idx:
                continue

            if confidence <= MIN_CONFIDENCE:
                continue

            box = detection[0:4] * np.array([w, h, w, h])
            center_x, center_y, width, height = box.astype("int")

            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    # ensure at least one detection exists

    if len(idxs) <= 0:
        return []

    rects: List[RectType] = []

    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y, w, h = boxes[i]
        rects.append((x, y, w, h))  # Top left and bottom right coordinates
    # return the list of results
    return rects
