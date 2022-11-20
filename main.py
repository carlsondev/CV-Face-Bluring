""" Source: https://data-flair.training/blogs/pedestrian-detection-python-opencv/"""

import numpy as np
import cv2
import imutils
import time
from typing import Tuple, List, Any

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

labels_path = "yolov4_dependencies/coco.names"
weights_path = "yolov4_dependencies/yolov4-tiny.weights"
config_path = "yolov4_dependencies/yolov4-tiny.cfg"

person_labels_idx = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)


def setup_model() -> Tuple[Any, List[str]]:
    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, layer_name


model, layer_name = setup_model()


def get_person_rects(
    image: np.ndarray,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

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

    rects = []

    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y, w, h = boxes[i]
        rects.append(((x, y), (x + w, y + h)))  # Top left and bottom right coordinates
    # return the list of results
    return rects


def main():

    video_capture = cv2.VideoCapture(
        "videos/walking_pedestrians1.mp4"
    )  # Open video capture object
    got_image, bgr_img = video_capture.read()  # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        exit(1)

    window_name = "FinalProject"
    cv2.namedWindow(window_name)
    window_is_open = True

    while window_is_open and got_image:
        start_time = time.time()
        bgr_img = imutils.resize(bgr_img, width=700)
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        boxes = get_person_rects(bgr_img)

        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

        if len(faces) > 0:
            print(len(faces))
            for face in faces:
                face_x, face_y, face_w, face_h = face
                # face_bl = (
                #     box_tl[0] + face_x,
                #     box_tl[1] + face_y,
                # )  # Transform smaller coordinate space to real image space
                cv2.circle(
                    bgr_img,
                    (face_x + face_w // 2, face_y + face_h // 2),
                    20,
                    (0, 0, 255),
                    -1,
                )

        for box in boxes:
            box_tl, box_br = box

            # person_img = gray_img[
            #     box_tl[1] : box_br[1],
            #     box_tl[0] : box_br[0],
            # ]

            cv2.rectangle(bgr_img, box_tl, box_br, (0, 255, 0), 2)

        print(f"Frame took {time.time() - start_time:.2f} seconds")

        cv2.imshow(window_name, bgr_img)

        got_image, bgr_img = video_capture.read()

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
