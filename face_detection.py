import numpy as np
import cv2

from typing import List, Tuple
from deepface.detectors import FaceDetector

from yolo_person_detection import RectType

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
selected_backend = backends[0]

face_detector = FaceDetector.build_model(selected_backend)

blur_kernel_size = (15, 15)


def should_blur_face(body_img: np.ndarray, face_rect: RectType) -> bool:
    return True


def blur_faces(
    bgr_img: np.ndarray, body_rects: List[RectType], face_rects: List[RectType]
) -> np.ndarray:
    temp_img = bgr_img.copy()
    mask_h, mask_w, _ = temp_img.shape
    mask = np.full((mask_h, mask_w, 1), 0, dtype=np.uint8)

    # Save the face matching with the body
    filtered_body_faces: List[Tuple[RectType, RectType]] = []

    for body in body_rects:
        body_x, body_y, body_w, body_h = body

        for face in face_rects:
            face_x, face_y, face_w, face_h = face

            if (face_x < body_x) or ((face_x + face_w) > (body_x + body_w)):
                # If outside horizontal bounds, ignore
                continue
            if (face_y < body_y) or ((face_y + face_h) > (body_y + body_h)):
                # If outside vertical bounds, ignore
                continue

            # We only care about the first face inside the person, this speeds things up drastically
            filtered_body_faces.append((body, face))
            break

    # Get image area of the body matched with the face rect
    body_img_faces = [
        (temp_img[body[1] : body[1] + body[3], body[0] : body[0] + body[2]], face)
        for (body, face) in filtered_body_faces
    ]

    # Get only the faces that should be blurred
    blurring_face_rects = [
        face for (body_img, face) in body_img_faces if should_blur_face(body_img, face)
    ]

    for face_rect in blurring_face_rects:

        # Convert from float to int
        face_x, face_y, face_w, face_h = tuple(int(x) for x in face_rect)

        # takeout image to blur (2x size, but x and y cannot be less than 0)
        face_img_x = max(face_x - face_w, 0)
        face_img_y = max(face_y - face_h, 0)
        face_img_w = face_x + (2 * face_w)
        face_img_h = face_y + (2 * face_h)

        face_img = temp_img[
            face_img_y : face_img_y + face_img_h,
            face_img_x : face_img_x + face_img_w,
        ]

        # Insert blur back into the image
        temp_img[
            face_img_y : face_img_y + face_img_h,
            face_img_x : face_img_x + face_img_w,
        ] = cv2.blur(face_img, blur_kernel_size)

        # create the circle in the mask and in the temp_img, notice the one in the mask is full
        mask = cv2.circle(
            mask,
            (face_x + int(face_w // 2), face_y + int(face_h // 2)),
            face_h,
            (255),
            -1,
        )

    img_bg = cv2.bitwise_and(bgr_img, bgr_img, mask=cv2.bitwise_not(mask))
    img_fg = cv2.bitwise_and(temp_img, temp_img, mask=mask)
    combined = cv2.add(img_bg, img_fg)

    return combined


def get_face_rects(img: np.ndarray) -> List[RectType]:

    ret_faces = FaceDetector.detect_faces(face_detector, selected_backend, img)
    ret_rects: List[RectType] = []
    for (face_img, face_region) in ret_faces:
        region = [float(x) for x in face_region]
        ret_rects.append((region[0], region[1], region[2], region[3]))

    return ret_rects
