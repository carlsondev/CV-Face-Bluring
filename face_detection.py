import numpy as np
import cv2

from typing import List, Tuple

from yolo_person_detection import RectType

try:
    from retinaface import RetinaFace
except ImportError:
    pass

blur_kernel_size = (51, 51)
g_sigma = 15

dilate_kernel = np.ones((31, 31), np.uint8)
min_red_percent = 10

small_h = 20
large_h = 160
min_s = 200
min_v = 50


ssd_face_detector = cv2.dnn.readNetFromCaffe(
    "./ssd_weights/deploy_lowres.prototxt",
    "./ssd_weights/res10_300x300_ssd_iter_140000.caffemodel",
)


def should_blur_face(img_dilation: np.ndarray) -> bool:
    """
    Returns whether a specific face on a specific body should be blurred

    :param img_dilation: The cropped image of the identified body (threshed and dilated)
    :type img_dilation: image
    :param face_rect: The dimensions of the face in relation to the cropped image
    :type face_rect: RectType aka (x, y, w, h)
    :return: Whether the face should be blurred or not
    :rtype: bool
    """

    img_area = img_dilation.shape[0] * img_dilation.shape[1]

    contours, _ = cv2.findContours(
        img_dilation,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if len(contours) == 0:
        return False

    areas_percents = [(cv2.contourArea(c) / img_area) * 100 for c in contours]
    for p in areas_percents:
        print(f"Red Percentage: {p:.2f}%")
    areas_percents = [p for p in areas_percents if p >= min_red_percent]

    if len(areas_percents) == 0:
        return False

    print(f"Filtered Red Area Count: {len(areas_percents)}")

    return True


def blur_face(
    img: np.ndarray, mask: np.ndarray, face_rect: RectType
) -> Tuple[np.ndarray, np.ndarray]:

    x, y, w, h = tuple(int(x) for x in face_rect)

    # takeout image to blur (2x size, but x and y cannot be less than 0)
    face_img_x = max(x - w, 0)
    face_img_y = max(y - h, 0)
    face_img_w = x + (2 * w)
    face_img_h = y + (2 * h)

    face_img = img[
        face_img_y : face_img_y + face_img_h,
        face_img_x : face_img_x + face_img_w,
    ]

    # Insert blur back into the image
    img[
        face_img_y : face_img_y + face_img_h,
        face_img_x : face_img_x + face_img_w,
    ] = cv2.blur(
        cv2.GaussianBlur(face_img, blur_kernel_size, g_sigma), blur_kernel_size
    )

    # create the circle in the mask and in the temp_img, notice the one in the mask is full
    return img, cv2.circle(
        mask,
        (x + int(w // 2), y + int(h // 2)),
        h // 2,
        (255),
        -1,
    )


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

            face_cx = face_x + (face_w // 2)
            face_cy = face_y + (face_h // 2)

            if (face_cx < body_x) or (face_cx > (body_x + body_w)):
                # If outside horizontal bounds, ignore
                continue
            if (face_cy < body_y) or (face_cy > (body_y + body_h)):
                # If outside vertical bounds, ignore
                continue

            # We only care about the first face inside the person, this speeds things up drastically
            filtered_body_faces.append((body, face))
            break

    hsv_img = cv2.cvtColor(bgr_img.copy(), cv2.COLOR_BGR2HSV)

    threshed_min = cv2.inRange(hsv_img, (0, min_s, min_v), (small_h, 255, 255))
    threshed_max = cv2.inRange(hsv_img, (large_h, min_s, min_v), (180, 255, 255))

    threshed = np.bitwise_or(threshed_min, threshed_max)

    threshed = cv2.dilate(threshed, dilate_kernel, iterations=1)

    for (body, face) in filtered_body_faces:
        # Get image area of the body matched with the face rect
        body_x, body_y, body_w, body_h = body
        face_x, face_y, face_w, face_h = face

        body_x = max(body_x, 0)
        body_y = max(body_y, 0)
        face_x = max(face_x, 0)
        face_y = max(face_y, 0)

        body_img = threshed[body_y : body_y + body_h, body_x : body_x + body_w]
        # Ignore empty image
        if 0 in body_img.shape:
            continue
        if sum(sum(body_img)) == 0:
            continue

        # Get only the faces that should be blurred
        if not should_blur_face(body_img):
            continue

        img, mask = blur_face(temp_img, mask, (face_x, face_y, face_w, face_h))

    img_bg = cv2.bitwise_and(bgr_img, bgr_img, mask=cv2.bitwise_not(mask))
    img_fg = cv2.bitwise_and(temp_img, temp_img, mask=mask)
    combined = cv2.add(img_bg, img_fg)

    return combined


def detect_retina_faces(img: np.ndarray) -> List[RectType]:

    obj = RetinaFace.detect_faces(img)

    if type(obj) != dict:
        return []

    rect_list: List[RectType] = []

    for key in obj:
        identity = obj[key]
        facial_area = identity["facial_area"]

        x = facial_area[0]
        y = facial_area[1]
        w = facial_area[2] - x
        h = facial_area[3] - y

        confidence = identity["score"]

        rect_list.append((x, y, w, h))

    return rect_list


def detect_ssd_faces(img: np.ndarray, confidence: float) -> List[RectType]:

    size = img.shape
    transformed_size = (300, 300)

    img = cv2.resize(img, transformed_size)

    aspect_x = size[1] / transformed_size[1]
    aspect_y = size[0] / transformed_size[0]

    blob = cv2.dnn.blobFromImage(image=img)

    ssd_face_detector.setInput(blob)
    ssd_rets = ssd_face_detector.forward()

    try:
        detections = ssd_rets[0][0]
        detection_count = ssd_rets.shape[2]
    except IndexError:
        return []

    ret_rects: List[RectType] = []

    for i in range(detection_count):
        obj_data: np.ndarray = detections[i]

        # If object is not a face, ignore
        if int(obj_data[1]) != 1:
            continue

        # If confidence is less than the given value, ignore
        if obj_data[2] < confidence:
            continue

        left, top, right, bottom = tuple(obj_data[3:7] * 300)

        rect = (
            int(left * aspect_x),
            int(top * aspect_y),
            int(right * aspect_x) - int(left * aspect_x),
            int(bottom * aspect_y) - int(top * aspect_y),
        )

        ret_rects.append(rect)

    return ret_rects


def segment_image(
    img_shape: Tuple[float, float], segment_count: int
) -> Tuple[int, List[RectType]]:

    img_w, img_h = img_shape
    # Generate image rects
    img_rects: List[RectType] = []

    x_seg_w = img_w // segment_count
    y_seg_h = img_h // segment_count

    for x_seg_idx in range(segment_count):
        curr_seg_x = x_seg_idx * x_seg_w
        current_seg_w = (img_w - curr_seg_x) - (
            (segment_count - x_seg_idx - 1) * x_seg_w
        )

        for y_seg_idx in range(segment_count):
            curr_seg_y = y_seg_idx * y_seg_h
            current_seg_h = (img_h - curr_seg_y) - (
                (segment_count - y_seg_idx - 1) * y_seg_h
            )
            img_rects.append((curr_seg_x, curr_seg_y, current_seg_w, current_seg_h))

    return segment_count, img_rects


seg_img_rects: List[Tuple[int, List[RectType]]] = []


def get_full_ssd_face_rects(img: np.ndarray) -> List[RectType]:

    global seg_img_rects
    if len(seg_img_rects) == 0:
        img_h, img_w, _ = img.shape
        img_shape = (img_w, img_h)
        seg_img_rects = [
            segment_image(img_shape, 2),
            segment_image(img_shape, 3),
        ]

    ret_rects: List[RectType] = detect_ssd_faces(img, 0.8)

    for (seg_count, img_rects) in seg_img_rects:
        for (x, y, w, h) in img_rects:
            img_section = img[y : y + h, x : x + w]
            faces = detect_ssd_faces(img_section, 0.9)
            for (face_x, face_y, face_w, face_h) in faces:

                # Transform back to full space
                wh_mult = seg_count / 2
                xy_delta = (wh_mult - 1) / 2
                ret_rects.append(
                    (
                        face_x + x - (face_w * xy_delta),
                        face_y + y - (face_h * xy_delta),
                        face_w * wh_mult,
                        face_h * wh_mult,
                    )
                )

    return ret_rects
