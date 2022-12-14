""" Source: https://data-flair.training/blogs/pedestrian-detection-python-opencv/"""

import numpy as np
import cv2
import time
import os

from typing import List, Tuple, Optional

from yolo_person_detection import get_person_rects, RectType
from face_detection import get_full_ssd_face_rects, blur_faces, detect_retina_faces

use_retina_face_detect = False

try:
    from retinaface import RetinaFace
    import tensorflow as tf

    use_retina_face_detect = False
except ImportError:
    pass


def draw_rects(
    bgr_img: np.ndarray, body_rects: List[RectType], face_rects: List[RectType]
) -> np.ndarray:
    """
    Optional method to draw body bounding rectangles and face circles
    :param bgr_img: Image to draw upon
    :param body_rects: List of body rects
    :param face_rects: List of face rects
    :return: Drawn image
    """
    img = bgr_img.copy()

    for body in body_rects:
        x, y, w, h = body

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for face in face_rects:
        face_x, face_y, face_w, face_h = face
        img = cv2.circle(
            img,
            (int(face_x + face_w // 2), int(face_y + face_h // 2)),
            1,
            (0, 0, 255),
            -1,
        )

    return img


def get_minutes_seconds(seconds: float) -> Tuple[int, int]:
    """
    Convert seconds to minutes and remainder seconds

    :param seconds: Seconds to convert
    :return: (minutes, seconds)
    """
    time_min = int(seconds // 60)
    time_s = int(seconds - (60 * time_min))

    return time_min, time_s


def main(input_video_path: str, output_video_path: str, extension: str = "mp4"):

    if not os.path.exists(input_video_path):
        print("Input file does not exist!")
        return

    if os.path.exists(os.path.join(output_video_path, extension)):
        os.remove(output_video_path)

    video_capture = cv2.VideoCapture(input_video_path)  # Open video capture object

    got_image, bgr_img = video_capture.read()  # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        exit(1)

    # Create video output
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    img_h, img_w, _ = bgr_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(output_video_path, extension), fourcc, fps, (img_w, img_h)
    )

    retina_out: Optional[cv2.VideoWriter] = None

    if use_retina_face_detect:
        retina_out = cv2.VideoWriter(
            os.path.join(output_video_path + "retina", extension),
            fourcc,
            fps,
            (img_w, img_h),
        )

    # Estimate the amount of time it will take to process.
    # ~0.22 s/frame on cuda, 2.5 s/frame on CPU
    start_avg = 0.25
    if use_retina_face_detect and tf.test.is_gpu_available():
        start_avg = 0.25
    est_min, est_s = get_minutes_seconds(start_avg * frame_count)

    print(
        f"File FPS: {fps}, Frame Count: {frame_count}, Est Max Time: {est_min}:{est_s}"
    )

    current_frame_num = 1
    program_start_time = time.time()

    # Used for average. The time added when storing to a list is negligible
    frame_comp_times: List[float] = []

    while got_image:
        start_time = time.time()

        print(f"Currently processing frame {current_frame_num}/{frame_count}")
        body_rects = get_person_rects(bgr_img)

        ssd_face_rects = get_full_ssd_face_rects(bgr_img)
        retina_face_rects = []
        if use_retina_face_detect:
            retina_face_rects = detect_retina_faces(bgr_img)

        print(f"Faces Count: {len(ssd_face_rects)}")

        bgr_img = blur_faces(bgr_img, body_rects, ssd_face_rects)
        retina_face_bgr_img: Optional[np.ndarray] = None
        if use_retina_face_detect:
            retina_face_bgr_img = blur_faces(
                bgr_img.copy(), body_rects, retina_face_rects
            )

        end_time = time.time()

        # Already calculated main computation time, add additional time for non-CV computations
        extra_start = time.time()

        # Compute the average computation time of every frame
        frame_comp_time = end_time - start_time
        frame_comp_times.append(frame_comp_time)

        frame_comp_time_avg = sum(frame_comp_times) / len(frame_comp_times)

        # Predict the total amount of time
        predicted_seconds_remaining = frame_count * frame_comp_time_avg

        pred_min, pred_s = get_minutes_seconds(predicted_seconds_remaining)

        # Get the total time the program has taken
        exec_time = end_time - program_start_time
        exec_time_min, exec_time_s = get_minutes_seconds(exec_time)

        print(
            f"Frame #{current_frame_num} took {frame_comp_time:.2f} seconds. Average of {frame_comp_time_avg:.2f} seconds"
        )
        print(
            f"Program execution time: {exec_time_min}:{exec_time_s:02}/{pred_min}:{pred_s:02}"
        )

        # Add additional exec time.
        frame_comp_times[-1] += time.time() - extra_start
        print("-" * 100)

        cv2.imshow("SSD Face Video", bgr_img)
        if use_retina_face_detect:
            cv2.imshow("Retina Face Video", retina_face_bgr_img)
            retina_out.write(retina_face_bgr_img)
        cv2.waitKey(1)

        out.write(bgr_img)
        current_frame_num += 1
        got_image, bgr_img = video_capture.read()

    print("Finished!")
    video_capture.release()
    out.release()
    if use_retina_face_detect:
        retina_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("videos/red_for_ed.mp4", "out.mp4")
