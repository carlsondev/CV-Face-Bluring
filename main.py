""" Source: https://data-flair.training/blogs/pedestrian-detection-python-opencv/"""

import numpy as np
import cv2
import imutils
import time

from typing import List, Tuple

from yolo_person_detection import get_person_rects, RectType
from face_detection import get_face_rects, blur_faces


def draw_rects(
    bgr_img: np.ndarray, body_rects: List[RectType], face_rects: List[RectType]
) -> np.ndarray:

    img = bgr_img.copy()

    for body in body_rects:
        x, y, w, h = body

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for face in face_rects:
        face_x, face_y, face_w, face_h = face
        img = cv2.circle(
            img,
            (int(face_x + face_w // 2), int(face_y + face_h // 2)),
            10,
            (0, 0, 255),
            -1,
        )

    return img


def get_minutes_seconds(seconds: float) -> Tuple[int, int]:
    time_min = int(seconds // 60)
    time_s = int(seconds - (60 * time_min))

    return time_min, time_s


def main():

    video_capture = cv2.VideoCapture(
        "videos/red_for_ed.mp4"
    )  # Open video capture object

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    got_image, bgr_img = video_capture.read()  # Make sure we can read video

    img_h, img_w, _ = bgr_img.shape

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter("out_small.mp4", fourcc, fps, (img_w, img_h))

    est_min, est_s = get_minutes_seconds(2.5 * frame_count)

    print(
        f"File FPS: {fps}, Frame Count: {frame_count}, Est Max Time: {est_min}:{est_s}"
    )

    if not got_image:
        print("Cannot read video source")
        exit(1)

    window_name = "FinalProject"
    cv2.namedWindow(window_name)
    window_is_open = True

    current_frame_num = 1
    program_start_time = time.time()
    frame_comp_times: List[float] = []

    while window_is_open and got_image:
        start_time = time.time()
        bgr_img = imutils.resize(bgr_img, width=700)
        print(f"Currently processing frame {current_frame_num}/{frame_count}")
        body_rects = get_person_rects(bgr_img)

        face_rects = get_face_rects(bgr_img)

        print(f"Faces Size: {len(face_rects)}")

        bgr_img = blur_faces(bgr_img, body_rects, face_rects)
        bgr_img = draw_rects(bgr_img, body_rects, face_rects)
        end_time = time.time()

        extra_start = time.time()
        # Compute the average computation time of every frame
        frame_comp_time = end_time - start_time
        frame_comp_times.append(frame_comp_time)

        frame_comp_time_avg = sum(frame_comp_times) / len(frame_comp_times)

        # Predict the amount of time remaining
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

        frame_comp_times[-1] += time.time() - extra_start
        print("-" * 100)

        cv2.imshow(window_name, bgr_img)
        cv2.waitKey(1)
        # out.write(bgr_img)
        current_frame_num += 1
        got_image, bgr_img = video_capture.read()

    video_capture.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
