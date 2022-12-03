""" Source: https://data-flair.training/blogs/pedestrian-detection-python-opencv/"""

import numpy as np
import cv2
import tensorflow
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
            1,
            (0, 0, 255),
            -1,
        )

    return img


def get_minutes_seconds(seconds: float) -> Tuple[int, int]:
    time_min = int(seconds // 60)
    time_s = int(seconds - (60 * time_min))

    return time_min, time_s


def main(input_video_path: str, output_video_path: str):

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
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (img_w, img_h))

    # Estimate the amount of time it will take to process.
    # ~0.22 s/frame on cuda, 2.5 s/frame on CPU
    start_avg = 0.22 if tensorflow.test.is_gpu_available() else 2.5
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

        face_rects = get_face_rects(bgr_img)

        print(f"Faces Count: {len(face_rects)}")

        bgr_img = blur_faces(bgr_img, body_rects, face_rects)

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

        cv2.imshow("Final Project", bgr_img)
        cv2.waitKey(1)

        out.write(bgr_img)
        current_frame_num += 1
        got_image, bgr_img = video_capture.read()

    print("Finished!")
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("videos/red_for_ed.mp4", "out.mp4")
