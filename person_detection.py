import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import sys
import time


def detect_people(frame):
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    boxes, weights = hog.detectMultiScale(
        frame, winStride=(3, 3), padding=(2, 2), scale=1.3
    )
    npboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    people = non_max_suppression(npboxes, probs=None, overlapThresh=0.65)
    return frame, people


def main():
    video_capture = cv2.VideoCapture(
        "videos/walking_pedestrians1.mp4"
    )  # Open video capture object
    got_image, bgr_img = video_capture.read()  # Make sure we can read video

    if not got_image:
        print("Cannot read video source")
        sys.exit()
    bgr_img_copy = bgr_img.copy()

    # Start window thread
    cv2.startWindowThread()
    windowName = "FinalProject"
    cv2.namedWindow(windowName)
    windowIsOpen = True

    while True:

        if windowIsOpen:
            start_time = time.time()
            bgr_img_copy, people = detect_people(bgr_img_copy)
            for (xA, yA, xB, yB) in people:
                cv2.rectangle(bgr_img_copy, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.imshow(windowName, bgr_img_copy)

            got_image, bgr_img = video_capture.read()
            bgr_img_copy = bgr_img.copy()

            print(f"Frame took {time.time()-start_time:.2f} seconds")

            if not got_image:
                break

        # If a user presses Q during the movie
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
