import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
import sys

frame = cv2.imread("images/pedestrian2.jpg")

def detect_people(frame):
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    boxes, weights = hog.detectMultiScale(frame, winStride=(2,2), padding=(2,2), scale=1.20)
    npboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    people = non_max_suppression(npboxes, probs=None, overlapThresh=0.65)
    return people

if __name__ == "__main__":
    video_reader = cv2.VideoCapture("videos/pedestrian_walking1.mp4")

    got_image, frame = video_reader.read()

    if not got_image:
        print("cannot read video")
        sys.exit()
    
    while True:
        if cv2.getWindowProperty("CCC Video",cv2.WND_PROP_VISIBLE) < 1:   # If user presses 'X' on the window,      
            cv2.destroyAllWindows()
            break

        got_image, bgr_image = video_capture.read() 

        if not got_image:
            break


        people = detect_people(frame)  

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("test", frame)

cv2.waitKey(0)