import cv2
import numpy as np
from typing import Tuple, List


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_face_in_img(img : np.ndarray) -> List[Tuple[float, float, float, float]]:
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
