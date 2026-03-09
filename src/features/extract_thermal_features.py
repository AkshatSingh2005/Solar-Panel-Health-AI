import cv2
from matplotlib.pyplot import gray
import numpy as np


def extract_features(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # temperature proxy statistics
    max_temp = np.max(gray)
    mean_temp = np.mean(gray)
    temp_variance = np.var(gray)

    # hotspot detection
    # detect local thermal anomalies using blur difference
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(gray, blur)

    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = gray.shape[0] * gray.shape[1]
    min_area = 0.001 * image_area

    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    hotspot_count = len(contours)

    hotspot_area = sum(cv2.contourArea(c) for c in contours)

    return {
        "max_temp": max_temp,
        "mean_temp": mean_temp,
        "temp_variance": temp_variance,
        "hotspot_count": hotspot_count,
        "hotspot_area": hotspot_area
    }