import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, img = cap.read()

    edges = cv2.Canny(img, 100, 200)

    cv2.imshow("edges", edges)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
