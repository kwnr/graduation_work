import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils

cap = cv2.VideoCapture(1)

target = cv2.imread(
    "/Users/hyeokbeom/Desktop/graduation_work/image_detecting/template_A.png",
    cv2.IMREAD_GRAYSCALE,
)
# target=cv2.resize(target,(200,200))

orb = cv2.ORB_create()

target_key, target_desc = orb.detectAndCompute(target, None)
target_draw = cv2.drawKeypoints(
    target, target_key, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

while cap.isOpened():
    _, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(
        img_gray, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    img_key, img_desc = orb.detectAndCompute(img_bin, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = matcher.knnMatch(img_desc, target_desc, 2)

    ratio = 0.75
    good_matches = [
        first for first, second in matches if first.distance < (second.distance * ratio)
    ]

    res = cv2.drawMatches(
        img,
        img_key,
        target,
        target_key,
        good_matches,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("result", res)
    cv2.imshow("A", target_draw)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
