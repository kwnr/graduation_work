import cv2
import numpy as np
import shapely

cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, img_bin = cv2.threshold(
        img_blur, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        eps = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps * 0.02, True)
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 3)
        cv2.putText(
            img,
            f"{len(approx)}",
            approx[0, 0],
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            3,
        )
    cv2.imshow("bin", img_bin)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
