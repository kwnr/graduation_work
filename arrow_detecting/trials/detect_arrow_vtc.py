import cv2
import numpy as np
import time
import shapely


def get_heading(head, center):
    hx, hy = head
    cx, cy = center
    if hy - cy == 0:
        if hx > cx:
            heading = np.pi
        else:
            heading = -np.pi
    else:
        heading = np.arctan(np.divide(hx - cx, hy - cy))
        if hy > cy:
            if hx < cx:
                heading = heading + np.pi
            else:
                heading = heading - np.pi
    return np.rad2deg(heading)


def setLabel(img, pts, label):
    # 사각형 좌표 받아오기
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


cap = cv2.VideoCapture(1)
frame_rate = 1
prev_time = 0
cap.set(cv2.CAP_PROP_FPS, 5)

w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


while cap.isOpened():
    curr_time = time.time()

    ret, frame = cap.read()

    term = curr_time - prev_time
    fps = 1 / term
    prev_time = curr_time

    status, img = cap.read()

    canv = np.zeros_like(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray, (21, 21), 0)  # blur capture

    # _,img_bin=cv2.threshold(img_canny,127,255,cv2.THRESH_BINARY)
    _, img_bin = cv2.threshold(
        img_gauss, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    for i in range(len(contours)):
        contour = contours[i]
        # cv2.putText(img,f'{i}',contours[i][0][0],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        if cv2.contourArea(contour) < 400:  # 노이즈 제거, 너무 작으면 무시
            continue
        if not shapely.Polygon(contour[:, 0, :]).is_valid:
            continue

        eps = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon=eps * 0.02, closed=True)
        # cv2.drawContours(img,[approx],0,(255,0,0),3)
        vtc = len(approx)
        # cv2.putText(img,f'vtc:{vtc}',approx[0][0],cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),3)
        if vtc == 7:
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
            hull = cv2.convexHull(approx, returnPoints=False)
            if hull is not None:
                try:
                    defects = cv2.convexityDefects(approx, hull)
                except:
                    print(f"****")
                    continue
                if defects is not None:
                    if len(defects) == 2:
                        M = cv2.moments(approx)
                        cx = int(M["m10"] / M["m00"])  # 폐곡선의 중심 계산
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(img, (cx, cy), 1, (0, 0, 255), 2)
                        cv2.putText(
                            img,
                            f"AREA: {cv2.contourArea(contours[i])}, COORD: ({cx}, {cy})",
                            tuple(contours[i][0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        points = np.array([defects[0][0][:2], defects[1][0][:2]])
                        if not ((points[:, 0] - points[:, 1]) % 7 == 5).all():
                            continue
                        if not np.sum(points.shape) == len(np.unique(points)):
                            continue

                        head_p = (
                            21 - np.sum(points[:, 0] + 1) - np.sum(points.reshape(4))
                        )
                        head = approx[head_p][0]
                        right_p = (head_p + 1) % 7
                        left_p = (head_p - 1) % 7
                        hrl = [head, approx[right_p][0], approx[left_p][0]]
                        cv2.circle(img, head, 10, (255, 255, 255), -1)
                        cv2.drawContours(img, [approx], 0, (0, 0, 255), 3)
                        cv2.drawContours(canv, [approx], 0, (255, 255, 255), 3)
                        setLabel(img, approx, f"{get_heading(head,[cx,cy]):.2f}")
                        for ix in range(len(hrl)):
                            cv2.circle(img, hrl[ix], 10, (0, 255, 0), -1)
                        cv2.putText(
                            img,
                            "right",
                            hrl[1],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4,
                            (0, 255, 0),
                            3,
                        )

    a = np.hstack((cv2.cvtColor(img_gauss, cv2.COLOR_GRAY2BGR), canv))
    b = np.hstack((cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR), img))
    merged = np.vstack((a, b))
    cv2.putText(
        merged, f"{fps:.2f}", [10, 50], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
    )

    cv2.imshow("img", merged)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
