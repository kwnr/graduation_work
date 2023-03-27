import cv2
import numpy as np
import time
import shapely
from scipy.spatial.transform import Rotation as R


def setLabel(img, pts, label):
    # 사각형 좌표 받아오기
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def preprocessing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray, (21, 21), 0)
    _, img_bin = cv2.threshold(
        img_gauss, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return img_bin


def detect_contours(img_bin):
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def detect_approxs(contours):
    """detects approximated contours using cv2.approxPolyDP from input binary image.
    returns approxed contours.

    Args:
        img_bin (array):
    """
    approxes = {}
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

        # cv2.putText(img,f'vtc:{vtc}',approx[0][0],cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),3)
        approxes[i] = approx
    return approxes


def find_arrow_like(approxes: dict, hierarchies):
    arrow_like = {}
    points_ls = {}
    parents_approx = {}
    for i in approxes.keys():
        approx = approxes[i]
        hierarchy = hierarchies[0][i]
        parent = hierarchy[3]
        if len(approx) == 7:
            if parent != -1:
                if parent not in approxes.keys():
                    continue
                parent_approx = approxes[parent]
                if len(parent_approx) == 4:
                    hull = cv2.convexHull(approx, returnPoints=False)
                    if hull is not None:
                        try:
                            defects = cv2.convexityDefects(approx, hull)
                        except:
                            print(f"****")
                            continue
                        if defects is not None:
                            if len(defects) == 2:
                                points = np.array(
                                    [defects[0][0][:2], defects[1][0][:2]]
                                )
                                if not ((points[:, 0] - points[:, 1]) % 7 == 5).all():
                                    continue
                                if (
                                    points[0, 1] == points[1, 0]
                                    or points[0, 0] == points[1, 1]
                                ):
                                    continue
                                arrow_like[i] = approx
                                points_ls[i] = points
                                parents_approx[i] = parent_approx
    return arrow_like, points_ls, parents_approx


def find_angle_vector(v1, v2):
    theta = np.arccos(
        np.dot(v1, v2)
        / (np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2))))
    )
    theta = np.rad2deg(theta)
    return theta


def find_arrow_like_properties(approx, contour, points):
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])  # 폐곡선의 중심 계산
    cy = int(M["m01"] / M["m00"])
    center = np.array([cx, cy])
    head_n = 21 - np.sum(points[:, 0] + 1) - np.sum(points.reshape(4))
    head = approx[head_n][0]
    right = approx[(head_n + 1) % 7][0]
    left = approx[(head_n - 1) % 7][0]
    # 중심축에서 오른쪽, 왼쪽 까지의 거리, https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points 참조
    right_distance = np.linalg.norm(
        np.cross(center - head, head - right)
    ) / np.linalg.norm(center - head)
    left_distance = np.linalg.norm(
        np.cross(center - head, head - left)
    ) / np.linalg.norm(center - head)
    head_distance = np.linalg.norm(
        np.cross(left - right, right - head)
    ) / np.linalg.norm(left - right)
    intersection = line_intersection([head, center], [right, left])
    properties = {
        "M": M,
        "center": center,
        "head": head,
        "right": right,
        "left": left,
        "right_distance": right_distance,
        "left_distance": left_distance,
        "head_distance": head_distance,
        "intersection": intersection,
    }
    return properties

    """cv2.circle(img,head,10,(255,255,255),-1)
    cv2.drawContours(img,[approx],0,(0,0,255),3)
    cv2.drawContours(canv,[approx],0,(255,255,255),3)
    setLabel(img,approx,f'{get_heading(head,[cx,cy]):.2f}')
    for ix in range(len(hrl)):
        cv2.circle(img,hrl[ix],10,(0,255,0),-1)
    cv2.putText(img,'right',hrl[1],cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),3)"""


def draw(img, approx, properties):
    cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
    cv2.circle(img, properties["center"], 3, (0, 0, 255))
    for pos in ["head", "left", "right"]:
        cv2.circle(img, properties[pos], 5, (255, 0, 0), 3)
    """for pos in ["left", "right"]:
        cv2.putText(
            img,
            f"{properties[f'{pos}_distance']:.2f}",
            properties[pos],
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3,
        )"""
    return img


def detect_arrow(img):
    """detect arrows in input image.

    Args:
        img : input image

    Returns:
        list: list of dictionary. each dictionary means a arrow.
    """
    arrows = []
    img_bin = preprocessing(img)
    contours, hierarchy = detect_contours(img_bin)
    approxes = detect_approxs(contours)
    arrow_like, points_ls, parents = find_arrow_like(approxes, hierarchy)
    for i in arrow_like.keys():
        properties = find_arrow_like_properties(
            arrow_like[i], contours[i], points_ls[i]
        )
        properties["contour"] = contours[i]
        properties["approx"] = arrow_like[i]
        properties["parent"] = parents[i]
        if (
            abs(properties["left_distance"] - properties["right_distance"])
            / max(properties["left_distance"], properties["right_distance"])
            < 0.2
        ):
            if (
                cv2.pointPolygonTest(contours[i], properties["intersection"], False)
                == 1
            ):
                arrows.append(properties)
    return arrows


def line_intersection(
    line1, line2
):  # from https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    frame_rate = 1
    prev_time = 0
    cap.set(cv2.CAP_PROP_FPS, 5)
    h = 1267
    w = 771

    while cap.isOpened():
        curr_time = time.time()
        term = curr_time - prev_time
        status, img = cap.read()
        fps = 1 / term
        prev_time = curr_time
        arrows = detect_arrow(img)
        for arrow in arrows:
            cv2.circle(img, arrow["center"], 5, (0, 255, 0), 3)
            cv2.drawContours(
                img, [arrow["contour"], arrow["parent"]], -1, (0, 0, 255), 3
            )
            distance = [
                np.sqrt(np.sum(np.square(arrow["parent"][i][0] - arrow["left"])))
                for i in range(4)
            ]
            tl = np.argmin(distance) - 1
            mtrx = cv2.getPerspectiveTransform(
                np.float32(
                    [
                        arrow["parent"][tl % 4, 0, :],
                        arrow["parent"][(tl + 1) % 4, 0, :],
                        arrow["parent"][(tl + 2) % 4, 0, :],
                        arrow["parent"][(tl + 3) % 4, 0, :],
                    ]
                ),
                np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
            )
            rtv = cv2.warpPerspective(img, mtrx, (w, h))
            cv2.imshow("warp", rtv)
            r = R.from_matrix(mtrx)
            euler = r.as_euler("zxy",degrees=True)
            cv2.putText(
                img,
                f"x:{euler[0]:.0f} y:{euler[1]:.0f} z:{euler[2]:.0f}",
                arrow["parent"][0][0],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

        cv2.putText(
            img, f"{fps:.2f}", [10, 50], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
        )
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
