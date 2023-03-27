import image_detecting.trials.detect_arrow as detect
import numpy as np
import cv2

original_ratio=1.5 #head/right

def find_angle_vector(v1,v2):
    theta=np.arccos(np.dot(v1,v2)/(np.sqrt(np.sum(np.square(v1)))*np.sqrt(np.sum(np.square(v1)))))
    theta=np.rad2deg(theta)
    return theta

def estimate_roll(properties):
    head=properties["head"]
    center=properties["center"]
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

def estimate_pitch_yaw(properties):
    ratio=properties['head_distance']/properties['right_distance']
    angle=find_angle_vector([properties['head'],properties['center']],[properties['right'],properties['left']])
    if ratio>1.5:
        if angle>90:
            yaw_like=
    


if __name__=='__main__':
    cap=cv2.VideoCapture(1)
    while cap.isOpened():
        arrows=detect.detect_arrow()
        for arrow in arrows:
            properties=arrow['properties']
            