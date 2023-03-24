import cv2
import numpy as np

def set_cap_resolution(cam, h)->None:
    '''change capture resolution with 16:9 ratio'''
    w=int(h/9*16)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    
    
def get_heading(head,center):
    hx,hy=head
    cx,cy=center
    if hy-cy==0:
        if hx>cx:
            heading=np.pi
        else:
            heading=-np.pi
    else:
        heading=np.arctan(np.divide(hx-cx,hy-cy))
        if hy<cy:
            heading=-heading
    return heading