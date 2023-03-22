import cv2

def set_cap_resolution(cam, h)->None:
    '''change capture resolution with 16:9 ratio'''
    w=int(h/9*16)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    
    

INDEX_PARAM=dict(
        algorithm=6,
        table_number=6,
        key_size=12,
        multi_probe_level=1
    )