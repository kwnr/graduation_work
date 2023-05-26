import cv2
import numpy as np

def draw_aruco(dict,id,out_size):
    aruco_img=cv2.aruco.drawMarker(dict,id,out_size)
    return aruco_img
    
def draw_aruco_board(dict,size:list,out_size:list,marker_length,marker_sep,ids:np.ndarray,margin=0):
    aruco_board=cv2.aruco.GridBoard(size,marker_length,marker_sep,dict,ids)
    board_img=cv2.aruco.Board.generateImage(aruco_board,out_size,marginSize=margin)
    return board_img

def draw_charuco_board(dict,size,out_size,margin,square_length,marker_length,ids:np.array):
    charuco_board=cv2.aruco.CharucoBoard(size,square_length,marker_length,dict,ids)
    charuco_img=cv2.aruco.Board.generateImage(charuco_board,out_size,marginSize=margin)
    return charuco_img

if __name__=="__main__":
    aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    img=draw_aruco_board(aruco_dict,[2,2],np.array([100//0.26,100//0.26],np.int32),0.045,0.010,np.array([1,11,21,31]),margin=0)
    cv2.imwrite('aruco_board.png',img)
    cv2.imshow("aruco",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()