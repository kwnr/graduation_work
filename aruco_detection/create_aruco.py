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
    aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    charuco_img=draw_charuco_board(aruco_dict,[3,3],[1000,1000],50,200,100,np.array([1,11,21,31]))
    cv2.imwrite('charuco.png',charuco_img)
    cv2.imshow("charuco",charuco_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()