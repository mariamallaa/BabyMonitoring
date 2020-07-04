import cv2 as cv
import numpy as np
from commonfunctions import *
from yolo_Face_Detector import *

def check_ifcover(image,dimensions,dimensionold):
    croppedframe2= image[dimensions[0][1]+dimensionold[0][1]-100:dimensions[0][3]+dimensionold[0][3]+100,dimensions[0][0]+dimensionold[0][0]-100:dimensions[0][2]+dimensionold[0][2]+100]
    show_images([croppedframe2])
    newdimensions=get_face_BB(croppedframe2)
    print("HIIIIIIIIIIIIIIIIIIIII")
    print(newdimensions)
    if(newdimensions!=[]):
        croppedframe2= croppedframe2[newdimensions[0][1]:newdimensions[0][3],newdimensions[0][0]:newdimensions[0][2]]

        show_images([croppedframe2])
        print("there is a face")
        return newdimensions
    else:
        print("dangeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeer")
        return []


    