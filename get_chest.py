import cv2 as cv
import numpy as np
import math

def get_chest_area(dimensions):
    
    croppedframe2= frame[dimensions[0][1]:dimensions[0][3],dimensions[0][0]:dimensions[0][2]]
    height=abs(dimensions[0][1]-dimensions[0][3])
    width=math.ceil(abs(dimensions[0][0]-dimensions[0][2]))
    croppedchest= frame[math.ceil(dimensions[0][1]+(1.5*height)):math.ceil(dimensions[0][3]+(1.5*height)),dimensions[0][0]-width:dimensions[0][2]+width]
    show_images([croppedframe2,croppedchest])
    
