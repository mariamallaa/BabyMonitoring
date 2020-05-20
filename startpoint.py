from yolo import *
from Harris import *
import matplotlib.pyplot as plt
import os 
import numpy as np
import cv2
from commonfunctions import *
from DangerZone import *


# Read the video from specified path 
cam = cv2.VideoCapture("D:\\GP\\GPTest\\7.mp4") 

#building yolo model
m,class_names=yolomodel() 

currentframe = 0
acc=0
ret,frame = cam.read() 
pts=initilizeboundingbox(frame)
while(True): 
    # reading from frame 
    ret,frame = cam.read() 
    if ret: 
        frame=Dangerzone(frame,pts)
        cv2.imshow('Main',frame)
        #dimensions=yoloboxes(frame,m,class_names)
        #croppedframe= frame[dimensions[0]:dimensions[2],dimensions[1]:dimensions[3]]
        #show_images([croppedframe])
        #print(croppedframe.shape)
        #gray = cv2.cvtColor(croppedframe, cv2.COLOR_BGR2GRAY)
        #harris(gray)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    else: 
        break
cam.release()
cv.destroyAllWindows()
