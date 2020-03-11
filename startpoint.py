from yolo import *
from Harris import *
import matplotlib.pyplot as plt
import os 
import numpy as np
import cv2
from commonfunctions import *


# Read the video from specified path 
cam = cv2.VideoCapture("D:\\breathing2\\breathing2.mp4") 

#building yolo model
m,class_names=yolomodel() 

currentframe = 0
acc=0
while(True): 
    # reading from frame 
    ret,frame = cam.read() 
    if ret: 
        
        dimensions=yoloboxes(frame,m,class_names)
        croppedframe= frame[dimensions[0]:dimensions[2],dimensions[1]:dimensions[3]]
        #show_images([croppedframe])
        #print(croppedframe.shape)
        gray = cv2.cvtColor(croppedframe, cv2.COLOR_BGR2GRAY)
        harris(gray)
        
    else: 
        break
  