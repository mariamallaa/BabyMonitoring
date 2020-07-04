
import matplotlib.pyplot as plt
import os 

from commonfunctions import *
import cv2 as cv
import numpy as np
from commonfunctions import *
from yolo_Face_Detector import *

def check_ifcover(image):
    
    newdimensions=get_face_BB(image)
    #print("HIIIIIIIIIIIIIIIIIIIII")
    #print(newdimensions)
    if(newdimensions!=[]):
        croppedframe2= image[newdimensions[0][1]:newdimensions[0][3],newdimensions[0][0]:newdimensions[0][2]]
        print("there is a cropped face")
        #show_images([croppedframe2])
        #print("there is a face")
        return newdimensions
    else:
        print("dangeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeer")
        return []


    


# Read the video from specified path 
cam = cv2.VideoCapture("WIN_20200704_15_14_19_Pro.mp4") 
ret,frame = cam.read() 
newdimensions=get_face_BB(frame)
if(newdimensions==[]):
    print("there is no face")

currentframe = 0
acc=0
olddim=newdimensions
facefound=True
while(True): 
    # reading from frame 
    ret,frame = cam.read() 
    if ret and facefound: 
        frameface=frame[olddim[0][1]-100:olddim[0][3]+100,olddim[0][0]-100:olddim[0][2]+100]
        print("this is the face")
        show_images([frame,frameface])
        dimen=check_ifcover(frameface)
        print(dimen)
        if(dimen==[]):
            newdimensions=get_face_BB(frame)
            dimen=[[0,0,0,0]]
            if(newdimensions==[]):
                print("no faceeeeeeeeeeeeeeeeeeeeeeee")
                newdimensions=[[0,0,0,0]]
                facefound=False
                continue
        olddim=newdimensions
        
        olddim[0][1]+=dimen[0][1]
        olddim[0][2]-=dimen[0][2]
        olddim[0][3]-=dimen[0][3]
        olddim[0][0]+=dimen[0][0]
        print(olddim[0][0],olddim[0][1],olddim[0][2],olddim[0][3])
        check=frame[olddim[0][1]:olddim[0][3],olddim[0][0]:olddim[0][2]]
        print("checkkkkkkkkkkkk")
        show_images([frameface,check])

        '''
        dimensions=yoloboxes(frame,m,class_names)
        croppedframe= frame[dimensions[0]:dimensions[2],dimensions[1]:dimensions[3]]
        #show_images([croppedframe])
        #print(croppedframe.shape)
        gray = cv2.cvtColor(croppedframe, cv2.COLOR_BGR2GRAY)
        harris(gray)
        '''
        cv.imshow('Main',frame)
        
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        
    elif ret:
        newdimensions=get_face_BB(frame)
        dimen=[[0,0,0,0]]
        if(newdimensions!=[]):
            facefound=True   
    else:
        break
        
  