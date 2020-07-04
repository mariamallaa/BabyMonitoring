import cv2 as cv
import numpy as np
from collections import namedtuple
import pygame
from bounding_box import *
from yolo import *
from commonfunctions import *
from yolo_Face_Detector import *


m,class_names=yolomodel() 
pygame.mixer.init()
pygame.mixer.music.load('D:\\GP\\GPTest\\hello-hello-female-romantic-voice-40646-50093.mp3')

def getImageDifference(first, second):
    return cv.absdiff(first, second)

def drawRectangle(contour, frame):
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return [x,y,x+w,y+h]
    
def bb_intersection_over_union(boxA, boxB,frame):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA+ 1)
    boxAArea = (abs(boxA[2] - boxA[0]) + 1) * (abs(boxA[3] - boxA[1])+ 1)
    boxBArea = (abs(boxB[2] - boxB[0]) + 1) * (abs(boxB[3] - boxB[1] )+ 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

#We need to identify the borders of the danger zone box


cap = cv.VideoCapture("D:\\GP\\GPTest\\9.mp4")


####################################################################################### Reading the bounding box coor


file1 = open("myfile.txt","r")#write mode 

value=file1.readlines()



with open("myfile.txt") as f:
    
    boundingboxpts=[]
    for line in f: # read rest of lines
        boundingboxpts.append(int(line))

file1.close() 


########################################################################################


fgbg = cv.createBackgroundSubtractorMOG2()
old_fg = 0
output= 0
#camera = cv.VideoCapture(0)
'''
while(1):
    #ret, frame=camera.read()
    ret, frame = cap.read()
    
    if not ret:
        break
'''
def Danger_zone(frame):
    fgmask = fgbg.apply(frame)
    output = cv.GaussianBlur(fgmask, (21, 21), 0)
    
    if old_fg is None:
        old_fg = fgmask
        return

    diff = cv.absdiff(old_fg,output)


    maskRGB = cv.cvtColor(fgmask,cv.COLOR_GRAY2BGR)

    threshold = cv.threshold(fgmask, 21, 255, cv.THRESH_BINARY)[1]

    threshold = cv.dilate(threshold, None, iterations = 2)

    contours, _= cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    sortedContours = sorted(contours, key = cv.contourArea, reverse = True)[:2] #this will track two objects simultaneously. If I want more, I'd have to come and change this value to whatever I want
    imageorginal=frame.copy()
    for contour in sortedContours:

        ObjCoordinates=drawRectangle(contour, frame)
        print("object")
        print(ObjCoordinates)
        Danger_zone=imageorginal[ObjCoordinates[1]:ObjCoordinates[3],ObjCoordinates[0]:ObjCoordinates[2]]
        #show_images([imageorginal,Danger_zone])
        cv.rectangle(frame,(boundingboxpts[0],boundingboxpts[1]),(boundingboxpts[2],boundingboxpts[3]),(255, 0, 0), 2)

        iou = bb_intersection_over_union(ObjCoordinates, [int(boundingboxpts[0]),int(boundingboxpts[1]),int(boundingboxpts[2]),int(boundingboxpts[3])],frame)
        cv.putText(frame, "IoU: {:.4f}".format(iou), (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(iou)
        
        if(iou>=0.5):

            Danger_zone=frame[int(boundingboxpts[1]):int(boundingboxpts[3]),int(boundingboxpts[0]):int(boundingboxpts[2])]
            #show_images([Danger_zone])
            #dimensions=yoloboxes(Danger_zone,m,class_names)
            dimensions=get_face_BB(Danger_zone)
            if(dimensions!=[]):
                croppedframe2= Danger_zone[dimensions[0][1]:dimensions[0][3],dimensions[0][0]:dimensions[0][2]]

                show_images([croppedframe2])
                print("Dangeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeer")
                pygame.mixer.music.play(-1)

            

        old_fg  = diff
    
    draw = frame & maskRGB

    cv.imshow('Main',frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        return

cap.release()
cv.destroyAllWindows()