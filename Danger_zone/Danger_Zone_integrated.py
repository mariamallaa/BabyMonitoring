import cv2 as cv
import numpy as np
from collections import namedtuple
import pygame
from Danger_zone.bounding_box  import *
from Danger_zone.yolo_Face_Detector import *
from Danger_zone.RegressionModel import *
from Danger_zone.Face_covered import *

import sys

from Danger_zone.commonfunctions import *

pygame.mixer.init()
pygame.mixer.music.load('D:\\GP\\GPTest\\hello-hello-female-romantic-voice-40646-50093.mp3')

def getImageDifference(first, second):
    return cv.absdiff(first, second)

def drawRectangle(contour, frame):
        (x, y, w, h) = cv.boundingRect(contour)
        #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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



class DangerZone:
    def __init__(self):
        self.boundingboxpts=[]
        self.fgbg = cv.createBackgroundSubtractorMOG2()
        self.fgmask = None
        self.output = None
        self.old_fg= None
        self.isDanger=False
    def setDangerZone(self,dangerzone):
        self.boundingboxpts=dangerzone
    def getDangerZone(self):
        return self.boundingboxpts
    def init_BB(self):
        #define_BB(frame)
        file1 = open("myfile.txt","r")#write mode 
        with open("myfile.txt") as f:
            boundingboxpts=[]
            for line in f: # read rest of lines
                boundingboxpts.append(int(line))
        file1.close() 
        return boundingboxpts

    def Danger_zone(self,frame,safezone):
        self.isDanger=False
        self.fgmask = self.fgbg.apply(frame)
        self.output = cv.GaussianBlur(self.fgmask, (21, 21), 0)
        if self.old_fg is None:
            self.old_fg = self.fgmask
        
        diff = cv.absdiff(self.old_fg,self.output)
        maskRGB = cv.cvtColor(self.fgmask,cv.COLOR_GRAY2BGR)
        threshold = cv.threshold(self.fgmask, 21, 255, cv.THRESH_BINARY)[1]
        threshold = cv.dilate(threshold, None, iterations = 2)

        contours, _= cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        sortedContours = sorted(contours, key = cv.contourArea, reverse = True)[:2] #this will track two objects simultaneously. If I want more, I'd have to come and change this value to whatever I want
        imageorginal=frame.copy()
        for contour in sortedContours:

            ObjCoordinates=drawRectangle(contour, frame)

            Contour_zone=imageorginal[ObjCoordinates[1]:ObjCoordinates[3],ObjCoordinates[0]:ObjCoordinates[2]]

            iou = bb_intersection_over_union(ObjCoordinates, [int(self.boundingboxpts[0]),int(self.boundingboxpts[1]),int(self.boundingboxpts[2]),int(self.boundingboxpts[3])],frame)


            danger_zone=imageorginal[int(self.boundingboxpts[1]):int(self.boundingboxpts[3]),int(self.boundingboxpts[0]):int(self.boundingboxpts[2])]
            if(iou>=0.5 and not safezone):
                #yolo
                #dimensions=get_face_BB(Contour_zone)
                #regressionmodel
                dimensions=detectface_RM(Contour_zone)
                if(dimensions!=[]):
                    croppedframe2= Contour_zone[dimensions[0][1]:dimensions[0][3],dimensions[0][0]:dimensions[0][2]]
                    #show_images([croppedframe2])
                    self.isDanger=True
                    #pygame.mixer.music.play(-1)
            elif(iou<=0.2 and  safezone):
                #yolo
                #dimensions=get_face_BB(danger_zone)
                #regressionmodel
                dimensions=detectface_RM(danger_zone)
                
                if(dimensions==[]):
                    self.isDanger=True

                

                

            self.old_fg  = diff
        
        
        
        return self.isDanger


    def Is_Danger(self):
        return self.isDanger
