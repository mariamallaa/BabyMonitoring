import cv2 as cv
import numpy as np
from collections import namedtuple
import pygame
from boundingbox import *
from yolo import *
from commonfunctions import *

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
    #print("corrdinates")
    #print(xA,yA,xB,yB)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA+ 1)
    '''
    if(interArea!=0):
        cv.rectangle(frame,(xA,yA),(xB,yB),(255, 0, 0), 2)
        show_images([frame])
    '''
    #print("intersection:")
    #print(interArea)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (abs(boxA[2] - boxA[0]) + 1) * (abs(boxA[3] - boxA[1])+ 1)
    boxBArea = (abs(boxB[2] - boxB[0]) + 1) * (abs(boxB[3] - boxB[1] )+ 1)
    #print(boxBArea)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

#We need to identify the borders of the danger zone box


# cap = cv.VideoCapture(
#     "C:\\Users\\Maram\\Desktop\\GP2\\youssef trimmed\\side cover.mp4")

cap = cv.VideoCapture("D:\\GP\\GPTest\\7.mp4")


#######################################################################################3


file1 = open("myfile.txt","r")#write mode 

value=file1.readlines()
#print("values")
#print(value)


with open("myfile.txt") as f:
    
    boundingboxpts=[]
    for line in f: # read rest of lines
        boundingboxpts.append(int(line))
#print(boundingboxpts)
file1.close() 


########################################################################################
# cap = cv.VideoCapture(
#     "C:\\Users\\Maram\\Downloads\\highway.mp4")

# ret, old_frame = cap.read()
# old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# while(ret):

#     ret, frame = cap.read()
#     if(ret == False):
#         break
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     subtraction = frame-old_frame
#     # print(subtraction)
#     old_frame = frame
#     cv.imshow("frame subtraction", subtraction)
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break


fgbg = cv.createBackgroundSubtractorMOG2()
old_fg = 0
output= 0
#camera = cv.VideoCapture(0)
while(1):
    #ret, frame=camera.read()
    ret, frame = cap.read()
    
    if not ret:
        break
   
    fgmask = fgbg.apply(frame)
    output = cv.GaussianBlur(fgmask, (21, 21), 0)
    
    if old_fg is None:
        old_fg = fgmask
        continue

    diff = cv.absdiff(old_fg,output)
    # print(diff)
    # check adaptive thresholding
   # _, diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
   # diff = cv.medianBlur(fgmask, 3)
   # diff = cv.medianBlur(diff, 3)
    #old_fg = fgmask
    #cv.imshow('frame', diff)
    ##fgmask = backgroundSubtractor.apply(frame, learningRate = 1.0/10)
    ##diff = cv.GaussianBlur(fgmask, (21, 21), 0)
    
    #frameDelta = getImageDifference(old_fg ,diff )
    maskRGB = cv.cvtColor(fgmask,cv.COLOR_GRAY2BGR)
    #frameDela = maskRGB
#    frameDelta = cv2.cvtColor(fgMask, cv2.COLOR_BGR2GRAY)
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
        #print(ObjCoordinates)
        #ObjCoordinates=[0,1000,250,0]
        #print(Danger_zone_pt1)
        #print(Danger_zone_pt2)
        iou = bb_intersection_over_union(ObjCoordinates, [int(boundingboxpts[0]),int(boundingboxpts[1]),int(boundingboxpts[2]),int(boundingboxpts[3])],frame)
        cv.putText(frame, "IoU: {:.4f}".format(iou), (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(iou)
        
        if(iou>=0.5):
            '''
            x1=max(boundingboxpts[0],boundingboxpts[1])
            x2=min(boundingboxpts[0],boundingboxpts[1])
            y1=max(boundingboxpts[0],boundingboxpts[1])
            '''
            #print("da5aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaal")
            #[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            Danger_zone=frame[int(boundingboxpts[1]):int(boundingboxpts[3]),int(boundingboxpts[0]):int(boundingboxpts[2])]
            #show_images([Danger_zone])
            dimensions=yoloboxes(Danger_zone,m,class_names)
            if(dimensions!=[]):
                croppedframe2= Danger_zone[dimensions[1]:dimensions[3],dimensions[0]:dimensions[2]]
                #print(croppedframe2)
                #print(croppedframe2.shape)
                #show_images([croppedframe2])
                print("Dangeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeer")
                pygame.mixer.music.play(-1)
            #croppedframe2= frame[dimensions[0]:dimensions[2],dimensions[1]:dimensions[3]]
            #show_images([croppedframe2])
            #pygame.mixer.music.play(-1)
            

        old_fg  = diff
    
    draw = frame & maskRGB

    cv.imshow('Main',frame)
    #cv.imshow('Background Subtraction', fgmask)
    #cv.imshow('Background Subtraction with color', draw)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()