from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
import cv2 as cv
import numpy as np
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput


def getImageDifference(first, second):
    return cv.absdiff(first, second)

def drawRectangle(contour, frame):
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return [x,y,x+w,y+h]
    
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class PlayingScreen(Screen):
    def on_enter(self):
        DangerZone_pt1= (350,250)
        DangerZone_pt2= (780,500)

        video_capture = cv.VideoCapture(0)
        count = 0
        fgbg = cv.createBackgroundSubtractorMOG2()
        old_fg = 0
        output= 0
        

        while (1):
            ret, frame = video_capture.read()
            if not ret:
                break

            fgmask = fgbg.apply(frame)
            output = cv.GaussianBlur(fgmask, (21, 21), 0)
    
            if old_fg is None:
                old_fg = fgmask
                continue
            diff = cv.absdiff(old_fg,output)

            threshold = cv.threshold(fgmask, 21, 255, cv.THRESH_BINARY)[1]

            threshold = cv.dilate(threshold, None, iterations = 2)

            contours, _= cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            sortedContours = sorted(contours, key = cv.contourArea, reverse = True)[:2] #this will track two objects simultaneously. If I want more, I'd have to come and change this value to whatever I want

            for contour in sortedContours:
                ObjCoordinates=drawRectangle(contour, frame)
                cv.rectangle(frame,DangerZone_pt1,DangerZone_pt2,(255, 0, 0), 2)
                iou = bb_intersection_over_union(ObjCoordinates, [DangerZone_pt1[0],DangerZone_pt1[1],DangerZone_pt2[0],DangerZone_pt2[1]])
                cv.putText(frame, "IoU: {:.4f}".format(iou), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                old_fg  = diff

            
            cv.imshow('Main',frame)
    
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        
        video_capture.release()
        cv.destroyAllWindows()
        return Label(text = "Danger Zone")
        
    