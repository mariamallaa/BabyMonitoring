import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from Danger_zone.RegressionModel import *

def is_face_same(boxA,boxB):
    print(boxA,boxB)
    boxA=np.asarray(boxA)
    boxB=np.asarray(boxB)
    diff = cv.absdiff(boxA, boxB)
    if(np.sum(diff)<8):
        return True
    else:
        return False



class CoverFace:

    def __init__(self):
        self.face_dimensions_old=[]
        self.face_dimensions=[]
        self.countframes=0
        self.iscovered=False
        self.firsttime=True
    def get_first_frame(self,old_frame):
        self.face_dimensions_old=detectface_RM(old_frame)
        self.firsttime=False

    def return_first_time(self):
        return self.firsttime
    def detect_covered(self,frame):
        self.iscovered=False
        self.face_dimensions=detectface_RM(frame)
        if(self.face_dimensions==[]):
            self.countframes+=1
            if(self.countframes==4):
                self.countframes=0
                self.iscovered=True
        else:
            self.countframes=0
        self.face_dimensions_old=self.face_dimensions
        return self.iscovered
    

    def Is_Covered(self):
        return self.iscovered
        
#    def detect_motion(self,frame):

