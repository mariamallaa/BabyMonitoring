import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from Danger_zone.RegressionModel import *
import numpy as np





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
            if(self.face_dimensions_old!=[]):
                boxA=np.asarray(self.face_dimensions)
                boxB=np.asarray(self.face_dimensions_old)
                diff = cv.absdiff(boxA, boxB)
                self.sum=np.sum(diff)

            self.countframes=0
        self.face_dimensions_old=self.face_dimensions
        return self.iscovered
    

    def Is_Covered(self):
        return self.iscovered

    def is_face_same(self):
        if(self.face_dimensions!=[] and self.face_dimensions_old!=[]):
            #print("Difffff")
            #print(self.sum)
            if(self.sum<100):
                return True
            else:
                return False
        else:
            #print(self.face_dimensions)
            #print(self.face_dimensions_old)
            #print("no face")
            return True
    
#    def set_oldface(self):
#        self.face_dimensions_old=self.face_dimensions

        
#    def detect_motion(self,frame):

