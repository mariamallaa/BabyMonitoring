from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
import cv2 as cv
import numpy as np
from collections import namedtuple
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from Danger_Zone_integrated import *


class PlayingScreen(Screen):
   def on_enter(self,**kwargs):
        fgbg = cv.createBackgroundSubtractorMOG2()
        
        bbpts=init_BB("karma rotated.mp4")
        #print("hellllllooooooooooooooooooooooooooo")
        cap = cv.VideoCapture("karma rotated.mp4")
        print("camera open")
        old_fg = 0
        while(1):
            #ret, frame=camera.read()
            ret, frame = cap.read()
            print("frame read")
            if not ret:
                break
            fgmask = fgbg.apply(frame)
            output = cv.GaussianBlur(fgmask, (21, 21), 0)
            if old_fg is None:
                old_fg = fgmask
                continue
            
            frame= Danger_zone(frame,old_fg,fgmask,output,False,bbpts)
            cv.imshow('Main',frame)
            
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        
    