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
    camera = cv.VideoCapture(0)
    
    while(1):
        #ret, frame=camera.read()
        ret, frame = cap.read()
        
        if not ret:
            break
        Danger_zone(frame)

        
    