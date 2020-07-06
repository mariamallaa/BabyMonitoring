from kivy.uix.screenmanager import Screen
import time
from kivy.uix.image import Image
from kivy.config import Config
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
import cv2 as cv
import numpy as np
from skimage import data, io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
import math
from scipy import fftpack
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy import signal
import statistics
import collections
from scipy.stats import entropy
from scipy.signal import butter, lfilter
from sklearn.decomposition import FastICA, PCA
from numpy import linalg as LA
import scipy.ndimage
from scipy import signal
import math
import pygame
from Breathing_Rate import *
from Danger_Zone_integrated import *


class SleepingScreen(Screen):

    def __init__(self, **kwargs):
        super(SleepingScreen, self).__init__(**kwargs)
        self.label = Label(text=" ",color=(1,0,0,1), font_size=(20),size_hint=(0.2,0.1), pos_hint={"center_x":0.5, "center_y":0.9})
        self.add_widget(self.label)
    
    def check_user_input(self,age):

        if age == '':
            self.label.text= "Please re-enter !"
            return self.__init__()

        elif age.isdigit() == False:
            self.label.text= "Please re-enter !"
            return self.__init__()
        
        else :
            
            self.detect_breathing_rate(age)
    


    def detect_breathing_rate(self,age):

        ######################initialize alarm##################
        pygame.mixer.init()
        pygame.mixer.music.load('fire-truck-air-horn_daniel-simion.wav')

        ############Start detecting breathing rate##############
        feature_params = dict(maxCorners=100, qualityLevel=0.05,
                      minDistance=20, blockSize=3)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        ##################need to get baby's age#######################
        breathing_rate("WIN_20200706_23_36_19_Pro.mp4",
               feature_params, lk_params, "results.txt", int(age))

        
        
       

    
        
