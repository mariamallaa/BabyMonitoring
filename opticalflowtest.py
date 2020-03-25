import cv2 as cv
import numpy as np
from skimage import data, io
import matplotlib.pyplot as plt
from commonfunctions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
import math  
 
     




def calcdisplacement(signals,currentframe,p1):
    disp=[]
    for i in range(signals.shape[1]):
        dispxy=[]
        x=p1[i,0,0]-signals[0,i,0,0]
        y=p1[i,0,1]-signals[0,i,0,1]
        dist = math.sqrt((x)**2 + (y)**2) 
        #dispxy.append([dist])
        disp.append(dist)

    return disp




cap = cv.VideoCapture("D:\\breathing2\\breathing2.mp4")
#D:\breathing2
#D:\\GP\\good.mp4



feature_params = dict(maxCorners=100, qualityLevel=0.01,
                      minDistance=10, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

# Take first frame and find corners in it
ret, old_frame = cap.read()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = []

p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


signals = []
disp=[]
signals.append(p0)
signals = np.asarray(signals)
currentframe=1
mask = np.zeros_like(old_frame)
while(1):
    
   
    ret, frame = cap.read()
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
   
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print("P1")
    print(p1.shape)
    disp.append(calcdisplacement(signals,currentframe,p1))
    currentframe+=1
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    
    for k, (new, old) in enumerate(zip(good_new, good_old)):
        
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        frame = cv.circle(frame, (b, a), 5, color, -1)
        
    

    output = cv.add(frame, mask)
    old_gray = frame_gray.copy()
    p0=p1
    cv.imshow("sparse optical flow", output)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    


disp = np.asarray(disp)


disp2=disp.transpose()




for i in range(disp2.shape[0]): 
    plt.plot(disp2[i],range(disp2.shape[1]))
    plt.show()

cap.release()
cv.destroyAllWindows()