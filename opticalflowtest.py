import cv2 as cv
import numpy as np
from skimage import data, io
import matplotlib.pyplot as plt
from commonfunctions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive






#def calcdisplacement(signals):
    

cap = cv.VideoCapture("D:\\breathing2\\breathing2.mp4")
#D:\breathing2
#D:\\GP\\good.mp4

feature_params = dict(maxCorners=50, qualityLevel=0.01,
                      minDistance=10, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

# Take first frame and find corners in it
ret, old_frame = cap.read()
'''
dimensions=[348, 219, 1084, 1949]
old_frame= old_frame[dimensions[0]:dimensions[2],dimensions[1]+400:dimensions[3]]
'''
show_images([old_frame])
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = []
p0 = []
'''
print(old_gray.shape)
for i in range(int(old_gray.shape[0]/40)):
    print("i")
    print(i*40)
    for j in range(int(old_gray.shape[1]/40)):
        gridP = cv.goodFeaturesToTrack(
            old_gray[i*40: (i+1)*40, j*40:(j+1)*40], mask=None, **feature_params)
        print("j")
        print(j*40)
        if gridP is None:
            print("")
        else:
            gridP = np.asarray(gridP)
            
            gridP[:, :, 0] += 40*i
            gridP[:, :, 1] += 40*j
            print(gridP)
            p0.extend(gridP)

p0 = np.asarray(p0)
'''
#print(p0.shape)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#p0 = np.int0(p0)
print(p0)
'''
print(p0.shape)
plt.plot(p0[:,0,0],p0[:,0,1])
plt.show()
'''
'''
ret, frame = cap.read()
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
for i in range(p0.shape[0]):
    print(p0[i,0,:])
    frame = cv.circle(frame, (p0[i,0,1], p0[i,0,0]), 5, color, -1)
show_images([frame])
'''
signals = []
signals.append(p0)
currentframe=1
mask = np.zeros_like(old_frame)
while(1):
    #signals.append(p0)
    #disp.append(calcdisplacement(signals,currentframe))
    currentframe+=1
    ret, frame = cap.read()
    #frame= frame[dimensions[0]:dimensions[2],dimensions[1]+400:dimensions[3]]
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    '''
    dimensions=[348, 219, 1084, 1949]
    frame_gray= frame_gray[dimensions[0]:dimensions[2],dimensions[1]+400:dimensions[3]]
    '''
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]
    print("points")
    print(good_new)
    print(good_old)
    
    
    
    for k, (new, old) in enumerate(zip(good_new, good_old)):
        
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        frame = cv.circle(frame, (b, a), 5, color, -1)
        
    #img = cv.add(frame, mask)
    # cv.imshow('frame',img)

    output = cv.add(frame, mask)
    old_gray = frame_gray.copy()
    #p0 = good_new.reshape(-1, 1, 2)
    p0=p1
    cv.imshow("sparse optical flow", output)
    #show_images([output])
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# pointsx=signals[:, 0, 0, 0]
# pointsy=signals[:, 0, 0, 1]
signals = np.asarray(signals)


for i in range(signals.shape[1]):
    ax.scatter(signals[:, i, 0, 0], signals[:, i, 0, 1],range(signals.shape[0]), c='r', marker='o')
    #ax.plot3D(signals[:, i, 0, 0], signals[:, i, 0, 1],np.linspace(0,signals.shape[0],signals.shape[0]))
    #plt.show()
    #plt.plot(signals[:, i, 0, 0], signals[:, i, 0, 1],range(signals.shape[0]))
   
  
plt.show()
cap.release()
cv.destroyAllWindows()