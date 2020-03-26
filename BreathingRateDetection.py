import numpy as np
import cv2
#import dlib
from commonfunctions import *
import math
import cv2 as cv
from skimage import data, io
import matplotlib.pyplot as plt
from commonfunctions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
from scipy import fftpack
from sklearn import preprocessing
import datetime
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

# HOG
'''
from imutils import face_utils
font = cv2.FONT_HERSHEY_SIMPLEX

face_detect = dlib.get_frontal_face_detector()
#rects = face_detect(gray, 1)

video_capture = cv2.VideoCapture("C:\\Users\\Mariam Alaa\\Downloads\\5518996\\sleep dataset\\sleep dataset\\1 cyc\\rgb.avi")
flag = 0

while True:

    ret, frame = video_capture.read()
   

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2=gray
    show_images([gray2])
    rects = face_detect(gray2, 1)
    print(rects)
    for (i, rect) in enumerate(rects):

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        y=math.floor(y+h/2)
        x=math.floor(x-w/2)
        w=2*w
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
'''
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


def calcdisplacement(signals, currentframe, p1):
    disp = []
    for i in range(signals.shape[1]):
        dispxy = []
        x = p1[i, 0, 0]-signals[0, i, 0, 0]
        y = p1[i, 0, 1]-signals[0, i, 0, 1]
        dist = math.sqrt((x)**2 + (y)**2)
        # dispxy.append([dist])
        disp.append(dist)

    return disp


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

# mariam
# face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
# cap = cv2.VideoCapture("C:\\Users\\Mariam Alaa\\Downloads\\5518996\\sleep dataset\\sleep dataset\\1 cyc\\rgb.avi")

# maram
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\Maram\\Anaconda3\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'C:\\Users\\Maram\\Anaconda3\\Lib\\site-packages\\cv2\data\\haarcascade_eye.xml')
cap = cv2.VideoCapture(
    "C:\\Users\\Maram\\Desktop\\GP2\\5518996\\sleep dataset\\28 zwh3\\rgb.avi")
# C:\\Users\\Mariam Alaa\\Downloads\\5518996\\sleep dataset\\sleep dataset\\2 dhy\\newfile.avi
# C:\\Users\\Mariam Alaa\\Downloads\\5518996\\sleep dataset\\sleep dataset\\3 dhy\\rgb.avi
# D:\\breathing2\\breathing2.mp4
# C:\\Users\\Mariam Alaa\\Downloads\\5518996\\sleep dataset\\sleep dataset\\1 cyc\\rgb.avi
#cap = cv2.VideoCapture(0)

feature_params = dict(maxCorners=10, qualityLevel=0.3,
                      minDistance=10, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

# Take first frame and find corners in it
frameId = cap.get(1)  # current frame number

frameRate = cap.get(5)  # frame rate

dimensions = [348, 219, 1084, 1949]

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################


ret, old_frame = cap.read()
# old_frame=old_frame[dimensions[0]:dimensions[2],dimensions[1]+400:dimensions[3]]
# show_images(old_frame)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(old_gray, 1.3, 5)
dimensionschest = []
for (x, y, w, h) in faces:
    y = math.floor(y+h/2+h)
    x = math.floor(x-w)
    w = 3*w
    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    old_gray = old_gray[y:y+h, x:x+w]
    roi_color = old_frame[y:y+h, x:x+w]
    dimensionschest.append(x)
    dimensionschest.append(w)
    dimensionschest.append(y)
    dimensionschest.append(h)
    show_images([old_frame, old_gray, roi_color])

    ''' EYE DETECTION
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''

p0 = []

p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)


signals = []
disp = []
signals.append(p0)
signals = np.asarray(signals)
currentframe = 1
mask = np.zeros_like(old_frame)

start = datetime.datetime.utcnow()
while(ret):
    # if (datetime.datetime.utcnow()-start).total_seconds() >= 30:
    #     break
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    # print(ret)
    if(ret == False):
        break
    print(dimensionschest)
    frame = frame[dimensionschest[2]:dimensionschest[2]+dimensionschest[3],
                  dimensionschest[0]:dimensionschest[0]+dimensionschest[1]]
    if(frameId % math.floor(frameRate) == 0 or frameId % math.floor(frameRate) == math.floor(frameRate/2)):
        # frame=frame[dimensions[0]:dimensions[2],dimensions[1]+400:dimensions[3]]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

        disp.append(calcdisplacement(signals, currentframe, p1))
        currentframe += 1
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for k, (new, old) in enumerate(zip(good_new, good_old)):

            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            frame = cv.circle(frame, (b, a), 5, color, -1)

        #output = cv.add(frame, mask)
        old_gray = frame_gray.copy()
        p0 = p1
        cv.imshow("sparse optical flow", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break


disp = np.asarray(disp)


disp2 = disp.transpose()

show_images([old_frame])
f_s = 2
for i in range(disp2.shape[0]):
    plt.plot(range(disp2.shape[1]), disp2[i])
    print(signals[0, i])
    plt.show()
    # fourier transform to check priodicity
    X = disp2[i]-np.mean(disp2[i])
    X = fftpack.fft(X)
    freqs = fftpack.fftfreq(len(disp2[i])) * f_s
    psd = np.abs(X)**2
    # print(psd.shape)
    norm_psd = preprocessing.normalize(psd[:, np.newaxis], axis=0).ravel()
    fig, ax = plt.subplots()
    ax.stem(freqs, np.abs(norm_psd))
    plt.show()
    rate = 60*freqs[np.argmax(norm_psd[freqs > 0.1])]
    print("rate=", rate)

cap.release()
cv.destroyAllWindows()
cv2.destroyAllWindows()
# cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
