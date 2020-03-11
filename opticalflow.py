import cv2 as cv
import numpy as np

cap = cv.VideoCapture(
    "C:\\Users\\Maram\\Desktop\\GP2\\Dataset\\VID_20200127_102320.mp4")

feature_params = dict(maxCorners=8, qualityLevel=0.01,
                      minDistance=0, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

# Take first frame and find corners in it
ret, old_frame = cap.read()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = []
for i in range(int(old_gray.shape[0]/40)):
    for j in range(int(old_gray.shape[1]/40)):
        gridP = cv.goodFeaturesToTrack(
            old_gray[i*40: i*41, j*40:j*41], mask=None, **feature_params)
        if gridP is None:
            print("empty")
        else:
            gridP = np.asarray(gridP)
            gridP[:, :, 0] += 40*i
            gridP[:, :, 1] += 40*j
            p0.extend(gridP)

p0 = np.asarray(p0)
mask = np.zeros_like(old_frame)
while(1):
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        frame = cv.circle(frame, (a, b), 5, color, -1)
    img = cv.add(frame, mask)
    # cv.imshow('frame',img)
    output = cv.add(frame, mask)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # update interest points every frame
    # p0 = []
    # for i in range(int(old_gray.shape[0]/40)):
    #     for j in range(int(old_gray.shape[1]/40)):
    #         gridP = cv.goodFeaturesToTrack(
    #             old_gray[i*40: i*41, j*40:j*41], mask=None, **feature_params)
    #         if gridP is None:
    #             print("empty")
    #         else:
    #             gridP = np.asarray(gridP)
    #             gridP[:, :, 0] += 40*i
    #             gridP[:, :, 1] += 40*j
    #             p0.extend(gridP)

    # p0 = np.asarray(p0)
    cv.imshow("sparse optical flow", output)
   # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
