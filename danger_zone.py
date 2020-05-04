import cv2 as cv
import numpy as np

# cap = cv.VideoCapture(
#     "C:\\Users\\Maram\\Desktop\\GP2\\youssef trimmed\\side cover.mp4")

cap = cv.VideoCapture(
    "C:\\Users\\Maram\\Desktop\\GP2\\dataset\\danger\\zizi roll.mp4")

# cap = cv.VideoCapture(
#     "C:\\Users\\Maram\\Downloads\\highway.mp4")

# ret, old_frame = cap.read()
# old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# while(ret):

#     ret, frame = cap.read()
#     if(ret == False):
#         break
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     subtraction = frame-old_frame
#     # print(subtraction)
#     old_frame = frame
#     cv.imshow("frame subtraction", subtraction)
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break


fgbg = cv.createBackgroundSubtractorMOG2()
old_fg = 0
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    diff = cv.absdiff(fgmask, old_fg)
   # print(diff)
    # check adaptive thresholding
   # _, diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
#    diff = cv.medianBlur(fgmask, 3)
    diff = cv.medianBlur(diff, 3)
    old_fg = fgmask
    cv.imshow('frame', diff)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
