import cv2
import numpy as np
from numpy import linalg as LA
import scipy.ndimage
from scipy import signal

import matplotlib.pyplot as plt
def optical_flow_harris(frame, nxt,frame1):
#cap = cv2.VideoCapture('E:\\senior 2\\Dataset\\Dataset\\motion1.mp4')
#ret, frame1 = cap.read()
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    #dst = cv2.cornerHarris(gray,2,3,0.04)
    #dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    #print(dst>0.01*dst.max())
    #frame1[dst>0.01*dst.max()]=[1,1,1]
    #frame1[dst<0.01*dst.max()]=[0,0,0]
    #print(frame1.shape)
    #frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    #print(frame1.shape)
    #ret, frame2 = cap.read()
    #nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #fig, ax = plt.subplots()
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w=7

    mode = 'same'
    fx = signal.convolve2d(frame1, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(frame1, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(nxt, kernel_t, boundary='symm', mode=mode) +signal.convolve2d(frame1, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)
    k= np.zeros(frame1.shape)
    for i in range(w, frame1.shape[0]-w):
        for j in range(w, frame1.shape[1]-w):
            if(frame1[i,j]!=0):
                Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
                b = np.reshape(It, (It.shape[0],1))
                A = np.vstack((Ix, Iy)).T 
                #print(np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))))
                #print(np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))))
                if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A))))>10: # get velocity here
                    #print(nu) 
                    nu = np.matmul(np.linalg.pinv(A), b)
                    print(i,j)
                    u[i,j]=nu[0]
                    v[i,j]=nu[1]
    return (u,v)
# feature_params = dict(maxCorners=8, qualityLevel=0.01,
#                       minDistance=0, blockSize=3)

# lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
#     cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# color = (0, 255, 0)

# # Take first frame and find corners in it
# ret, old_frame = cap.read()

# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = []
# for i in range(int(old_gray.shape[0]/40)):
#     for j in range(int(old_gray.shape[1]/40)):
#         gridP = cv2.goodFeaturesToTrack(
#             old_gray[i*40: i*41, j*40:j*41], mask=None, **feature_params)
#         if gridP is None:
#             print("empty")
#         else:
#             gridP = np.asarray(gridP)
#             gridP[:, :, 0] += 40*i
#             gridP[:, :, 1] += 40*j
#             p0.extend(gridP)

# p0 = np.asarray(p0)
# mask = np.zeros_like(old_frame)
# #while(1):
# ret, frame1 = cap.read()
# prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# ret, frame2 = cap.read()
# nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

# kernel_x = np.array([[-1., 1.], [-1., 1.]])
# kernel_y = np.array([[-1., -1.], [1., 1.]])
# kernel_t = np.array([[1., 1.], [1., 1.]])
# w=7
# I1g = prev / 255 # normalize pixels
# nxt = nxt / 255
# mode = 'same'
# fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
# fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
# ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) +signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
# u = np.zeros(I1g.shape)
# v = np.zeros(I1g.shape)
# k= np.zeros(I1g.shape)
# # within window window_size * window_size
# for i in range(w, I1g.shape[0]-w):
#     for j in range(w, I1g.shape[1]-w):
#         Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
#         Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
#         It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
#         b = np.reshape(It, (It.shape[0],1))
#         A = np.vstack((Ix, Iy)).T 
#         print(np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))))
#         if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= t:
#             nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
#             print(nu) 
#             print(i,j)
#             u[i,j]=nu[0]
#             v[i,j]=nu[1]
#     if(i>13):
#         break       

# print('u',u)
# print('v',v)
# prev = nxt
