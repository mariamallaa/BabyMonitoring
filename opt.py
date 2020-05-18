import cv2
import numpy as np
from numpy import linalg as LA
import scipy.ndimage
from scipy import signal
import math

import matplotlib.pyplot as plt

def optical_flow_harris( nxt,prev,p0):
    
#cap = cv2.VideoCapture('E:\\senior 2\\Dataset\\Dataset\\motion1.mp4')
#ret, frame1 = cap.read()
   # gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    #gray = np.float32(gray)
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
    prev2 = np.zeros(prev.shape)
    for i in range(len(p0)):
        prev2[int(p0[i][0][0]),int(p0[i][0][1])]=1
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w=7

    mode = 'same'
    fx = signal.convolve2d(prev, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(prev, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(nxt, kernel_t, boundary='symm', mode=mode) +signal.convolve2d(prev, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(prev.shape)
    v = np.zeros(prev.shape)
    p1=[]
    p2=[]
    h=0
    for i in range(w, prev.shape[0]-w):
        for j in range(w, prev.shape[1]-w):
            #i=frame1[k,2]
            #j=frame1[k,1]
           # if frame[i,j]==1:
            
             if(prev2[i,j]==1):
                Ix = fx[i-w:i+w+1, j-w:j+w+1]
                Iy = fy[i-w:i+w+1, j-w:j+w+1]
                It = ft[i-w:i+w+1, j-w:j+w+1]
                Ix=np.reshape(Ix,225).T
                Iy=np.reshape(Iy,225).T
                It=np.reshape(It,225).T
            
                b = -It
                A = np.array([Ix,Iy]).T 
                
                #print(np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))))
                #print(np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))))
                # get velocity here
                    #print(nu) 
                #if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= 1e-2:
                nu = np.matmul(np.linalg.pinv(A), b)
                
                
                u[i,j]=nu[0]
                v[i,j]=nu[1]
                np_arr1 = np.array([u[i,j]*math.cos(v[i,j])])
                np_arr2= np.array(u[i,j]*math.sin(v[i,j]))
                p1.append([[u[i,j]*math.cos(v[i,j])],[u[i,j]*math.sin(v[i,j])]])
                p2.append([[u[i,j]*math.cos(v[i,j])+p0[h][0][0]],[u[i,j]*math.sin(v[i,j])+ p0[h][0][1]]])
                h=h+1
               
   
        
    
    print(p2)
    return (u,v,p2)


cap = cv2.VideoCapture('E:\\senior 2\\Dataset\\Dataset\\breathing2.mp4')
feature_params = dict(maxCorners=8, qualityLevel=0.01,
                      minDistance=0, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)
ret, frame1 = cap.read()
prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
p0 = []
for i in range(int(prev.shape[0]/40)):
    for j in range(int(prev.shape[1]/40)):
        gridP = cv2.goodFeaturesToTrack(
            prev[i*40: i*41, j*40:j*41], mask=None, **feature_params)
        if gridP is None:
            print("empty")
        else:
            gridP = np.asarray(gridP)
            gridP[:, :, 0] += 40*i
            gridP[:, :, 1] += 40*j
            p0.extend(gridP)

p0 = np.asarray(p0)
print(len(p0))




#gray = np.float32(prev)
#dst = cv2.cornerHarris(gray,2,3,0.04)
#dst = cv2.dilate(dst,None)
#frame1[dst>0.1*dst.max()]=1

#frame1[dst<0.04*dst.max()]=[0,0,0]
#for i in range(frame1.shape[0]):
#    for j in range(frame1.shape[1]):
#        if(frame1[i,j]==1):


#prev2 = cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)

while(1):
    ret, frame2 = cap.read()
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    u = np.zeros(prev.shape)
    v = np.zeros(prev.shape)
    p2=[]
    u,v,p2=optical_flow_harris(nxt,prev,p0)

    


    # with open('outfile.txt','w') as f:
    #     for line in p0:
    #         np.savetxt(f, line, fmt='%.2f')
    # hsv[...,0] = v*180/np.pi/2
    # hsv[...,2] = cv2.normalize(u,None,0,255,cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # cv2.imshow('frame2',bgr)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     cv2.destroyAllWindows()  
    # elif k == ord('s'):
    #     cv2.imwrite('opticalfb.png',frame2)
    #     cv2.imwrite('opticalhsv.png',bgr)

    prev=nxt
    
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
