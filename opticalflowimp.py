import cv2
import numpy as np
from numpy import linalg as LA
import scipy.ndimage
from scipy import signal
def optical_flow_harris(frame1, frame2):
#cap = cv2.VideoCapture('E:\\senior 2\\Dataset\\Dataset\\motion1.mp4')
#ret, frame1 = cap.read()
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    t=1e-2
    windowsize=15

    #while(1):
    #ret, frame2 = cap.read()
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w=7
    I1g = prev / 255 # normalize pixels
    I2g = nxt / 255
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) +signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    k= np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0],1))
            A = np.vstack((Ix, Iy)).T 
            print(np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))))
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= t:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                print(nu) 
                print(i,j)
                u[i,j]=nu[0]
                v[i,j]=nu[1]
    return(u,v)


# while(1):
#     ret, frame2 = cap.read()
#     nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#     h=fspecial_gauss(3,1)
#     prev=scipy.ndimage.correlate(prev, h, mode='constant').transpose()
#     nxt=scipy.ndimage.correlate(nxt, h, mode='constant').transpose()
#     mat=np.array([[-1,8,0,-8,1],[-1,8,0,-8,1]])
#     matrix=1/12*mat
#     hitmap=np.zeros(prev.shape)
#     Ix=np.convolve(prev,matrix,'same')
#     Iy=np.convolve(np.reshape(prev,prev.size),np.transpose(matrix),'same')
#     It=nxt-prev
#     It=np.reshape(It,prev.size)
#     X2=np.convolve(Ix**2,np.ones(windowsize),'same')
#     print(X2)
#     Y2=np.convolve(Iy**2,np.ones(windowsize),'same')
#     print(Y2)
#     XY=np.convolve(Ix*Iy,np.ones(windowsize),'same')
#     Xt=np.convolve(Ix*It,np.ones(windowsize),'same')
#     Yt=np.convolve(Iy*It,np.ones(windowsize),'same')
#     u=np.zeros(prev.shape)
#     v=np.zeros(prev.shape)
#     win=7
#     X2 = np.reshape(X2,(720,1280))
#     Y2 = np.reshape(Y2,(720,1280))
#     XY = np.reshape(XY,(720,1280))
#     Xt = np.reshape(Xt,(720,1280))
#     Yt = np.reshape(Yt,(720,1280))
   
#     # for i in range(w,int(frame1.shape[0]-w),int(windowsize)):
#     #    for j in range(w,int(frame1.shape[1]-w),int(windowsize)):
#     #         A=np.array([[X2[i,j],XY[i,j]],[XY[i,j],Y2[i,j]]])
#     #         B=(-1)*np.array([[Xt[i,j]],[Yt[i,j]]])
#     #         r=np.linalg.matrix_rank(A)
#     #         w, v = LA.eig(A)
#     #         if(np.amin(w)>t):
#     #             U = np.linalg.solve(A,B)
#     #             u[i,j]=U[0]
#     #             v[i,j]=U[1]
#     left=0
#     for i in range(0,prev.shape[0]):
#         for j in range(0,1273):
#             left = j-win
#             right = j+win
#             top = i-win
#             bottom = i+win
#             if (left<0):
#                 left=0
#             if(right>prev.shape[1]):
#                 right=prev.shape[1]
#             if(top<0):
#                 top=0
#             if(bottom>prev.shape[0]):
#                 bottom=prev.shape[0]
#             ws = (right-left+1)*(bottom-top+1)
           
#             A=np.array([[X2[i,j],XY[i,j]],[XY[i,j],Y2[i,j]]])
#             A=np.divide(A,ws)
#             print(A)
#             B=(-1)*np.array([[Xt[i,j]],[Yt[i,j]]])
            
#             B=np.divide(B,ws)
#             print(B)
#             r=np.linalg.matrix_rank(A)
#             w, v = LA.eig(A)
#             w=np.amin(w)
#             print(np.amin(w))
#             if(np.amin(w)>t):
#                 U = np.linalg.solve(A,B)
#                 u[i,j]=U[0]
#                 v[i,j]=U[1]
#                 print('true')
#                 print('u',u)
#                 print('v',v)


            
#     print('u',u)
#     print('v',v)
#     prev = nxt




