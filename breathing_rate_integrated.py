import cv2 as cv
import numpy as np
from skimage import data, io
import matplotlib.pyplot as plt
#from commonfunctions import *
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


def optical_flow_harris(nxt, prev, p0):
    shapex = prev.shape[0]
    shapey = prev.shape[1]
    # print(prev.shape)
    borderType = cv.BORDER_CONSTANT
    prev = cv.copyMakeBorder(prev, 14, 14, 14, 14, borderType)
    nxt = cv.copyMakeBorder(nxt, 14, 14, 14, 14, borderType)
    # print(prev.shape)
    prev2 = np.zeros(prev.shape)
    j = 0

    for i in range(len(p0)):
        j = j+1
        prev2[int(p0[i][0][0]), int(p0[i][0][1])] = 1
        # prev2
    # print("Sum:",np.sum(prev2))
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w = 7

    mode = 'same'
    fx = signal.convolve2d(prev, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(prev, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(nxt, kernel_t, boundary='symm', mode=mode) + \
        signal.convolve2d(prev, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(prev.shape)
    v = np.zeros(prev.shape)
    p1 = []
    p2 = []
    h = 0
    y = 0
    z = 0
    # print(covariance)
    u, s, vh = np.linalg.svd(covariance, full_matrices=True)
    return u, s


def projectData(X, U, K):
    Z = 0
    # TODO 3: Fill the function projectData(X,U,K)
    Z = (U.transpose())@(X.transpose())
    Z = Z.transpose()
    Z = Z[:, 0:K]
    return Z


def pca_pattern(X):
    normalized_X, mu = featureNormalize(X)
    #print("shape=", X.shape, "normalized=", normalized_X.shape)
    u, s = pca(normalized_X)
    #print("eigenvectors=", u.shape)
    pca_components = projectData(normalized_X, u, 5)
    #print("components", pca_components.shape)
    return pca_components


# C:\\Users\\Maram\\Desktop\\GP2\\5518996\\sleep dataset\\1 cyc\\rgb.avi
# C:\\Users\\Maram\\Desktop\\GP2\\Dataset\\Breathing\\sample breathing.mp4
# C:\\Users\\Mariam Alaa\\Downloads\\5518996\\sleep dataset\\sleep dataset\\1 cyc\\rgb.avi
# D:\breathing2
# D:\\GP\\good.mp4

def breathing_rate(video, feature_params, lk_params, results_file):
    output_file = open(results_file, "w+")
    cap = cv.VideoCapture(video)
    # Take first frame and find corners in it
    frameId = cap.get(1)  # current frame number

    frameRate = cap.get(5)  # frame rate

    ret, old_frame = cap.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    # print(old_gray.shape)
    p0 = []

    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # print(p0[0])
    p0 = np.flip(p0, axis=2)
    # print(p0[0])

    signals = []
    disp = []
    signals.append(p0)
    signals = np.asarray(signals)
    currentframe = 1
    mask = np.zeros_like(old_frame)
    frames_count = 0
    calculated = 0
    prev_rates = []
    while(ret):

        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if(ret == False):
            break
        if(frameId % math.floor(frameRate) == 0 or frameId % math.floor(frameRate) == math.floor(frameRate/2)):
            # frame=frame[dimensions[0]:dimensions[2],dimensions[1]+400:dimensions[3]]
            frames_count += 1
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # p1, st, err = cv.calcOpticalFlowPyrLK(
            #     old_gray, frame_gray, p0, None, **lk_params)

            u, v, p1 = optical_flow_harris(frame_gray, old_gray, p0)
            p1 = np.asarray(p1)

            if frames_count == 1:
                disp.append(calcdisplacement(signals, currentframe, p1))
                disp = np.asarray(disp)
            else:
                disp = np.vstack((disp, calcdisplacement(
                    signals, currentframe, p1)))
            if frames_count >= 60 and frames_count % 2 == 0:
                # print(frameId)
                components_rates = get_rates(disp)
                if frames_count == 60:
                    prev_rates.append(components_rates[0, 0])
                    output_file.write(str(components_rates[0, 0])+"\n")
                else:
                    rates_diff = np.absolute(
                        components_rates[:, 0]-prev_rates[-1])
                    prev_rates.append(
                        components_rates[np.argmin(rates_diff), 0])
                    output_file.write(str(components_rates[np.argmin(rates_diff), 0])+"\n"
                                      )
                disp = disp[2:, :]
                calculated += 1

            currentframe += 1

            output = cv.add(frame, mask)
            old_gray = frame_gray.copy()
            p0 = p1
            cv.imshow("sparse optical flow", output)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    print(prev_rates)
    cap.release()
    cv.destroyAllWindows()


feature_params = dict(maxCorners=50, qualityLevel=0.05,
                      minDistance=30, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# cap = cv.VideoCapture(
#     "C:\\Users\\Maram\\Desktop\\GP2\\labeled dataset\\test\\4.avi")
breathing_rate("C:\\Users\\Maram\\Desktop\\GP2\\labeled dataset\\test\\baby.mp4",
               feature_params, lk_params, "results.txt")
