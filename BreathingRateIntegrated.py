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

    borderType = cv.BORDER_CONSTANT
    prev = cv.copyMakeBorder(prev, 14, 14, 14, 14, borderType)
    nxt = cv.copyMakeBorder(nxt, 14, 14, 14, 14, borderType)

    prev2 = np.zeros(prev.shape)
    j = 0

    for i in range(len(p0)):
        j = j+1
        prev2[int(round(p0[i][0][0])), int(round(p0[i][0][1]))] = 1
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
    for u in range(len(p0)):

        i = int(p0[u][0][0])-w
        j = int(p0[u][0][1])-w

        if(i-w < 0 or j-w < 0):
            p2.append([[p0[u][0][0], p0[u][0][1]]])
            continue
        Ix = fx[i-w:i+w+1, j-w:j+w+1]
        Iy = fy[i-w:i+w+1, j-w:j+w+1]
        It = ft[i-w:i+w+1, j-w:j+w+1]
        Ix = np.reshape(Ix, 225).T
        Iy = np.reshape(Iy, 225).T
        It = np.reshape(It, 225).T

        b = -It
        A = np.array([Ix, Iy]).T

        nu = np.matmul(np.linalg.pinv(A), b)

        if(nu[0]*math.cos(nu[1])+p0[u][0][0] < shapex and nu[0]*math.cos(nu[1])+p0[u][0][0] >= 0 and nu[0]*math.sin(nu[1]) + p0[u][0][1] < shapey and nu[0]*math.sin(nu[1]) + p0[u][0][1] >= 0):
            p2.append([[nu[0]*math.cos(nu[1])+p0[u][0][0],
                        nu[0]*math.sin(nu[1]) + p0[u][0][1]]])
        else:
            print("out of range")
            p2.append([[p0[u][0][0], p0[u][0][1]]])
    return p2


def calcdisplacement(signals, currentframe, p1):
    disp = []
    for i in range(p1.shape[0]):
        dispxy = []
        x = p1[i, 0, 0]-signals[0, i, 0, 0]
        y = p1[i, 0, 1]-signals[0, i, 0, 1]
        dist = math.sqrt((x)**2 + (y)**2)
        disp.append(dist)

    return disp


def remove_noise(signals, age):
    filtered_signals = []
    for i in range(signals.shape[0]):
        nsamples = len(signals[i])
        if age >= 1.5:
            b, a = butter(5, [0.2, 0.6], btype='band')
        else:
            b, a = butter(5, [0.4, 0.8], btype='band')
        filtered = lfilter(b, a, signals[i])
        filtered_signals.append(filtered)

    filtered_signals = np.asarray(filtered_signals)
    filtered_signals = np.transpose(filtered_signals)
    return filtered_signals


def get_components_pca(signals):
    pca = PCA(n_components=5)
    pca_components = pca.fit_transform(signals)
    return pca.explained_variance_, pca_components


def get_components_ica(signals):
    ica = FastICA(n_components=5)
    ica_components = ica.fit_transform(signals)
    return ica_components


def get_rates(disp, age):
    disp2 = disp.transpose()
    print(disp2.shape)
    f_s = 2
    differences = []
    for i in range(disp2.shape[0]):
        differences.append(np.max(np.diff(disp2[i])))
    differences = np.asarray(differences)
    differences = np.argsort(differences)
    length = len(differences)
    differences = differences[int(0.25*length):int(0.75*length)+1]
    disp2 = disp2[differences]
    print(disp2.shape)

    filtered_signals = remove_noise(disp2, age)
    #("filtered", filtered_signals.shape)
    explained_variance, components = get_components_pca(filtered_signals)

    #components = pca_pattern(filtered_signals)

    rates = []
    for i in range(components.shape[1]):

        X = fftpack.fft(components[:, i])
        freqs = fftpack.fftfreq(len(components[:, i])) * f_s
        psd = np.abs(X)**2
        psd = psd/np.sum(psd)
        uncertainty = entropy(psd, base=2)
        #uncertainty = np.max(psd)/np.sum(psd)
        variance = np.var(psd)

        offset = next((i for i, x in enumerate(freqs) if x > 0.15), None)
        rate = 60*freqs[np.argmax(psd[freqs > 0.15])+offset]
        #uncertaintyModified = np.max(psd[freqs > 0.15])/np.sum(psd)

        rates.append([rate, uncertainty, variance,
                      explained_variance[i]]
                     )

    rates = np.asarray(rates)
    # print(rates.shape)
    rates = rates[rates[:, 1].argsort()]
    print(rates)
    return rates


def featureNormalize(X):
    normalized_X = X
    mu = 0
    mu = np.mean(X, axis=0)
    normalized_X = X-mu

    return normalized_X, mu


def pca(X):
    covariance = np.cov(X.transpose())
    u, s, vh = np.linalg.svd(covariance, full_matrices=True)
    return u, s


def projectData(X, U, K):
    Z = 0
    Z = (U.transpose())@(X.transpose())
    Z = Z.transpose()
    Z = Z[:, 0:K]
    return Z


def pca_pattern(X):
    normalized_X, mu = featureNormalize(X)

    u, s = pca(normalized_X)

    pca_components = projectData(normalized_X, u, 5)

    return pca_components


def breathing_rate(video, feature_params, lk_params, results_file, age):
    minRate = 0
    maxRate = 0
    if age >= 0 and age < 1:
        minRate = 30
        maxRate = 60
    elif age >= 1 and age < 2:
        minRate = 24
        maxRate = 40

    elif age >= 2 and age < 6:
        minRate = 17
        maxRate = 34
    elif age >= 6 and age < 18:
        minRate = 18
        maxRate = 30
    elif age > 18:
        minRate = 12
        maxRate = 25

    output_file = open(results_file, "w+")
    cap = cv.VideoCapture(video)
    frameId = cap.get(1)  # current frame number
    frameRate = cap.get(5)  # frame rate
    ret, old_frame = cap.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = []

    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    #p0 = np.flip(p0, axis=2)

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

            frames_count += 1
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            p1, st, err = cv.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params)

            # p1 = optical_flow_harris(frame_gray, old_gray, p0)
            # p1 = np.asarray(p1)

            if frames_count == 1:
                disp.append(calcdisplacement(
                    signals, currentframe, p1))
                disp = np.asarray(disp)
            else:
                disp = np.vstack((disp, calcdisplacement(
                    signals, currentframe, p1)))
            if frames_count >= 60 and frames_count % 2 == 0:

                components_rates = get_rates(
                    disp, age)

                if frames_count == 60:
                    components_rates = components_rates[components_rates[:, 0] >= minRate]
                    components_rates = components_rates[components_rates[:, 3].argsort()[
                        ::-1]]
                    current_rate = components_rates[0, 0]

                else:

                    window_length = min(3, len(prev_rates))
                    window_avg = sum(prev_rates[-window_length:])/window_length
                    lowest_uncertainty = components_rates[0, 0]
                    print("best uncertainty", lowest_uncertainty)
                    highest_variance = components_rates[components_rates[:, 3].argsort()[
                        ::-1]][0, 0]
                    print("best variance", highest_variance)

                    # if np.absolute(lowest_uncertainty-prev_rates[-1]) < np.absolute(highest_variance-prev_rates[-1]):
                    if np.absolute(lowest_uncertainty-window_avg) < np.absolute(highest_variance-window_avg):
                        current_rate = lowest_uncertainty
                    else:
                        current_rate = highest_variance

                prev_rates.append(current_rate)
                print(current_rate)
                if current_rate < minRate or current_rate > maxRate:
                    print("DANGER")
                output_file.write(str(current_rate)+"\n")

                disp = disp[2:, :]
                calculated += 1

            currentframe += 1

            output = cv.add(frame, mask)
            old_gray = frame_gray.copy()

            color = (0, 255, 0)
            for k, (new, old) in enumerate(zip(p1, p0)):

                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)),
                               (int(c), int(d)), color, 2)
                frame = cv.circle(frame, (int(b), int(a)), 5, color, -1)
                p0 = p1

            cv.imshow("sparse optical flow", output)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    print(prev_rates)
    cap.release()
    cv.destroyAllWindows()


feature_params = dict(maxCorners=100, qualityLevel=0.05,
                      minDistance=30, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# cap = cv.VideoCapture(
#     "C:\\Users\\Maram\\Desktop\\GP2\\labeled dataset\\test\\4.avi")
breathing_rate("C:\\Users\\Maram\\Desktop\\GP2\\labeled dataset\\test\\back dim.mp4",
               feature_params, lk_params, "results.txt", 0.9)
