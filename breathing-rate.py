import cv2 as cv
import numpy as np
from skimage import data, io
import matplotlib.pyplot as plt
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


def calcdisplacement(signals, currentframe, p1):
    disp = []
    for i in range(signals.shape[1]):
        dispxy = []
        x = p1[i, 0, 0]-signals[0, i, 0, 0]
        y = p1[i, 0, 1]-signals[0, i, 0, 1]
        dist = math.sqrt((x)**2 + (y)**2)
        disp.append(dist)

    return disp


def remove_noise(signals):
    filtered_signals = []
    for i in range(signals.shape[0]):
        nsamples = len(signals[i])
        b, a = butter(5, [0.2, 0.6], btype='band')
        filtered = lfilter(b, a, signals[i])
        filtered_signals.append(filtered)

    filtered_signals = np.asarray(filtered_signals)
    filtered_signals = np.transpose(filtered_signals)
    return filtered_signals


def get_components_pca(signals):
    pca = PCA(n_components=5)
    pca_components = pca.fit_transform(signals)
    return pca_components


def get_components_ica(signals):
    ica = FastICA(n_components=5)
    ica_components = ica.fit_transform(signals)
    return ica_components


def get_rates(disp):
    disp2 = disp.transpose()
    f_s = 2
    differences = []
    for i in range(disp2.shape[0]):
        differences.append(np.max(np.diff(disp2[i])))
    differences = np.asarray(differences)
    differences = np.argsort(differences)
    length = len(differences)
    differences = differences[int(0.25*length):int(0.75*length)+1]
    disp2 = disp2[differences]

    filtered_signals = remove_noise(disp2)
    components = get_components_pca(filtered_signals)
    rates = []
    for i in range(components.shape[1]):

        X = fftpack.fft(components[:, i])
        freqs = fftpack.fftfreq(len(components[:, i])) * f_s
        psd = np.abs(X)**2
        psd = psd/np.sum(psd)
        uncertainty = entropy(psd, base=2)
        variance = np.var(psd)

        offset = next((i for i, x in enumerate(freqs) if x > 0.15), None)
        rate = 60*freqs[np.argmax(psd[freqs > 0.15])+offset]
        rates.append([rate, uncertainty, variance])

    rates = np.asarray(rates)
    rates = rates[rates[:, 1].argsort()]
    print(rates)
    return rates


cap = cv.VideoCapture(
    "C:\\Users\\Maram\\Desktop\\GP2\\labeled dataset\\test\\covered.mp4")


feature_params = dict(maxCorners=50, qualityLevel=0.05,
                      minDistance=30, blockSize=3)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = (0, 255, 0)

# Take first frame and find corners in it
frameId = cap.get(1)  # current frame number

frameRate = cap.get(5)  # frame rate

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = []

p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

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
        if frames_count == 1:
            disp.append(calcdisplacement(signals, currentframe, p1))
            disp = np.asarray(disp)
        else:
            disp = np.vstack((disp, calcdisplacement(
                signals, currentframe, p1)))
        if frames_count >= 60 and frames_count % 2 == 0:
          
            components_rates = get_rates(disp)
            if frames_count == 60:
                prev_rates.append(components_rates[0, 0])
            else:
                rates_diff = np.absolute(
                    components_rates[:, 0]-prev_rates[-1])
                prev_rates.append(
                    components_rates[np.argmin(rates_diff), 0])
            disp = disp[2:, :]
            calculated += 1

        currentframe += 1
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for k, (new, old) in enumerate(zip(good_new, good_old)):

            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            frame = cv.circle(frame, (b, a), 5, color, -1)

        output = cv.add(frame, mask)
        old_gray = frame_gray.copy()
        p0 = p1
        cv.imshow("sparse optical flow", output)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

print(prev_rates)
cap.release()
cv.destroyAllWindows()
