from kivy.uix.screenmanager import Screen
import time
from kivy.uix.image import Image
from kivy.config import Config
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
import cv2 as cv
import pygame
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
from numpy import linalg as LA
import scipy.ndimage
from scipy import signal
import math

def optical_flow_harris_old(nxt, prev, p0):
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
    for i in range(w, prev.shape[0]):
        for j in range(w, prev.shape[1]):
            if(prev2[i-w, j-w] == 1):
                # print("p")
                # print(i, j)
                l = i-w
                m = j-w
                z += 1
                y = y+1
                Ix = fx[i-w:i+w+1, j-w:j+w+1]
                Iy = fy[i-w:i+w+1, j-w:j+w+1]
                It = ft[i-w:i+w+1, j-w:j+w+1]
                Ix = np.reshape(Ix, 225).T
                Iy = np.reshape(Iy, 225).T
                It = np.reshape(It, 225).T
                b = -It
                A = np.array([Ix, Iy]).T
                nu = np.matmul(np.linalg.pinv(A), b)
                u[i-w, j-w] = nu[0]
                v[i-w, j-w] = nu[1]
                np_arr1 = np.array([u[i, j]*math.cos(v[i, j])])
                np_arr2 = np.array(u[i, j]*math.sin(v[i, j]))
                p1.append([[u[i-w, j-w]*math.cos(v[i-w, j-w])],
                           [u[i-w, j-w]*math.sin(v[i-w, j-w])]])
                if(u[i-w, j-w]*math.cos(v[i-w, j-w])+p0[h][0][0] < shapex and u[i-w, j-w]*math.cos(v[i-w, j-w])+p0[h][0][0] >= 0 and u[i-w, j-w]*math.sin(v[i-w, j-w]) + p0[h][0][1] < shapey and u[i-w, j-w]*math.sin(v[i-w, j-w]) + p0[h][0][1] >= 0):
                    p2.append([[u[i-w, j-w]*math.cos(v[i-w, j-w])+p0[h][0][0],
                                u[i-w, j-w]*math.sin(v[i-w, j-w]) + p0[h][0][1]]])
                else:

                    #print("out of range")
                    p2.append([[p0[h][0][0], p0[h][0][1]]])
                h = h+1

    # print("z", z)
    # print("negative:", p2[-3])
    # if z < 28:
    #     print("p0", p0)
    #     print(l, m
    #           )
    return p2


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
    # print(p1.shape)
    for i in range(p1.shape[0]):
        dispxy = []
        x = p1[i, 0, 0]-signals[0, i, 0, 0]
        y = p1[i, 0, 1]-signals[0, i, 0, 1]
        dist = math.sqrt((x)**2 + (y)**2)
        # dispxy.append([dist])
        disp.append(dist)

    return disp


def remove_noise(signals, lower, upper):
    filtered_signals = []
    for i in range(signals.shape[0]):
        nsamples = len(signals[i])
        #t = np.linspace(0, nsamples/2, nsamples, endpoint=False)
        # plt.plot(t, signals[i], label="Noisy")

        #b, a = butter(5, [lower, upper], btype='band')
        b, a = butter(5, [0.2, 0.6], btype='band')
        filtered = lfilter(b, a, signals[i])
        # plt.plot(t, filtered, label="filtered")
        # plt.legend(loc='upper left')
        # plt.show()
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


def get_rates(disp, lower, upper):
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

    filtered_signals = remove_noise(disp2, lower, upper)
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
    # TODO 1: Fill the function featureNormalize(X)
    mu = np.mean(X, axis=0)
    normalized_X = X-mu

    return normalized_X, mu


def pca(X):
    # TODO 2: Fill the function PCA(X)
    covariance = np.cov(X.transpose())
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
    print("shape=", X.shape, "normalized=", normalized_X.shape)
    u, s = pca(normalized_X)
    print("eigenvectors=", u.shape)
    pca_components = projectData(normalized_X, u, 5)
    print("components", pca_components.shape)
    return pca_components

def breathing_rate(video, feature_params, lk_params, results_file, age):
    minRate = 0
    maxRate = 0
    if age >= 0 and age < 1:
        minRate = 30
        maxRate = 60
    elif age >= 1 and age < 3:
        minRate = 24
        maxRate = 40
    elif age >= 3 and age < 6:
        minRate = 22
        maxRate = 34
    elif age >= 6 and age < 18:
        minRate = 18
        maxRate = 30
    elif age > 18:
        minRate = 12
        maxRate = 25

    output_file = open(results_file, "w+")
    cap = cv.VideoCapture(video)
    # Take first frame and find corners in it
    frameId = cap.get(1)  # current frame number

    frameRate = cap.get(5)  # frame rate

    ret, old_frame = cap.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    print(old_gray.shape)
    p0 = []

    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    p0 = np.flip(p0, axis=2)

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

            p1 = optical_flow_harris(frame_gray, old_gray, p0)
            p1 = np.asarray(p1)

            # print("new positions", p1.shape)
            if frames_count == 1:
                disp.append(calcdisplacement(
                    signals, currentframe, p1))
                disp = np.asarray(disp)
            else:
                disp = np.vstack((disp, calcdisplacement(
                    signals, currentframe, p1)))
            if frames_count >= 60 and frames_count % 2 == 0:
                # print(frameId)
                components_rates = get_rates(
                    disp, (minRate)/60, (maxRate+7)/60)

                if frames_count == 60:
                    components_rates = components_rates[components_rates[:, 3].argsort()[
                        ::-1]]
                    current_rate = components_rates[0, 0]
                    #prev_rates.append(components_rates[0, 0])
                    #output_file.write(str(components_rates[0, 0])+"\n")
                else:
                    #prev_avg = sum(prev_rates)/len(prev_rates)
                    lowest_uncertainty = components_rates[0, 0]
                    print("best uncertainty", lowest_uncertainty)
                    highest_variance = components_rates[components_rates[:, 3].argsort()[
                        ::-1]][0, 0]
                    print("best variance", highest_variance)
                    if np.absolute(lowest_uncertainty-prev_rates[-1]) < np.absolute(highest_variance-prev_rates[-1]):
                        current_rate = lowest_uncertainty
                    else:
                        current_rate = highest_variance

                    # rates_diff = np.absolute(
                    #     components_rates[:, 0]-prev_avg)
                    # current_rate = components_rates[np.argmin(rates_diff), 0]

                prev_rates.append(current_rate)
                print(current_rate)
                if current_rate < minRate or current_rate > maxRate:
                    #print("DANGER")
                    pygame.mixer.music.play(-1)
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
                mask = cv.line(mask, (int(b), int(a)),
                               (int(d), int(c)), color, 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color, -1)
                p0 = p1

            cv.imshow("sparse optical flow", output)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                pygame.mixer.pause()
                break

    
    print(prev_rates)
    cap.release()
    cv.destroyAllWindows()


class SleepingScreen(Screen):

    def __init__(self, **kwargs):
        super(SleepingScreen, self).__init__(**kwargs)
        self.label = Label(text=" ",color=(1,0,0,1), font_size=(20),size_hint=(0.2,0.1), pos_hint={"center_x":0.5, "center_y":0.9})
        self.add_widget(self.label)
    
    def check_user_input(self,age):

        if age == '':
            self.label.text= "Please re-enter !"
            return self.__init__()

        elif age.isdigit() == False:
            self.label.text= "Please re-enter !"
            return self.__init__()
        
        else :
            self.detect_breathing_rate(age)

    def detect_breathing_rate(self,age):

        ######################initialize alarm##################
        pygame.mixer.init()
        pygame.mixer.music.load('fire-truck-air-horn_daniel-simion.wav')

        ############Start detecting breathing rate##############
        feature_params = dict(maxCorners=100, qualityLevel=0.05,
                      minDistance=20, blockSize=3)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        ##################need to get baby's age#######################
        breathing_rate("6.avi",
               feature_params, lk_params, "results.txt", int(age))

        
        
       

    
        
