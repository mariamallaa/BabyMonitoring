import math
import cv2 as cv
import numpy as np
from scipy import fftpack
from scipy.stats import entropy
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA


class breathing_rate:
    def __init__(self, feature_params, lk_params, age=0):
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.age = age
        self.min_rate = 0
        self.max_rate = 0
        self.set_normal_range()
        self.frames_count = 0
        self.old_interest_points = None
        self.signals = []
        self.disp = []
        self.prev_rates = []
        self.state = None

    def set_age(self, age):
        self.age = age
        self.set_normal_range()

    def set_normal_range(self):

        if self.age >= 0 and self.age < 1:
            self.min_rate = 30
            self.max_rate = 60
        elif self.age >= 1 and self.age < 2:
            self.min_rate = 24
            self.max_rate = 40

        elif self.age >= 2 and self.age < 6:
            self.min_rate = 17
            self.max_rate = 34
        elif self.age >= 6 and self.age < 18:
            self.min_rate = 18
            self.max_rate = 30
        elif self.age > 18:
            self.min_rate = 12
            self.max_rate = 25

    def calcdisplacement(self, signals, currentframe, new_interest_pts):
        disp = []
        for i in range(new_interest_pts.shape[0]):

            x = new_interest_pts[i, 0, 0]-signals[0, i, 0, 0]
            y = new_interest_pts[i, 0, 1]-signals[0, i, 0, 1]
            dist = math.sqrt((x)**2 + (y)**2)
            disp.append(dist)

        return disp

    def remove_noise(self, signals):
        filtered_signals = []
        for i in range(signals.shape[0]):
            nsamples = len(signals[i])
            if self.age >= 1.5:
                b, a = butter(5, [0.2, 0.6], btype='band')
            else:
                b, a = butter(5, [0.4, 0.8], btype='band')
            filtered = lfilter(b, a, signals[i])
            filtered_signals.append(filtered)

        filtered_signals = np.asarray(filtered_signals)
        filtered_signals = np.transpose(filtered_signals)
        return filtered_signals

    def get_components_pca(self, signals):
        pca = PCA(n_components=5)
        pca_components = pca.fit_transform(signals)
        return pca.explained_variance_, pca_components

    def get_rates(self, disp):
        disp2 = disp.transpose()
        #print(disp2.shape)
        f_s = 2
        differences = []
        for i in range(disp2.shape[0]):
            differences.append(np.max(np.diff(disp2[i])))
        differences = np.asarray(differences)
        differences = np.argsort(differences)
        length = len(differences)
        differences = differences[int(0.25*length):int(0.75*length)+1]
        disp2 = disp2[differences]
        # print(disp2.shape)

        filtered_signals = self.remove_noise(disp2)
        #("filtered", filtered_signals.shape)
        explained_variance, components = self.get_components_pca(
            filtered_signals)

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
        # print(rates)
        return rates

    def featureNormalize(self, X):
        normalized_X = X
        mu = 0
        mu = np.mean(X, axis=0)
        normalized_X = X-mu

        return normalized_X, mu

    def motion_detection(self, old_frame, curr_frame):
        diff = cv.absdiff(curr_frame, old_frame)
        _, diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
        diff = cv.medianBlur(diff, 3)
        if np.sum(diff):
            #print("motion detected")
            return True

        return False

    def get_breathing_rate(self):
        if len(self.prev_rates) == 0:
            return None, None
        else:
            return self.prev_rates[-1], self.state

    def estimate_breathing_rate(self, frame):
        #print("Received frame", self.age)
        if self.frames_count == 0:
            self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.old_interest_points = cv.goodFeaturesToTrack(
                self.old_gray, mask=None, **self.feature_params)
            self.signals.append(self.old_interest_points)
            self.signals = np.asarray(self.signals)
            self.frames_count += 1
            #self.old_interest_points = np.flip(self.old_interest_points, axis=2)

        else:

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            new_interest_pts, st, err = cv.calcOpticalFlowPyrLK(
                self.old_gray, frame_gray, self.old_interest_points, None, **self.lk_params)

            # new_interest_pts = optical_flow_harris(frame_gray, old_gray, old_interest_points)
            # new_interest_pts = np.asarray(new_interest_pts)

            if self.frames_count == 1:
                self.disp.append(self.calcdisplacement(
                    self.signals, self.frames_count, new_interest_pts))
                self.disp = np.asarray(self.disp)
            else:
                self.disp = np.vstack((self.disp, self.calcdisplacement(
                    self.signals, self.frames_count, new_interest_pts)))
            if self.frames_count >= 60 and self.frames_count % 2 == 0:

                components_rates = self.get_rates(
                    self.disp)

                if self.frames_count == 60:
                    components_rates = components_rates[components_rates[:, 0]
                                                        >= self.min_rate]
                    components_rates = components_rates[components_rates[:, 3].argsort()[
                        ::-1]]
                    current_rate = components_rates[0, 0]

                else:

                    window_length = min(3, len(self.prev_rates))
                    window_avg = sum(
                        self.prev_rates[-window_length:])/window_length
                    lowest_uncertainty = components_rates[0, 0]
                    #print("best uncertainty", lowest_uncertainty)
                    highest_variance = components_rates[components_rates[:, 3].argsort()[
                        ::-1]][0, 0]
                    #print("best variance", highest_variance)

                    # if np.absolute(lowest_uncertainty-self.prev_rates[-1]) < np.absolute(highest_variance-self.prev_rates[-1]):
                    if np.absolute(lowest_uncertainty-window_avg) < np.absolute(highest_variance-window_avg):
                        current_rate = lowest_uncertainty
                    else:
                        current_rate = highest_variance

                self.prev_rates.append(current_rate)
                #print(current_rate)
                if current_rate < self.min_rate or current_rate > self.max_rate:
                    #print("DANGER")
                    self.state="Danger"
                else:
                    self.state="Normal"

                self.disp = self.disp[2:, :]

            self.old_gray = frame_gray.copy()
            self.frames_count += 1
