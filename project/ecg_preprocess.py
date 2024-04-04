#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:54:19 2019

@author: shih3801
"""
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, lfilter, lfilter_zi, filtfilt, sosfreqz, resample
from utils import hamilton_detector, christov_detector, findpeaks, engzee_detector
from ecg_detectors.ecgdetectors import Detectors, MWA, panPeakDetect, searchBack


class Ecg_preprocess():
    def __init__(self, data, lowcut, highcut, fs, order=5):
        self.data = data
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        pass

    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        sos = butter(self.order, [low, high], analog=False, btype="band", output="sos")
        return sos


    def butter_bandpass_filter(self):
        sos = self.butter_bandpass()
        y = sosfilt(sos,
                    self.data)  # Filter data along one dimension using cascaded second-order sections. Using lfilter for each second-order section.
        return y

    def butter_bandpass_filter_once(self):
        sos = self.butter_bandpass()
        # Apply the filter to data. Use lfilter_zi to choose the initial condition of the filter.
        zi = sosfilt_zi(sos)
        z, _ = sosfilt(sos, self.data, zi=zi * self.data[0])
        return sos, z, zi

    def butter_bandpass_filter_again(sos, z, zi):
        # Apply the filter again, to have a result filtered at an order the same as filtfilt.
        z2, _ = sosfilt(sos, z, zi=zi * z[0])
        return z2

    def butter_bandpass_forward_backward_filter(self):
        sos = self.butter_bandpass()
        y = sosfiltfilt(sos,
                        self.data)  # Apply a digital filter forward and backward to a signal.This function applies a linear digital filter twice, once forward and once backwards. The combined filter has zero phase and a filter order twice that of the original.
        return y

    def pan_tompkins_detector(self, raw_ecg, mwa, N):
        """ detect r peaks
        raw_ecg: ecg values
        mwa: after using nn.convolve of original ecg
        N: Moving window"""
        N = int(N / 100 * self.fs)  # N:100 = 50/100*200
        mwa_peaks = panPeakDetect(mwa, self.fs)
        r_peaks = searchBack(mwa_peaks, raw_ecg, N)
        return r_peaks

    def pan_tompkins_alg(self, denoise_y):
        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(denoise_y)

        # Squaring - intensifies values received in derivative.
        # This helps restrict false positives caused by T waves with higher than usual spectral energies..
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integration_window = 50  # Change proportionally when adjusting frequency (in samples)
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window))

        # Fiducial mark - peak detection on integrated measurements.
        rpeaks = self.pan_tompkins_detector(self.data, integrated_ecg_measurements, integration_window)
        return rpeaks

    def Nmaxelements(self, list1, N):
        """Get the N maximum values in list1"""
        final_list = []
        for i in range(0, N):
            max1 = 0
            for j in range(len(list1)):
                if list1[j] > max1:
                    max1 = list1[j];
            list1.remove(max1);
            final_list.append(max1)
        return final_list


    def clean_repeat_rpeaks(self, rpeaks):
        """Delete too closed rpeaks"""
        new_r = []
        new_r.append(rpeaks[0])
        for i in range(rpeaks.size - 1):
            if rpeaks[i + 1] - rpeaks[i] > 40:
                new_r.append(rpeaks[i + 1])
        new_r = np.array(new_r)
        return new_r

    def tailor_rpeaks_length(self, data, peaks):
        """Deleted fake r-peaks
        data: original ecg data
        rpeaks: raw rpeaks
        return:
        rpeaks_dict: pure rpeaks dictionary, with rpeaks_dict{index: rpeaks_values}
        Q: Q raw dictionary
        S: S raw dictionary
        """
        min_left = 0
        min_right = 0
        amplify = []
        R_indice_len = {}
        Q = {}
        S = {}

        rpeaks = self.clean_repeat_rpeaks(peaks)
        # find the best r-peaks values and index
        for i in rpeaks:
            start = i - 3
            end = i + 3
            if i - 3 < 0:
                start = 0
            if i + 3 > len(data):
                end = len(data) - 1
            value = np.max(data[start: end])
            indice = np.argmax(data[start: end]) + start  # get the index of max in the range
            posi_left = indice - 30
            posi_right = indice + 50
            if posi_left <= 0:
                min_right = np.min(data[indice: posi_right])
                posi_min_right = np.argmin(data[indice: posi_right]) + indice
                length = value - min_right
                # record S values
                S[posi_min_right] = min_right

            if posi_right >= len(data):
                min_left = np.min(data[posi_left: indice])
                posi_min_left = np.argmin(data[posi_left: indice]) + posi_left
                length = value - min_left
                # record Q values
                Q[posi_min_left] = min_left

            if posi_left > 0 and posi_right < len(data):
                min_right = np.min(data[indice: posi_right])
                posi_min_right = np.argmin(data[indice: posi_right]) + indice
                min_left = np.min(data[posi_left: indice])
                posi_min_left = np.argmin(data[posi_left: indice]) + posi_left
                # length = value - min_left  origin

                length = value - min_left if value - min_right < (value - min_left) else (value - min_right)
                # record Q and S values
                Q[posi_min_left] = min_left
                S[posi_min_right] = min_right

            R_indice_len[indice] = length
            amplify.append(length)

        # delete the bias of rpeaks values and index
        amplify = self.Nmaxelements(amplify, 3)
        compare_length = sum(amplify) / 3
        rpeaks_dict = {i: R_indice_len[i] for i in R_indice_len
                       if R_indice_len[i] > 0.5 * compare_length}
        return rpeaks_dict, Q, S

    def Q_S_detector(self, rpeaks, Q_dict, S_dict):
        """clean unwanted Q and S values
        rpeaks: index of r-peaks
        Q_dict: record the original Q values
        S_dict: record the original S values
        """
        # delete the started errors of Q and S
        first_rpeaks = rpeaks[0]
        a = np.array(list(Q_dict.keys()))
        clean_q = a[a >= (first_rpeaks - 30)]
        c = np.array(list(S_dict.keys()))
        clean_s = c[c > first_rpeaks]
        #print('rpeaks is ', rpeaks)
        #print('clean_s is ', clean_s)
        if len(rpeaks) == 1:
            Q = {first_rpeaks-1:0}
            S = {first_rpeaks+1:0}
            return Q, S
        Q = {i: Q_dict[i] for i in Q_dict if i in clean_q}
        S = {i: S_dict[i] for i in S_dict if i in clean_s}

        # delete unwanted Q,S values in middle
        new_Q = []
        new_Q.append(clean_q[0])
        if len(Q) == rpeaks.size + 1:
            for i in range(1, rpeaks.size):
                if rpeaks[i] > clean_q[i] and rpeaks[i] < clean_q[i+1]:
                    new_Q.append(clean_q[i])
                elif rpeaks[i] > clean_q[i] and rpeaks[i] > clean_q[i+1]:
                    new_Q.append(clean_q[i+1])
            Q = {i: Q_dict[i] for i in Q_dict if i in new_Q}
        elif len(Q) > rpeaks.size + 1:
            j = 1
            for i in range(1, rpeaks.size):
                while j < clean_q.size - 1:
                    if rpeaks[i] > clean_q[j] and rpeaks[i] > clean_q[j+1]:
                        j += 1
                    elif rpeaks[i] > clean_q[j] and rpeaks[i] < clean_q[j+1]:
                        new_Q.append(clean_q[j])
                        break
            if rpeaks.size == len(new_Q)+1:
                new_Q.append(clean_q[j])
            Q = {i: Q_dict[i] for i in Q_dict if i in new_Q}
        new_S = []
        new_S.append(clean_s[0])
        if len(S) == rpeaks.size + 1:
            k = 0
            for i in range(1, rpeaks.size):
                if i + k >= clean_s.size:
                    break
                if rpeaks[i] > clean_s[i + k - 1] and (rpeaks[i] < clean_s[i + k]):
                    new_S.append(clean_s[i+k])

                elif rpeaks[i] > clean_s[i + k] and (rpeaks[i] < clean_s[i + k + 1]):
                    new_S.append(clean_s[i+k+1])
                    k += 1

            S = {i: S_dict[i] for i in S_dict if i in new_S}
        elif len(S) > rpeaks.size + 1:
            j = 1
            for i in range(1, rpeaks.size):
                while j < clean_s.size:
                    if rpeaks[i] > clean_s[j]:
                        j += 1
                    elif rpeaks[i] > clean_s[j - 1] and (rpeaks[i] < clean_s[j]):
                        new_S.append(clean_s[j])
                        break
            S = {i: S_dict[i] for i in S_dict if i in new_S}
        if len(S) < rpeaks.size - 2:
            j = 1
            for i in range(1, rpeaks.size):
                while j < clean_s.size:
                    if rpeaks[i] > clean_s[j]:
                        j += 1
                    elif rpeaks[i] > clean_s[j - 1] and (rpeaks[i] < clean_s[j]):
                        new_S.append(clean_s[j])
                        break
            S = {i: S_dict[i] for i in S_dict if i in new_S}

        return Q, S


    def T_detector(self, data, rpeaks):
        """Detect T waves
        rpeaks: index of r-peaks
        """
        T = {}
        sum_inter = 0
        if len(rpeaks) == 1:
            first_rpeaks = rpeaks[0]
            T = {first_rpeaks+10:0}
            return T
        for i in range(len(rpeaks)):
            if i < len(rpeaks) - 1:
                interval = rpeaks[i + 1] - rpeaks[i]
                sum_inter += interval
                inf = int(rpeaks[i] + 0.16 * interval)
                end = int(rpeaks[i] + 0.57 * interval)
                t_max = np.max(data[inf:end])
                t_index = np.argmax(data[inf:end]) + inf
                T[t_index] = t_max
            last_inter = sum_inter / (len(rpeaks) - 1)
            if i == len(rpeaks) - 1:
                inf = int(rpeaks[i] + 0.16 * last_inter)
                end = int(rpeaks[i] + 0.57 * last_inter)
                if end < len(data):
                    t_max = np.max(data[inf:end])
                    t_index = np.argmax(data[inf:end]) + inf
                    T[t_index] = t_max
        return T

    def P_detector(self, data, rpeaks):
        """Detect P waves
        rpeaks: index of r-peaks
        """
        P = {}
        for i in range(len(rpeaks)):
            # look for p wave
            inf = int(rpeaks[i] - 92)
            end = int(rpeaks[i] - 45)
            if inf > 0:
                p_max = np.max(data[inf:end])
                p_index = np.argmax(data[inf:end]) + inf
                P[p_index] = p_max
        return P
