import time
import numpy as np
import pandas as pd
import librosa
from .isolator import isolator


def get_best_threshold(lim_start, lim_end, original, audio_dir):
    threshold_values = np.arange(lim_start, lim_end, 0.0005)
    max_thres_value = threshold_values[0]
    max_mean = 1
    min_rel_std = 100000
    keys_s = list('1234567890QWERTYUIOPASDFGHJKLZXCVBNMÑ+-')
    for threshold_value in threshold_values:
        lengths = []
        for key in keys_s:
            sample, sr = librosa.load(f'{audio_dir}{"audio_" if original else ""}{key}.wav')
            lengths.append(len(isolator(sample, sr, 1024, 225, 2200, 11000, threshold_value)))
        print(f'Thres: {threshold_value:.4f}' , end=' ')
        mean = np.mean(lengths)
        rel_std = (np.std(lengths)/np.mean(lengths))*100
        if 38 <= mean <= 42:
            print(lengths)
        if rel_std < min_rel_std:
            min_rel_std = rel_std
            max_thres_value = threshold_value
            max_mean = mean
        print(f'threshold: {threshold_value:.4f}  /  mean: {mean:.2f}  /  rel std dev: {((np.std(lengths)/mean) *100):.3f}%  /  max: {np.max(lengths)}  /  min: {np.min(lengths)}')
    print(f'Min relative std: {min_rel_std:.4f} threshold value:  {max_thres_value:.4f} mean: {max_mean:.4f}')
    return max_thres_value