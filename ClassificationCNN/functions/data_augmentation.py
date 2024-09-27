import random
import numpy as np


def time_shift(samples):
    samples = samples.flatten()
    shift = int(len(samples) * 0.4)  # Max shift (0.4)
    random_shift = random.randint(0, shift)  # Random number between 0 and 0.4*len(samples)
    data_roll = np.roll(samples, random_shift)
    return data_roll


def masking(samples):
    num_mask = 2
    freq_masking_max_percentage = 0.10
    time_masking_max_percentage = 0.10
    spec = samples
    mean_value = spec.mean()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape[1], spec.shape[1]
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = mean_value

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = mean_value
    return spec
