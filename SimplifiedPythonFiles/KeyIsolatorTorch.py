# imports
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import array
from pydub import AudioSegment, silence
from collections import deque
from detecta import detect_peaks
import pyloudnorm as pyln
import pandas as pd
import torch
import time

# Supporting Functions

# Waveform plotting function
def disp_waveform(signal, title='', sr=None, color='blue'):
    plt.figure(figsize=(7, 2))
    plt.title(title)
    plt.plot(signal.t().numpy(), color=color)
    if sr:
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
    plt.show()

# Finding key presses using waveform
def find_key_presses(waveform, res, waveform_threshold, waveform_max, threshold_background, history_size, remove_low_power, clear_previous=False):
    # Clear previous results
    if clear_previous:
        res.clear()
        waveform_threshold = torch.zeros_like(waveform)
        waveform_max = torch.zeros_like(waveform)
    
    # Starting values
    rb_begin = 0
    rb_average = 0.0
    rb_samples = torch.zeros(history_size)

    k = history_size
    que = deque(maxlen=k)

    samples = torch.abs(waveform)  # Taking absolute values like waveformAbs in C++
    n = len(samples)
    overall_loudness = 0
    len_ovr_loudness = 0
    for i in range(n):
        ii = i - k // 2
        if ii >= 0:
            rb_average *= len(rb_samples)
            rb_average -= rb_samples[rb_begin]
            acur = samples[i]
            rb_samples[rb_begin] = acur
            rb_average += acur
            rb_average /= len(rb_samples)
            rb_begin = (rb_begin + 1) % len(rb_samples)
        if i < k:
            # Handling initial filling of the deque
            while que and samples[i] >= samples[que[-1]]:
                que.pop()
            que.append(i)
        else:
            # Maintain the deque as a max-queue for the sliding window
            while que and que[0] <= i - k:
                que.popleft()

            # same code as if i<k
            while que and samples[i] >= samples[que[-1]]:
                que.pop()
            que.append(i)

            itest = i - k // 2
            if  k <= itest < n - k and que[0] == itest:
                acur = samples[itest]
                if acur > threshold_background * rb_average:
                    res.append({
                        'waveform': waveform[itest - k//6 : itest + (5*k)//6],
                        'index': itest
                    })
                    quiet_part = samples[itest + (3*k)//6 : itest + (5*k)//6]
                    len_ovr_loudness += len(quiet_part)
                    overall_loudness += torch.sum(quiet_part)
            waveform_threshold[itest] = threshold_background * rb_average
            waveform_max[itest] = samples[que[0]]

    if remove_low_power:
        while True:
            old_n = len(res)

            avg_power = sum(samples[kp["position"]] for kp in res) / len(res)

            tmp_res = res[:]
            res.clear()

            for kp in tmp_res:
                if samples[kp["position"]] > 0.3 * avg_power:
                    res.append(kp)

            if len(res) == old_n:
                break
    
    #if len_ovr_loudness == 0:
    #    len_ovr_loudness = 1
    avg_loudness = overall_loudness / len_ovr_loudness

    return {'waveform_threshold': waveform_threshold, 
            'waveform_max': waveform_max,
            'res': res,
            'avg_loudness': avg_loudness
            }

# Peak counter
def count_peaks(samples, key_length=14400):
    final_samples = pyln.normalize.peak(samples, -1)
    indexes = detect_peaks(final_samples[key_length: -key_length], show=True, mpd=key_length, mph=0.04)
    return len(indexes)

# Isolator - New version
def isolator_new(file_path, sr, key_length=14400, k=0.15):
    # samples for audiosegment
    pydub_samples = AudioSegment.from_file(file_path, format="wav", frame_rate=sr)
    # samples for librosa
    librosa_samples, sr = torchaudio.load(file_path, sr=44100)
    # finding silences in audio
    silences = silence.detect_silence(pydub_samples, silence_thresh=1.01*pydub_samples.dBFS, min_silence_len=50)
    # finding average dbfs
    ovr_dbms = []
    for start_ind, final_ind in silences:
        ovr_dbms.append(pydub_samples[start_ind:final_ind].dBFS)
    avg_dbfs = torch.mean(torch.tensor(ovr_dbms))
    # finding key presses    
    librosa_samples_tensor = torch.tensor(librosa_samples)  # Convert to tensor
    return_dic = find_key_presses(librosa_samples_tensor,[],{},{},torch.abs(k*avg_dbfs), key_length, False)
    return return_dic

# Creating dataset
def create_dataset(keys, initial_k, key_length=8820):
    data_dict = {'Key':[], 'File':[]}
    # For each key marked in keys, we'll isolate the keystrokes and add them to the data dictionary
    for i, key in enumerate(keys):
        curr_key = key
        if key.isalpha and not key.isalnum(): # if the key is a string
            curr_key = key.lower() # convert to lowercase
        # Path to audio file corresponding to current audio key.
        # TODO: Adapt for generic use 
        file_path = f'../MKA-dataset/{curr_key}mac.wav'
        # Generate samples
        samples, sr = torchaudio.load(file_path, sr=44100)
    
        # count peaks in samples
        peaks_count = count_peaks(samples, key_length)
        
        # find the starter value for finding the keystrokes
        k = initial_k 
        #jorge fix:
        curr_array=isolator_new(file_path, sr, key_length, k)['res']
        strokes = [curr['waveform'] for curr in curr_array]
        num_keys = len(strokes)
        
        # iterate until the number of keystrokes detected is equal to the number of peaks
        print(f'key {key}')
        # print(f'initial k={k:.3f}\tnum_keys={num_keys}\tpeaks={peaks_count}')
        # while num_keys != peaks_count:
        #     k += 0.01 if num_keys > peaks_count else -0.01
        #     curr_array=isolator_new(file_path, sr, key_length, k)['res']
        #     strokes = [curr['waveform'] for curr in curr_array]
        #     num_keys = len(strokes)
        #     print(f'k={k:.3f}\tnum_keys={num_keys}')
        print(f'final k={k:.3f}\tnum_keys={num_keys}\tpeaks={peaks_count}')
        print()
        
        # add keys to dictionary
        label = [keys[i]]*num_keys
        data_dict['Key'] += label
        data_dict['File'] += strokes
    # Convert complete dictionary to dataframe
    df = pd.DataFrame(data_dict)
    mapper = {}
    counter = 0
    for l in df['Key']:
        if not l in mapper:
            mapper[l] = counter
            counter += 1
    df.replace({'Key': mapper}, inplace = True)
    
    return df