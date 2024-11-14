# imports
import matplotlib.pyplot as plt
import librosa
import numpy as np
import array
from pydub import AudioSegment, silence
from collections import deque
from detecta import detect_peaks
import pyloudnorm as pyln
import pandas as pd

# Supporting Functions

# Waveform plotting function
def disp_waveform(signal,title='', sr=None, color='blue'):
    plt.figure(figsize=(7,2))
    plt.title(title)
    return librosa.display.waveshow(signal, sr=sr, color=color)

def signum(x):
    return 1 if x>0 else -1

def partition_audio(samples_arr : np.array):
    ret_samples = np.abs(samples_arr)
    return ret_samples

# Numpy to AudioSegment converter
def numpy_to_audiosegment(samples, sample_rate=44100, sample_width=2, channels=1):
    # Ensure the numpy array is in the correct dtype (int16 or int32 based on sample_width)
    if sample_width == 2:
        samples = np.int16(samples)
    elif sample_width == 4:
        samples = np.int32(samples)
    
    # Convert numpy array to byte data
    audio_data = array.array('h', samples)  # 'h' for 16-bit PCM audio
    byte_data = audio_data.tobytes()
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        data=byte_data,
        sample_width=sample_width,  # 2 for 16-bit, 4 for 32-bit
        frame_rate=sample_rate,
        channels=channels
    )
    
    return audio_segment

# Finding key presses using waveform
def find_key_presses(waveform, res, waveform_threshold, waveform_max, threshold_background, history_size, remove_low_power, clear_previous=False):
    # Clear previous results
    if clear_previous:
        res.clear()
        waveform_threshold = np.zeros_like(waveform)
        waveform_max = np.zeros_like(waveform)
    
    # Starting values
    rb_begin = 0
    rb_average = 0.0
    rb_samples = np.zeros(history_size)

    k = history_size
    que = deque(maxlen=k)

    samples = np.abs(waveform)  # Taking absolute values like waveformAbs in C++
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
                    overall_loudness += np.sum(quiet_part)
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

# Isolator functions

# Isolator - Legacy functionality
def isolator(signal, sample_rate, n_fft, hop_length, before, after, threshold, show=False):
    # variable for storing isolated keystrokes
    strokes = []

    # -- signal'
    fft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    norm = np.linalg.norm(energy)
    energy = energy/norm

    # -- energy'
    threshed = energy > threshold

    # -- peaks'
    if show:
        disp_waveform(threshed.astype(float), sr=sample_rate)
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)

    # '-- isolating keystrokes'
    timestamps = []
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*hop_length) + n_fft//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            if len(keystroke) >= before + after:
                strokes.append(keystroke)
                timestamps.append(timestamp)
                if show and len(strokes) <= 3:
                    disp_waveform(keystroke, title=f'keystroke {len(strokes)}', sr=sample_rate)
                prev_end = timestamp+after
    return peaks, strokes, timestamps

# Isolator - New version
def isolator_new(file_path, sr, key_length=14400, k=0.15):
    # samples for audiosegment
    pydub_samples = AudioSegment.from_file(file_path, format="wav", frame_rate=sr)
    # samples for librosa
    librosa_samples, sr = librosa.load(file_path, sr=44100)
    # finding silences in audio
    silences = silence.detect_silence(pydub_samples, silence_thresh=1.01*pydub_samples.dBFS, min_silence_len=50)
    # finding average dbfs
    ovr_dbms = []
    for start_ind, final_ind in silences:
        ovr_dbms.append(pydub_samples[start_ind:final_ind].dBFS)
    avg_dbfs = np.average(ovr_dbms)
    # finding key presses
    return_dic = find_key_presses(librosa_samples,[],{},{},np.abs(k*avg_dbfs), key_length, False)
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
        samples, sr = librosa.load(file_path, sr=44100)
    
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