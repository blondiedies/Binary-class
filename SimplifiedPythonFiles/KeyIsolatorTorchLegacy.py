# imports
import matplotlib.pyplot as plt
import librosa
from detecta import detect_peaks
import pyloudnorm as pyln
import pandas as pd
import torch

# Supporting Functions


# Waveform plotting function
def disp_waveform(signal, title='', sr=None, color='blue'):
    plt.figure(figsize=(7, 2))
    plt.title(title)
    return librosa.display.waveshow(signal.numpy(), sr=sr, color=color)


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
    fft = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    energy = torch.abs(torch.sum(fft, dim=0)).float()
    norm = torch.linalg.norm(energy)
    energy = energy / norm

    # -- energy'
    threshed = energy > threshold

    # -- peaks'
    if show:
        disp_waveform(threshed.float(), sr=sample_rate)
    peaks = torch.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate * 0.1 * (-1)

    # '-- isolating keystrokes'
    timestamps = []
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak * hop_length) + n_fft // 2
        if timestamp > prev_end + (0.1 * sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            if len(keystroke) >= before + after:
                strokes.append(keystroke)
                timestamps.append(timestamp)
                if show and len(strokes) <= 3:
                    disp_waveform(keystroke, title=f'keystroke {len(strokes)}', sr=sample_rate)
                prev_end = timestamp + after
    return peaks, strokes, timestamps


# Creating dataset
def create_dataset(labels, file_dir, prefix, suffix, noise_level, show=False, n_fft=2048, hop_length=512, before=2205, after=2205, threshold=0.1):
    data_dict = {'Key': [], 'File': []}
    # generate keys
    keys = [prefix + k + suffix + '.wav' for k in labels]
    # For each key marked in keys, we'll isolate the keystrokes and add them to the data dictionary

    # Load noise profile
    noise_profile, _ = librosa.load("Dataset-custom-audio\audio-standby-files\noise-profile\Noise.wav", sr=44100)
    noise_profile = torch.tensor(noise_profile)
    
    for i, File in enumerate(keys):

        # Path to audio file corresponding to current audio key.
        file_path = file_dir + File
        print(f'path: {file_path}')
        # Generate samples
        samples, sr = librosa.load(file_path, sr=44100)
        samples = torch.tensor(samples)

        # Add noise to the samples
        noise = noise_profile[torch.randint(0, len(noise_profile), (len(samples),))]
        signal_noise = samples + noise_level * noise

        # Isolator function
        _, strokes, _ = isolator(signal_noise, sr, n_fft=n_fft,hop_length=hop_length,before=before, after=after,threshold=threshold, show=show)
        num_keys = len(strokes)

        # add keys to dictionary
        label = [labels[i]] * num_keys
        data_dict['Key'] += label
        data_dict['File'] += strokes

        print(f'Key: {labels[i]} | Number of keystrokes: {num_keys}')
    
    # Convert complete dictionary to dataframe
    df = pd.DataFrame(data_dict)
    print(df)
    mapper = {}
    counter = 0
    for l in df['Key']:
        if l not in mapper:
            mapper[l] = counter
            counter += 1
    df.replace({'Key': mapper}, inplace=True)
    
    return df

# split individual audio
def process_ind_audio(file_dir, filename, threshold, show=False, n_fft=2048, hop_length=512, before=2205, after=2205):
    #we just need to isolate each key and add it to a list, regardless of labels
    # Path to audio file corresponding to current audio key.
    file_path = file_dir + filename
    print(f'path: {file_path}')
    # Generate samples
    samples, sr = librosa.load(file_path, sr=44100)
    samples = torch.tensor(samples)
    #audio_length = len(samples) / sr
    #threshold=audio_length*0.01/8
    print(f"threshold: {threshold}")

    # Isolator function
    _, strokes, _ = isolator(samples, sr, n_fft=n_fft,hop_length=hop_length,before=before, after=after,threshold=threshold, show=show)
    num_keys = len(strokes)

    print(f'Number of keystrokes: {num_keys}')
    
    return strokes