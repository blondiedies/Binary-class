import random
import numpy as np
import librosa
import torch
from scipy.signal import resample
import cv2
import time
import pandas as pd
from datetime import datetime

def save_csv(model_name, num_epochs, description, accuracy, precision, recall, f1_score):
    csv_file_path = 'model_comparison.csv'
    
    # Read the existing CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file does not exist, create an empty DataFrame with the correct columns
        df = pd.DataFrame(columns=['Datetime', 'Name', 'Epochs', 'Description', 'Accuracy', 'Precision', 'Recall', 'F1'])
        
    # Data to append
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Remove newline characters from the description
    description = description.replace('\n', ' ').replace('\r', ' ')
    
    # Create a new column with the relevant information
    new_data = {
        'Datetime': [current_datetime],
        'Name': [model_name],
        'Epochs': [num_epochs],
        'Description': [description],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1_score],
    }
    
    new_df = pd.DataFrame(new_data)
    
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

class TimeShifting():
    def __call__(self, samples):
#       samples_shape = samples.shape
        samples = samples.flatten()
        
        shift = int(len(samples) * 0.4) #Max shift (0.4)
        random_shift = random.randint(0, shift) #Random number between 0 and 0.4*len(samples)
        data_roll = np.roll(samples, random_shift)
        return data_roll
    
def time_shift(samples):
    samples = samples.flatten()
    shift = int(len(samples) * 0.4) #Max shift (0.4)
    random_shift = random.randint(0, shift) #Random number between 0 and 0.4*len(samples)
    data_roll = np.roll(samples, random_shift)
    return data_roll

class SpecAugment(): #added from new version
    def __call__(self, samples):
        num_mask = 2
        freq_masking_max_percentage=0.10
        time_masking_max_percentage=0.10
        spec = samples.copy()
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

def masking(samples): #added from new version
    num_mask = 2
    freq_masking_max_percentage=0.10
    time_masking_max_percentage=0.10
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

class ToMelSpectrogram:
    def __init__(self, audio_length=14400, hop_length=225, n_fft=1024):
        self.audio_length = audio_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __call__(self, samples):
        if len(samples) > self.audio_length:
            #print("Enters if")
            samples = samples[:self.audio_length]
        elif len(samples) < self.audio_length:
            #print("Enters elif")
            samples = np.pad(samples, (0, self.audio_length - len(samples)), mode='constant')
        
        start_time = time.time()
        mel_spec = librosa.feature.melspectrogram(y=samples, sr=44100, n_mels=64, n_fft=self.n_fft, hop_length=self.hop_length)
        print(f"Mel spectrogram time: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        mel_spec_resized = cv2.resize(mel_spec, (64, 64), interpolation=cv2.INTER_AREA)
        print(f"Resizing time: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        mel_spec_resized = np.expand_dims(mel_spec_resized, axis=0)
        print(f"Expanding time: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        result = torch.tensor(mel_spec_resized)
        print(f"Tensor conversion time: {time.time() - start_time:.4f} seconds")
        
        return result


class ToMelSpectrogramMfcc:
    def __init__(self, audio_length=14400, hop_length=225, n_fft=1024):
        self.audio_length = audio_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __call__(self, samples):
        if len(samples) > self.audio_length:
            #print("Enters if")
            samples = samples[:self.audio_length]
        elif len(samples) < self.audio_length:
            #print("Enters elif")
            samples = np.pad(samples, (0, self.audio_length - len(samples)), mode='constant')

        #print("Creating melspectrogram")
        mel_spec = librosa.feature.melspectrogram(y=samples, sr=44100, n_mels=64, n_fft=self.n_fft, hop_length=self.hop_length)
        #print("mfcc")
        mel_spec = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
        #print("resizing")
        mel_spec_resized = cv2.resize(mel_spec, (64, 64), interpolation=cv2.INTER_AREA)
        #print("expanding")
        mel_spec_resized = np.expand_dims(mel_spec_resized, axis=0)

        return torch.tensor(mel_spec_resized)


class ToMfcc:
    def __init__(self, audio_length=14400):
        self.audio_length = audio_length

    def __call__(self, samples):
        if len(samples) > self.audio_length:
            samples = samples[:self.audio_length]
        elif len(samples) < self.audio_length:
            samples = np.pad(samples, (0, self.audio_length - len(samples)), mode='constant')
        
        mfcc_spec = librosa.feature.mfcc(y=samples, sr=44100)
        mfcc_spec = np.transpose(mfcc_spec)
        return torch.tensor(mfcc_spec)

def split_into_batches(data, batch_size):
    """Split data into batches of size batch_size."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def downsample_audio(audio_sample, target_length):
    """Downsample the audio sample to the target length."""
    return resample(audio_sample, target_length)