#%%
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torchvision.transforms import Compose
import random
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from pydub import AudioSegment, silence
import pickle
import pyloudnorm as pyln
from sklearn.metrics import make_scorer
#%%
# waveform function for me to not bang my keyboard
def disp_waveform(signal,title='', sr=None, color='blue'):
    plt.figure(figsize=(7,2))
    plt.title(title)
    librosa.display.waveshow(signal, sr=sr, color=color)
#%%
import noisereduce as nr

def signum(x):
    return 1 if x>0 else -1

def isolator(signal, sample_rate, n_fft, hop_length, before, after, threshold, show=False):
    strokes = []
    # -- signal'
    denoised_signal = nr.reduce_noise(signal, sr=sample_rate)
    # if show:
    #     disp_waveform(denoised_signal, 'signal waveform DENOISED', sr=sample_rate)
    #     disp_waveform(signal, 'signal waveform NOISED', sr=sample_rate)
    #     disp_waveform(denoised_signal_boosted, 'signal waveform DENOISED n BOOSTED', sr=sample_rate)
    signal = denoised_signal
    fft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    norm = np.linalg.norm(energy)
    energy = energy/norm
    # -- energy'
    threshed = energy > threshold
    # -- peaks'
    if show:
        # disp_waveform(threshed.astype(float), sr=sample_rate)
        pass
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
            # strokes.append(torch.tensor(keystroke)[None, :])
            # keystroke = transform(keystroke)
            if len(keystroke) >= before + after:
                strokes.append(keystroke)
                timestamps.append(timestamp)
                if len(strokes) <= 1:
                    # disp_waveform(keystroke, title=f'keystroke {len(strokes)}', sr=sample_rate)
                    pass
                prev_end = timestamp+after
    return peaks, strokes, timestamps
#%%
def partition_audio(samples_arr : np.array):
    ret_samples = np.abs(samples_arr)
    return ret_samples
#%%
import numpy as np
from collections import deque

def find_key_presses(waveform, res, waveform_threshold, waveform_max, threshold_background, history_size, remove_low_power):
    # Clear previous results
    # res.clear()
    # waveform_threshold = np.zeros_like(waveform)
    # waveform_max = np.zeros_like(waveform)
    # 
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
                
    if len_ovr_loudness > 0:
        avg_loudness = overall_loudness / len_ovr_loudness
    else:
        avg_loudness = 0
        
    return {'waveform_threshold': waveform_threshold, 
            'waveform_max': waveform_max,
            'res': res,
            'avg_loudness': avg_loudness
            }
#%%
import array

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
#%%
# constants
N_FFT, HOP_LENGTH, BEFORE, AFTER = 1024, 225, 2400, 12000


#%%
from detecta import detect_peaks

def count_peaks(samples, key_length=14400, show=True):
    # meter = pyln.Meter(44100)  # Create BS.1770 meter
    # loudness = meter.integrated_loudness(samples)
    # target_loudness = -25.0
    # samples = pyln.normalize.loudness(samples, loudness, target_loudness
    threshold = np.percentile(samples, 96.6)
    # final_samples = pyln.normalize.peak(samples, 0.75)
    indexes = detect_peaks(samples[int(key_length/3): -int(key_length/3)], show=show, mpd=key_length - key_length/3, mph=threshold)
    return len(indexes)

def isolator_new(file_path, sr, key_length=14400, k=0.15):
    pydub_samples = AudioSegment.from_file(file_path, format="wav", frame_rate=sr)
    silences = silence.detect_silence(pydub_samples, silence_thresh=1.01*pydub_samples.dBFS, min_silence_len=50)
    ovr_dbms = []
    for start_ind, final_ind in silences:
        ovr_dbms.append(pydub_samples[start_ind:final_ind].dBFS)
    avg_dbfs = np.average(ovr_dbms)
    samples, sr = librosa.load(file_path, sr=44100)
    samples = nr.reduce_noise(samples, sr=44100)
    return_dic = find_key_presses(samples,[],{},{},np.abs(k*avg_dbfs), key_length, False)
    return return_dic
#%%
samples, sr = librosa.load(f'../MKA datasets/Mac/Raw data/0.wav')
samples = nr.reduce_noise(samples, sr=44100)
threshold = np.percentile(samples, 97.5)
final_samples = pyln.normalize.peak(samples, 0.75)
indexes = detect_peaks(samples[int(8800/2): -int(8800/2)], show=True, mpd=8800-8800/3, mph=threshold)
print(len(indexes))
#%%
def create_dataset_viejo(n_fft, hop_length, before, after, keys, audio_dir, curr_labels, prom=0.2391, original=True, key_length=14400):
    data_dict = {'Key':[], 'File':[]}
    base_step = 0.01
    for i, File in enumerate(keys):
        curr_step = base_step
        loc = audio_dir + File
        samples, sr = librosa.load(loc)
        # samples = nr.reduce_noise(samples, sr=44100)
        show = (File[6 if original else 0] == '0')
        peaks_count = count_peaks(samples, key_length, True)
        strokes = isolator(samples, sr, n_fft, hop_length, before, after, prom, show)[1]
        num_keys = len(strokes)
        count = 0
        k = prom
        prev_k = prom
        print(f'num_keys: {num_keys} // peaks_count: {peaks_count} // prom: {prom}')
        while num_keys != peaks_count:
            if num_keys > peaks_count:
                if count > 0 and prev_k == k + curr_step:
                    curr_step /= 2
                elif count > 0:
                    curr_step += (curr_step / 2)
                prev_k = k
                k += curr_step
            else:
                if count > 0 and prev_k == k - curr_step:
                    curr_step /= 2
                elif count > 0:
                    curr_step += (curr_step / 2)
                prev_k = k
                k += -curr_step
            strokes = isolator(samples, sr, n_fft, hop_length, before, after, k, show)[1]
            num_keys = len(strokes)
            # print(f'actual k: {k:.3f} // num strokes: {num_keys}')
            # time.sleep(1)
            count += 1
        
        print(f'{File}. Len strokes: {len(strokes)}')
        if show:
            print(f'Length strokes: {len(strokes)}')
        label = [curr_labels[i]]*len(strokes)
        data_dict['Key'] += label
        data_dict['File'] += strokes
        
        

    df = pd.DataFrame(data_dict)
    mapper = {}
    counter = 0
    for l in df['Key']:
        if not l in mapper:
            mapper[l] = counter
            counter += 1
    df.replace({'Key': mapper}, inplace = True)

    return df

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#%%
# currently used keys
curr_keys = list('1234567890QWERTYUIOPASDFGHJKLZXCVBNM')
# curr_keys.append("space")
N_FFT, HOP_LENGTH, BEFORE, AFTER = 1024, 225, 2400, 12000
MBP_AUDIO_DIR, labels, audiostr = ('/Users/jorgeleon/Binary-class/Dataset-for-Binary/base-audio/', list('1234567890QWERTYUIOPASDFGHJKLZXCVBNM'), 'audio_')
# MBP_AUDIO_DIR, audiostr = '/Users/jorgeleon/Binary-class/MKA datasets/Mac/Raw data/', ''
keys = [audiostr + k + '.wav' for k in labels]
key_length=14400
BEFORE = int(key_length / 6)
AFTER = int(5 * (key_length / 6))
# Create the final dataset
mbp_dataset = create_dataset_viejo(N_FFT, HOP_LENGTH, BEFORE, AFTER, keys, MBP_AUDIO_DIR, labels, prom=0.2391,original=False, key_length=key_length)
#%%
#%%
#%%
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#%%
import audiosegment

def get_audio_length(audio_path):
    audio = audiosegment.from_file(audio_path)
    return audio.duration_seconds

def convert_to_ms(t):
    return round(t*1000)

def get_audio_length_average(audio_path, keys):
    lengths = []
    for i, File in enumerate(keys):
        loc = audio_path + File
        length = get_audio_length(loc)
        print(f'File {loc} length: {length:2f}\n')
        lengths.append(length)
    average = np.mean(lengths)
    return convert_to_ms(average)
#%%

def time_shift(samples):
    samples = samples.flatten()
    shift = int(len(samples) * 0.4) #Max shift (0.4)
    random_shift = random.randint(0, shift) #Random number between 0 and 0.4*len(samples)
    data_roll = np.roll(samples, random_shift)
    return data_roll

def masking(samples):
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
#%%
from skimage.transform import resize

class ToMelSpectrogram:
    def __init__(self, audio_length=14400):
        self.audio_length = audio_length

    def __call__(self, samples):
        if len(samples) > self.audio_length:
            samples = samples[:self.audio_length]
        elif len(samples) < self.audio_length:
            samples = np.pad(samples, (0, self.audio_length - len(samples)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=samples, sr=44100, n_mels=64, n_fft=1024, hop_length=225)
        mel_spec_resized = resize(mel_spec, (64, 64), anti_aliasing=True)
        mel_spec_resized = np.expand_dims(mel_spec_resized, axis=0)
        return torch.tensor(mel_spec_resized)


class ToMelSpectrogramMfcc:
    def __init__(self, audio_length=14400):
        self.audio_length = audio_length

    def __call__(self, samples):
        if len(samples) > self.audio_length:
            samples = samples[:self.audio_length]
        elif len(samples) < self.audio_length:
            samples = np.pad(samples, (0, self.audio_length - len(samples)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=samples, sr=44100, n_mels=64, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
        mel_spec_resized = resize(mel_spec, (64, 64), anti_aliasing=True)
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

#%%
transform = Compose([ToMelSpectrogram(key_length)])
transform_mfcc = Compose([ToMfcc(key_length)])
#%%
audio_samples = mbp_dataset['File'].values.tolist()
labels = mbp_dataset['Key'].values.tolist()

audio_samples_no_masking = audio_samples.copy()
labels_no_masking = labels.copy()
audio_samples_new = audio_samples.copy() # audio samples CNN
print(len(audio_samples))

print(type(audio_samples[0]))

for i, sample in enumerate(audio_samples):
    audio_samples_new.append(time_shift(sample))
    labels.append(labels[i])

# convert labels to a numpy array
labels = np.array(labels)
print(len(audio_samples_new))
print(len(labels))
#%%
audioDatasetFin, audioDatasetFinMasking, audioDatasetMfcc, audioDatasetMfccMasking = [], [], [], []

for i in range(len(audio_samples_new)):
    transformed_sample = transform(audio_samples_new[i])
    transformed_mfcc = transform_mfcc(audio_samples_new[i])
    
    # CoAtNet part
    audioDatasetFin.append((transformed_sample, labels[i]))
    audioDatasetFinMasking.append((masking(transformed_sample), labels[i]))
    
    # masking part
    audioDatasetMfcc.append((transformed_sample, transformed_mfcc, labels[i]))
    audioDatasetMfccMasking.append((masking(transformed_sample), transformed_mfcc, labels[i]))

#%%
import time
from sklearn.base import BaseEstimator


class MfccLSTM(nn.Module, BaseEstimator):
    def __init__(self, batch_size=16, num_epochs=500, patience=120):
        super(MfccLSTM, self).__init__()        
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_size = batch_size
        
        hidden_size = 32
        input_size = 20
        dropout = 0.2 
        num_classes = 36
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.LazyLinear(128)
        self.final_lstm = nn.LSTM(1, 128, batch_first=True, proj_size=64)
        
        self.fc = nn.LazyLinear(num_classes)
    
    def forward(self, images, sequences):
        # must return shape (batch_size, num_classes) 
        # batch_size: right now is 16
        # num_classes: right now is 36
        x1 = self.conv(images)
        # print(f'input of first lstm: {sequences.shape[1:]}')
        out1, _ = self.lstm(sequences)
        out1_dp = self.dropout(out1)
        # print(f'output of first lstm: {out1_dp.shape[1:]}')
        # print(f'input of second lstm: {out1_dp[:, -1, :].shape[1:]}')
        out2, _ = self.lstm2(out1_dp[:, -1, :])
        out2_dp = self.dropout(out2)
        # print(f'output of second lstm: {out2_dp.shape[1:]}')
        x2 = self.fc2(self.fc1(out2_dp))
        x3 = torch.cat((x1, x2), 1)
        # print(f'output of concatenation: {x3.shape[1:]}')
        # x4 = self.fc3(x3)
        # # print(f'input final lstm: {x4[:,-1,:].shape[1:]}')
        # print(f'x4.shape: {x4.shape[1:]}')
        # x_final = self.final_lstm(x4)
        # # x = self.fc(final_out[:, -1, :])
        x = self.fc(x3)
        return x
    
    def fit(self, X, y):
        self._optimizer = optim.Adam(self.parameters(), lr=5e-4)
        # same training method but now inside the class
        model = self.to(device)
        
        # loss criterion
        criterion = nn.CrossEntropyLoss()
        
        # # concatenate so it has the same shape as before
        # dataset = np.concatenate((X, y), axis=1)
         # concatenate so it has the same shape as before
        dataset = [(X[i], y[i]) for i in range(len(X))]
        train_set, val_set = train_test_split(dataset, test_size=0.005)
        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        
        best_val_acc, epochs_no_imp = 0, 0
        train_accuracies, val_accuracies = [], []
        
        for epoch in range(self.num_epochs):
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            tic = time.perf_counter()
            
            for (images, sequences), labels in train_loader:
                images = images.to(device)
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                self._optimizer.zero_grad()
        
                # converting labels to Long to avoid error "not implemented for Int"
                labels = labels.long()
                
                # Forward pass
                outputs = model(images, sequences)
                loss = criterion(outputs, labels)
                epoch_train_loss += loss.item() * images.size(0)
        
                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
                
                # Backward pass
                loss.backward()
                self._optimizer.step()
            
            toc = time.perf_counter()
            time_taken = toc - tic
            
            epoch_train_loss /= len(train_loader.dataset)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            
            # Evaluation of the model
            model.eval()
            total, correct = 0, 0
            
            for (images, sequences), labels in val_loader:
                images = images.to(device)
                sequences = sequences.to(device)
                labels = labels.to(device)
        
                outputs = model(images, sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            #
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Iter Time: {time_taken:.2f}s")
                
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                epochs_no_imp = 0
                best_model_state = model.state_dict()  # Save the best model
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= self.patience:
                print(f'Early stopping after {epoch+1} epochs')
                model.load_state_dict(best_model_state)  # Load the best model
                break
        return self
        
        
    def predict(self, X):
        argnames=["images", "sequences"]
        fin_dict = {}
        # create the list with each of the ith range tuples
        for i in range(len(X[0])-1):
            fin_dict[argnames[i]] = [torch.tensor(t[i]) for t in dataset]

        # torch.stack each one of the lists
        for key in fin_dict.keys():
            fin_dict[key] = torch.stack(fin_dict[key]).to(device)
        
        images = [tup[0] for tup in X]
        sequences = [tup[1] for tup in X]
        images_torch, sequences_torch = torch.tensor(np.array(images)).to(device), torch.tensor(np.array(sequences)).to(device)
        # model specifying
        model = self.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(images_torch, sequences_torch)
            _, predicted = torch.max(outputs.data, 1)
        
        pred = []
        # phrase = predicted.tolist()
        # for i in range(len(phrase)):
        #     pred.append(self.keys[phrase[i]])
        # 
        # pred_df = pd.DataFrame(pred)
        # return np.squeeze(pred_df.to_numpy().T)
        return predicted.tolist()
#%%
from coatnet import CoAtNet as CoAtNetImp
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR

num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D

class CoAtNet(nn.Module, BaseEstimator):
    def __init__(self, lr=5e-6, num_epochs=500, patience=30, keys='1234567890QWERTYUIOPASDFGHJKLZXCVBNM'):
        super(CoAtNet, self).__init__()    
        self.keys = keys
        self.model = CoAtNetImp((64, 64), 1, num_blocks, channels, num_classes=len(self.keys))
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, X, y, model_name, batch_size=16, lr=5e-6, random_state=42):
        # concatenate so it has the same shape as before
        dataset = [(X[i], y[i]) for i in range(np.array(X).shape[0])]
        # dataset = np.concatenate((X, y), axis=1)
        train_set, val_set = train_test_split(dataset, test_size=0.005, random_state=random_state, shuffle=True)
        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader, val_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g), DataLoader(val_set, batch_size=batch_size, shuffle=True, generator=g)

        # Initialize model, optimizer, and loss function
        self._optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # scheduler = CyclicLR(self._optimizer, base_lr=lr, max_lr=5e-5, step_size_up=5, mode="triangular")
        # scheduler = OneCycleLR(self._optimizer, max_lr=lr, steps_per_epoch=int(len(dataset)/batch_size), total_steps=self.num_epochs*int(len(dataset)/batch_size))
        # same training method but now inside the class
        model = self.model.to(device)
        
        # loss criterion
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc, epochs_no_imp = 0, 0
        train_accuracies, val_accuracies = [], []
        
        for epoch in range(self.num_epochs):
            model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            tic = time.perf_counter()
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                self._optimizer.zero_grad()
        
                # converting labels to Long to avoid error "not implemented for Int"
                labels = labels.long()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_train_loss += loss.item() * images.size(0)
        
                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
                
                # Backward pass
                loss.backward()
                self._optimizer.step()
            
            # scheduler.step()
            toc = time.perf_counter()
            time_taken = toc - tic
            
            epoch_train_loss /= len(train_loader.dataset)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            
            # Evaluation of the model
            model.eval()
            total, correct = 0, 0
            
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
        
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            if (epoch + 1) % 1 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {epoch_train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f} Iter Time: {time_taken:.2f}s")
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                epochs_no_imp = 0
                best_model_state = model.state_dict()  # Save the best model
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= self.patience:
                print(f'Early stopping after {epoch+1} epochs')
                model.load_state_dict(best_model_state)  # Load the best model
                break
            
        torch.save(self.model.state_dict(), f'models/{model_name}.pth')
        #     # Plot accuracy curves
        # plt.plot(range(1, self.num_epochs+1), train_accuracies, label='Training Accuracy')
        # plt.plot(range(1, self.num_epochs+1), val_accuracies, label='Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy vs Epoch')
        # plt.legend()
        # plt.show()
        return self
    
    def predict(self, X):
        argnames=["x"]
        fin_dict = {}
        # create the list with each of the ith range tuples
        # print(range(len(X[0])-1))
        # for i in range(len(X[0])-1):
        #     fin_dict[argnames[i]] = [t[i] for t in dataset]
        #     
        # # torch.stack each one of the lists
        # for key in fin_dict.keys():
        #     fin_dict[key] = torch.stack(fin_dict[key]).to(device)
        
        X = torch.tensor(np.array(X)).to(device)
        
        # model specifying
        model = self.model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
        
        pred = []
        # phrase = predicted.tolist()
        # for i in range(len(phrase)):
        #     pred.append(self.keys[phrase[i]])
        # 
        # pred_df = pd.DataFrame(pred)
        # return np.squeeze(pred_df.to_numpy().T)
        return predicted.tolist()
#%%
def getIndCurrKeys(ind: int):
    return curr_keys[ind]
#%%
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
#%%
from datetime import datetime

# dataset = audio_samples_no_masking
# dataset_labels = labels_no_masking

dataset = audio_samples
dataset_labels = labels_no_masking
train_set, test_set, labels_train_set, labels_test_set = train_test_split(dataset, dataset_labels, test_size=0.001, random_state=SEED, shuffle=True)
final_train_set = []
coatnet = True

if coatnet:
    model = CoAtNet(keys=curr_keys)
    for i in range(len(train_set)):
        transformed_sample = transform(train_set[i])
        transformed_sample_ts = transform(time_shift(train_set[i]))
        # append to final train set
        final_train_set.append((transformed_sample, labels_train_set[i]))
        final_train_set.append((transformed_sample_ts, labels_train_set[i]))
        # final_train_set.append((masking(transformed_sample), labels_train_set[i]))
        # final_train_set.append((masking(transformed_sample), labels_train_set[i]))
        # final_train_set.append((masking(transformed_sample_ts), labels_train_set[i]))
        # final_train_set.append((masking(transformed_sample_ts), labels_train_set[i]))
    #   Copy final train set to iterate over it
    X_train = [t[0] for t in final_train_set]
    y_train = [t[1] for t in final_train_set]
    print(f'LEN FINAL TRAIN SET: {len(final_train_set)}')
else:
    model = MfccLSTM()
    for i in range(len(train_set)):
        transformed_mfcc = transform_mfcc(train_set[i])
        transformed_sample = transform(train_set[i])
        final_train_set.append((transformed_sample, transformed_mfcc, labels_train_set[i]))
        final_train_set.append((masking(transformed_sample), transformed_mfcc, labels_train_set[i]))
    X_train = [(t[0],t[1]) for t in final_train_set]
    y_train = [t[2] for t in final_train_set]

param_grid = {
    'patience': [75],
    'lr': [5e-6],
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0.0),
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0.0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0.0),
}


grid_search = GridSearchCV(CoAtNet(), param_grid, cv=5, scoring=scoring, refit=False, verbose=3)
fit_params = {"model_name": 'grid_search_model_22-01-25', "batch_size": 16}
grid_search.fit(X_train, y_train, **fit_params)
# Get the current date and time
# current_datetime = datetime.now()
# formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
# print("Formatted Date and Time:", formatted_datetime)
# batch_size = 16
# lr = 1e-6
# model_name = f'model_{batch_size}_{lr}_{formatted_datetime}'
# model.fit(X_train, y_train,model_name,batch_size,lr)
#
# # # # Load the existing checkpoint
# # checkpoint = torch.load("models/13-01-24.pth", weights_only=True)
# #
# # # Rename the keys
# # new_checkpoint = {}
# # for k, v in checkpoint.items():
# #     new_checkpoint[f"model.{k}"] = v
# #
# # model.load_state_dict(new_checkpoint, strict=True)
# # model.load_state_dict(torch.load("models/13-01-24.pth", weights_only=True), strict=True)
#
# final_test_set = list(map(transform, test_set))
# print(final_test_set[0].shape)
# prediction = model.predict(final_test_set)
# np_prediction = np.array(prediction)
# accuracy = accuracy_score(labels_test_set, np_prediction)
# print(f'Final Accuracy: {accuracy:.3f}')
# # Calculate precision, recall, and F1 score
# precision = precision_score(labels_test_set, np_prediction, average='weighted')
# recall = recall_score(labels_test_set, np_prediction, average='weighted')
# f1 = f1_score(labels_test_set, np_prediction, average='weighted')
#
# print(f'Precision: {precision:.3f}')
# print(f'Recall: {recall:.3f}')
# print(f'F1 Score: {f1:.3f}')
#%%
cv_results_df = pd.DataFrame(grid_search.cv_results_)
 
sorted_df = cv_results_df.sort_values(by=['rank_test_accuracy'])

for ind, row in sorted_df.iterrows():
    print(f'Rank: {row["rank_test_accuracy"]}')
    print(f'Params: {row["params"]}')
    print(f'Test accuracy: {row["mean_test_accuracy"]:.3f}', end=" / ")
    print(f'F1 Weighted: {row["mean_test_f1_weighted"]:.3f}', end=" / ")
    print(f'Recall Weighted: {row["mean_test_recall_weighted"]:.3f}', end=" / ")
    print(f'Precision Weighted: {row["mean_test_precision_weighted"]:.3f}', end="\n\n")
#%%
# final_test_set = list(map(transform, test_set))
# print(final_test_set[0].shape)
# prediction = model.predict(final_test_set)
# np_prediction = np.array(prediction)
# accuracy = accuracy_score(labels_test_set, np_prediction)
# print(f'Final Accuracy: {accuracy:.3f}')
# # Calculate precision, recall, and F1 score
# precision = precision_score(labels_test_set, np_prediction, average='weighted')
# recall = recall_score(labels_test_set, np_prediction, average='weighted')
# f1 = f1_score(labels_test_set, np_prediction, average='weighted')
#
# print(f'Precision: {precision:.3f}')
# print(f'Recall: {recall:.3f}')
# print(f'F1 Score: {f1:.3f}')
#%%

#%% md
# e
#%%
import ollama
print(sorted(labels_test_set))
while True:
    word = input("\nIntroduce la palabra:")
    if word == 'exit':
        break
    word = word.upper()
    curr_word, curr_labels = [], []

    for letter in word:
        letter_index = curr_keys.index(letter)
        # Convert numpy array to list
        labels_test_set_list = labels_test_set.tolist()
        
        # Get the index of the first occurrence
        try:
            final_index = labels_test_set_list.index(letter_index)
        except ValueError:
            print(f'letter {letter} not found')
            final_index = None  # Handle the case when the value is not found
        # append to curr_word the value in that index of X_test
        curr_word.append(transform(test_set[final_index]))
        curr_labels.append(labels_test_set[final_index])
    print("curr_word[0].shape")
    print(curr_word[0].shape)
    model.eval()
    prediction = model.predict(curr_word)
    prediction_list = list(map(getIndCurrKeys, prediction)) 
    print(f'prediction: {prediction_list}')
    print(f'real labels: {list(map(getIndCurrKeys, curr_labels))}')
    
    response = ollama.chat(model='spanishSpellchecker', messages=[
      {
        'role': 'user',
        'content': ''.join(prediction_list)
      },
    ])
    print(response['message']['content'])
    # time.sleep(3)
#%%