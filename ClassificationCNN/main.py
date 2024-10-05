import sklearn
from torchvision.transforms import Compose
from ClassificationCNN.functions.data_augmentation import time_shift, masking
from ClassificationCNN.functions.grid_search import grid_search
from ClassificationCNN.functions.transformations import ToMfcc, ToMelSpectrogram
from ClassificationCNN.models.coatnet import CoAtNet
from ClassificationCNN.models.mfcc_lstm import MfccLSTM
from functions.create_dataset import create_dataset
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd


# Creation of the dataset
N_FFT, HOP_LENGTH, BEFORE, AFTER = 1024, 225, 2400, 12000
MBP_AUDIO_DIR, labels, audiostr = ('/Users/jorgeleon/Binary-class/Dataset-for-Binary/base-audio/',
                                   list('1234567890QWERTYUIOPASDFGHJKLZXCVBNM'),
                                   'audio_')
keys = [audiostr + k + '.wav' for k in labels]
MBP_AUDIO_DIR_2, labels_2, audiostr = '../Dataset-custom-audio/base-audio-denoised-normalized/', list(
    '1234567890QWERTYUIOPASDFGHJKLZXCVBNMÑ+-'), ''
keys_2 = [audiostr + k + '.wav' for k in labels_2]
N_FFT_2, HOP_LENGTH_2, BEFORE_2, AFTER_2 = 1024, 225, 2200, 11000
original = True
concat = True

# Create the final dataset
mbp_dataset = create_dataset(N_FFT, HOP_LENGTH, BEFORE, AFTER, keys, MBP_AUDIO_DIR, labels, prom=0.004,
                             original=original)
if concat:
    mbp_dataset_2 = create_dataset(N_FFT_2, HOP_LENGTH_2, BEFORE_2, AFTER_2, keys_2, MBP_AUDIO_DIR_2, labels_2, prom=0.004, original=not original)
    print(mbp_dataset)
    print(mbp_dataset_2)
    mbp_dataset = pd.concat([mbp_dataset, mbp_dataset_2])

# Current device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# keys list
data_dict = {'Key': [], 'File': []}
# for custom audio testing
data_dict_t = {'Key': [], 'File': []}

# divide between audio and labels
audio_samples, labels = mbp_dataset['File'].values.tolist(), mbp_dataset['Key'].values.tolist()

# numpy array of the samples
audioDataset = np.array(audio_samples, dtype=object)

# transform functions
transform = Compose([ToMelSpectrogram()])
transform_mfcc = Compose([ToMfcc()])

audio_samples_new = audio_samples.copy()  # audio samples CNN

for i, sample in enumerate(audio_samples):
    audio_samples_new.append(time_shift(sample))
    labels.append(labels[i])

# convert labels to a numpy array
labels = np.array(labels)
audioDatasetFin, audioDatasetFinMasking, audioDatasetMfcc, audioDatasetMfccMasking = [], [], [], []

# creation of datasets
for i in range(len(audio_samples_new)):
    transformed_sample = transform(audio_samples_new[i])
    transformed_mfcc = transform_mfcc(audio_samples_new[i])

    # CoAtNet part
    audioDatasetFin.append((transformed_sample, labels[i]))
    audioDatasetFinMasking.append((masking(transformed_sample), labels[i]))

    # masking part
    audioDatasetMfcc.append((transformed_sample, transformed_mfcc, labels[i]))
    audioDatasetMfccMasking.append((masking(transformed_sample), transformed_mfcc, labels[i]))

# original = True --> CoAtNet // original = False --> MfccLSTM
original = True
if original:
    model = CoAtNet(num_epochs=700, patience=200)
    dataset = audioDatasetFin + audioDatasetFinMasking
    X = [t[0] for t in audioDatasetFin]
    X_masking = [t[0] for t in audioDatasetFinMasking]
    y = [t[1] for t in audioDatasetFin]
    y_masking = [t[1] for t in audioDatasetFinMasking]
else:
    model = MfccLSTM(patience=1)
    dataset = audioDatasetMfcc + audioDatasetMfccMasking
    X = [(t[0], t[1]) for t in audioDatasetMfcc]
    X_masking = [(t[0], t[1]) for t in audioDatasetMfccMasking]
    y = [t[2] for t in audioDatasetMfcc]
    y_masking = [t[2] for t in audioDatasetMfccMasking]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
grid_search = grid_search(X_train + X_masking, y_train + y_masking, model)
model.fit(X_train + X_masking, y_train + y_masking)
prediction = model.predict(X_test)
print(model.predict(X_test))
print(y_test)
# Metrics
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average='macro')
recall = recall_score(y_test, prediction, average='macro')
f1 = sklearn.metrics.f1_score(y_test, prediction, average='macro')

# Save csv data for later comparison
# save_csv(model_name, int(real_num_epochs), description, accuracy, precision, recall, f1)

# params_array = ["images", "sequences"]
# prediction = predict(test_set, model, params_array, model_name, device)
# labels_set = [t[-1] for t in test_set]
# final_labels_set = [complete_set[ind] for ind in labels_set]
