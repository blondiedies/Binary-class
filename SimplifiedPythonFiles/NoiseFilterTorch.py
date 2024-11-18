# peak identification - torch version
from datetime import datetime
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import pydub 
import numpy as np

COLORS = ['blue','red']	
color_counter=0
TARGET_LENGTH = 16000 

### Utilities:
#Saves model results post-training to a CSV file
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

# Utility function to find relative path of file
def pathfinder (base_path, target_path):
    relative_path = os.path.relpath(target_path, start=os.path.dirname(base_path))
    return relative_path

# Normalize using minmax scaling
def normalize_tensor(tensor, min_value=-1, max_value=1):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # Scale to [0, 1]
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value  # Scale to [min_value, max_value]
    return normalized_tensor

# Dataset builder
def create_dataset(labels, audio_dir, plot=False, preffix="", suffix="", length=TARGET_LENGTH):
    # generate data dictionary
    data_dict = {'Key':[], 'File':[]}
    # generate keys
    keys = [preffix + k + suffix + '.wav' for k in labels]
    # Load noise sample
    noise_path = "../Dataset-custom-audio/audio-standby-files/noise-profile/Noise.wav"
    chunks_amount=[]

    for i, File in enumerate(keys):
        audio_path = audio_dir + File
        print(f'path: {audio_path}')

        # Separator
        divider = peakIdentification(audio_path, noise_path)
        chunks, chunks_n = divider.split_audio_at_peaks(plot=plot)
        print(f'File {File} chunks: {chunks_n}')

        #normalizing chunks to target lenght
        normalized_chunks = [normalize_tensor(chunk) for chunk in chunks]

        #add amount to list
        chunks_amount.append(chunks_n)

        # Add to dict
        temp = i
        while temp > len(labels):
            temp = i - len(labels)
        label = [labels[temp]] * chunks_n
        print(label)
        data_dict['Key'] += label
        print(data_dict['Key'])
        data_dict['File'] += normalized_chunks

    df = pd.DataFrame(data_dict)
    mapper = {}
    counter = 0
    for l in df['Key']:
        if l not in mapper:
            print("Enters mapper if")
            mapper[l] = counter
            counter += 1
        print(mapper)
    df.replace({'Key': mapper}, inplace=True)

    #calculate average amount of chunks
    average_chunks = np.average(chunks_amount)
    print(f'Average amount of chunks: {average_chunks}')

    return df, average_chunks

#Converts audiosegment type to a flattened numpy array
def audiosegment_to_flatarray(audiosegment):
    return np.array(audiosegment.get_array_of_samples()).flatten()

#Converts audiosegment type to a torchaudio tensor
def audiosegment_to_torchaudio(audiosegment):
    # Export AudioSegment to raw audio data
    raw_data = audiosegment.raw_data
    
    # Convert raw audio data to NumPy array
    audio_array = np.frombuffer(raw_data, dtype=np.int16)
    
    # Convert NumPy array to PyTorch tensor
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    
    # Reshape tensor to match the expected shape (channels, samples)
    audio_tensor = audio_tensor.view(1, -1)
    
    return audio_tensor

#Alternates between red and blue colors for plot purposes
def get_next_color():
    global color_counter
    color = COLORS[color_counter]
    color_counter = (color_counter + 1) % len(COLORS)
    return color

#Plots the chunks of audio data in the same plot with different colors
def plot_segments(chunks, title="Chunks"):
    # Calculate the lengths of each chunk
    chunk_lengths_base = [len(chunk) for chunk in chunks]
    x_offset_base = np.cumsum([0]+chunk_lengths_base[:-1])

    #graph this
    plt.figure(figsize=(20,10))

    for i, (chunk, x_offset_base) in enumerate(zip(chunks, x_offset_base)):
        x=np.arange(len(chunk))+x_offset_base
        plt.plot(x, chunk, label=f"Chunk {i+1}", color= get_next_color(), alpha=0.4)

    plt.title(title)
    plt.legend()
    plt.show()

#Plots the peaks in the audio data
def plot_peaks(audio_array, peaks, mean_peak_height, median_peak_height, lower_limit, title="Peaks"):
    plt.figure(figsize=(20, 10))
    plt.plot(audio_array, label='Audio Data')
    plt.plot(peaks, audio_array[peaks], "x", label='Peaks')
    plt.plot([0, len(audio_array)], [mean_peak_height, mean_peak_height], label='Mean Peak Height')
    plt.plot([0, len(audio_array)], [median_peak_height, median_peak_height], label='Median Peak Height')
    plt.plot([0, len(audio_array)], [lower_limit, lower_limit], label='Lower Limit')
    plt.title(title)
    plt.legend()
    plt.show()

## Class for peak identification in audio files and splitting them into chunks
class peakIdentification():
    #init using audiosegment 
    def __init__(self, audio_path, noise_path):
        self.audio_segment= pydub.AudioSegment.from_file(audio_path)
        self.audio_array=audiosegment_to_flatarray(self.audio_segment)
        self.noise_audiosegment=pydub.AudioSegment.from_file(noise_path)
        self.noise=audiosegment_to_torchaudio(self.noise_audiosegment)
        self.threshold=self.threshold=self.noise_audiosegment.dBFS+10
        self.audio_length=self.audio_segment.duration_seconds*1000 #converted to ms
        self.sample_rate=self.audio_segment.frame_rate

    #finds the amount of peaks in the audio file and returns the indexes in a tensor
    def find_n_peaks(self, plot): 
        # distance is the real kicker here. It's the minimum distance between peaks. The value at the end adjusts how finicky it is.
        distance = (self.sample_rate + self.threshold) / 1.5
        peaks, _ = find_peaks(self.audio_array, prominence=75, height=abs(abs(self.audio_segment.dBFS)), distance=distance) 

        #find median
        mean_peak_height = self.audio_array[peaks].mean() * 0.25
        median_peak_height = torch.median(torch.tensor(self.audio_array[peaks])).item() * 0.25
        lower_limit = (median_peak_height + mean_peak_height) / 2

        #remove those under median
        peaks = torch.tensor(peaks[self.audio_array[peaks] > lower_limit])

        #find peaks that are too close to each other and delete those
        peak_diffs = torch.diff(torch.tensor(peaks)).float()
        avg_peak_diff = torch.mean(peak_diffs)
        peaks = peaks[torch.where(peak_diffs >= avg_peak_diff / 2)[0] + 1]

        if plot:
            plot_peaks(self.audio_array, peaks, mean_peak_height, median_peak_height, lower_limit)

        return peaks
    
    #splits the audio file into chunks based on the peaks obtained
    def split_audio_at_peaks(self, plot=False):
        # Step 1: Find peaks
        peaks = self.find_n_peaks(plot)
        
        # Step 2: Split the audio array at each peak
        chunks = []
        start_idx = 0
        for peak in peaks:
            end_idx = int (peak.item()-(250*self.sample_rate/1000)) #250ms before peak
            chunk = self.audio_array[start_idx:end_idx]
            chunks.append(chunk)
            start_idx = end_idx
        
        # Add the last chunk
        chunks.append(self.audio_array[start_idx:])
        
        # Step 3: Convert chunks to tensors
        chunk_tensors = [torch.tensor(chunk, dtype=torch.float32) for chunk in chunks]

        if plot:
            plot_segments(chunk_tensors)
        
        # Step 4: Return the list of tensors and the total amount of chunks
        return chunk_tensors, len(chunk_tensors)