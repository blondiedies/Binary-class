
# peak identification

import numpy as np
import pydub.silence
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import ffmpeg_normalize as normalizer
import torch
import noisereduce as nr
from scipy.io import wavfile
import pydub
import os
#For tools
from pydub import AudioSegment
import csv

### Utilities:
# Utility function to find relative path of file
def pathfinder (base_path, target_path):
    relative_path = os.path.relpath(target_path, start=os.path.dirname(base_path))
    return relative_path

def get_audio_length(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return audio.duration_seconds

def convert_to_ms(time):
    return round(time*1000)

def get_audio_length_average(audio_path, keys):
    lenghts = []
    for i, File in enumerate(keys):
        loc = audio_path + File
        length=get_audio_length(loc)
        print(f'File {loc} length: {length}')
        lenghts.append(length)

    average=np.mean(lenghts)
    print(f'Average audio length: {average}')
    return convert_to_ms(average)

def empty_file(csv_file_path):
    # Read the header (first row) of the CSV file
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the first row (header)
    
    # Write only the header back to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Wr`ite the header back to the file

### Classes:
# Class for normalization and denoising of .wav files
class normalizationAndDenoising():
    def __init__(self):
        self.normalizer=normalizer.FFmpegNormalize(normalization_type='ebu', target_level=-23,dual_mono=True)
        self.device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    def normalize(self, audio_location, new_file_name):
        self.normalizer.add_media_file(audio_location, new_file_name)
        self.normalizer.run_normalization()

    def denoise(self, audio_location, new_file_name, noise_location, stationary=True): #default is stationary, false is non-stationary
        # obtain normalized file
        self.rate_normal, self.data_normal = wavfile.read(audio_location)
        #noise profile
        audio_for_profile = pydub.AudioSegment.from_file(noise_location)
        _, noise_data = wavfile.read(noise_location) # noise profile, pre-normalized
        # find noise db and calculate noise threshold
        threshold=audio_for_profile.dBFS+10

        if stationary:
            #perform noise reduction
            reduced_noise = nr.reduce_noise(y=self.data_normal, sr=self.rate_normal, stationary=True, prop_decrease=2.0, freq_mask_smooth_hz=500,device=self.device, n_std_thresh_stationary=200)
            wavfile.write(new_file_name, self.rate_normal, reduced_noise)

        else:
            reduced_noise = nr.reduce_noise(y=self.data_normal, sr=self.rate_normal, y_noise=noise_data, stationary=False, prop_decrease=2.0, freq_mask_smooth_hz=500,device=self.device, thresh_n_mult_nonstationary=abs(threshold))
            wavfile.write(new_file_name, self.rate_normal, reduced_noise)


# Class for peak identification in audio files and splitting them into chunks
class peakIdentification():
    def __init__(self, audio, noise): #audio and noise must be pydub.AudioSegment
        self.audio=audio
        self.noise=noise
        self.threshold=self.noise.dBFS+10
        pass

    def match_target_amplitude (aChunk, target_dBFS): #normalizing audio
        change_in_dBFS = target_dBFS - aChunk.dBFS
        return aChunk.apply_gain(change_in_dBFS)

    def get_silence_length(self, expected_length, rate): #audio must be pydub.AudioSegment
        audio_length=self.audio.duration_seconds*1000 #converted to ms
        silence_length=int(audio_length/expected_length/rate)
        return int(silence_length)

    def find_chunk_amount(self,silence_length):
        average=self.audio.dBFS
        chunk_amount=pydub.silence.detect_nonsilent(
            self.audio, 
            min_silence_len=silence_length,
            silence_thresh=average)
        return len(chunk_amount)
        
    def split_audio(self, silence_length): #audio must be pydub.AudioSegment
        #find average dbfs of audio first
        average=self.audio.dBFS

        #splitting audio
        chunks = pydub.silence.split_on_silence(
            self.audio, #audio file
            min_silence_len=silence_length, #length of required silence chunk in ms
            silence_thresh=average) #a chunk is silent if it's below this threshold
    
        # Convert chunks to numpy arrays
        chunk_arrays = [np.array(chunk.get_array_of_samples()) for chunk in chunks]

    
        #return amount of chunks
        return len(chunks), chunk_arrays

    def find_n_peaks(self, plot): #finds amount of peaks in audio

        #distance is the real kicker here. It's the minimum distance between peaks. Need to find a way to calculate it.
        audio_array=np.array(self.audio.get_array_of_samples())
        distance=(self.audio.frame_rate+self.threshold)/1.5
        peaks,_=find_peaks(audio_array,prominence=75, height= (abs(self.audio.dBFS)), distance=distance)

        # Find median
        mean_peak_height = np.mean(audio_array[peaks])*.25
        median_peak_height = np.median(audio_array[peaks])*.25
        lower_limit=(median_peak_height+mean_peak_height)/2

        #Remove those under mdian
        peaks = peaks[audio_array[peaks] > lower_limit]

        #find peaks that are too close to each other and delete those
        #peak_diffs = np.diff(peaks)
        #avg_peak_diff = np.mean(peak_diffs)
        #peaks = np.delete(peaks, np.where(peak_diffs < avg_peak_diff/2))

        if plot:
            # Plot the audio data and the peaks
            plt.figure(figsize=(20, 10))
            plt.plot(audio_array, label='Audio Data')
            plt.plot(peaks, audio_array[peaks], "x", label='Peaks')
            plt.plot([0, len(audio_array)], [mean_peak_height, mean_peak_height], label='Mean Peak Height')
            plt.plot([0, len(audio_array)], [median_peak_height, median_peak_height], label='Median Peak Height')
            plt.plot([0, len(audio_array)], [lower_limit, lower_limit], label='Lower Limit')
            plt.legend()
            plt.show()

        return len(peaks)

    def get_silence_length(self, expected_length, rate): #audio must be pydub.AudioSegment
        audio_length=self.audio.duration_seconds*1000 #converted to ms
        silence_length=int(audio_length/expected_length/rate)
        return int(silence_length)

    def divide_into_chunks(self, plot=False):
        expected_length=self.find_n_peaks(plot=plot)
        print("Peak amount: "+str(expected_length))
        chunk_amount=expected_length+10
        rate=0.1*expected_length
        counter=['','','']
        while expected_length!=chunk_amount:
            silence_length = self.get_silence_length(expected_length, rate)
            chunk_amount=self.find_chunk_amount(silence_length)
            print("Amount of chunks vs expected: "+str(chunk_amount)+"/"+str(expected_length))
            if counter[0]=='down' and counter[1]=='up' and counter[2]=='down':
                print("Rate is stuck; selecting closest accurate.")
                rate=rate+0.05
                break
            elif chunk_amount<expected_length:
                print("Augmenting rate.")
                rate=rate+0.05
                counter[2]=counter[1]
                counter[1]=counter[0]
                counter[0]='up'
            elif chunk_amount>expected_length:
                print("Diminishing rate.")
                rate=rate-0.05
                counter[2]=counter[1]
                counter[1]=counter[0]
                counter[0]='down'
            print("-----")

        chunk_amount, chunks = self.split_audio( silence_length)
        print("Real amount of chunks:"+str(chunk_amount))
        return chunks, chunk_amount