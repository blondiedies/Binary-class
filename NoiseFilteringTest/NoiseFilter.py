
# peak identification

import numpy as np
import pydub.silence
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


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