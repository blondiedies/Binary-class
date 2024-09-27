import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import librosa


def disp_waveform(signal, title='', sr=None, color='blue'):
    """
    # waveform function for me to not bang my keyboard
    """
    plt.figure(figsize=(7, 2))
    plt.title(title)
    return librosa.display.waveshow(signal, sr=sr, color=color)






