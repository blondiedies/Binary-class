import matplotlib.pyplot as plt
import librosa


def disp_waveform(signal, title='', sr=None, color='blue'):
    """
    waveform function for me to not bang my keyboard
    """
    plt.figure(figsize=(7,2))
    plt.title(title)
    librosa.display.waveshow(signal, sr=sr, color=color)
    plt.show()


def signum(x):
    return 1 if x > 0 else -1