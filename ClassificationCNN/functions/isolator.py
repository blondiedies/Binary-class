import numpy as np
import librosa
import noisereduce as nr
from .helper_functions import disp_waveform


def isolator(signal, sample_rate, n_fft, hop_length, before, after, threshold, show=False):
    strokes = []
    # -- signal'
    denoised_signal = nr.reduce_noise(signal, sr=sample_rate)
    denoised_signal_boosted = np.where((np.abs(denoised_signal)>.04) & (np.abs(denoised_signal)<.4), (0.4-denoised_signal) * 1.15, denoised_signal)
    if show:
        disp_waveform(denoised_signal, 'signal waveform DENOISED', sr=sample_rate)
        disp_waveform(signal, 'signal waveform NOISED', sr=sample_rate)
        disp_waveform(denoised_signal_boosted, 'signal waveform DENOISED n BOOSTED', sr=sample_rate)
    signal = denoised_signal
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
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*hop_length) + n_fft//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            # strokes.append(torch.tensor(keystroke)[None, :])
            # keystroke = transform(keystroke)
            if len(keystroke) >= before + after:
                strokes.append(keystroke)
                if show and len(strokes) <= 5:
                    disp_waveform(keystroke, title=f'keystroke {len(strokes)}', sr=sample_rate)
                prev_end = timestamp+after
    return strokes
