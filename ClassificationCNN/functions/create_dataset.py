import librosa
import pandas as pd
from .isolator import isolator


def create_dataset(n_fft, hop_length, before, after, keys, audio_dir, curr_labels, prom=0.2391, original=True):
    data_dict = {'Key': [], 'File': []}
    for i, File in enumerate(keys):
        loc = audio_dir + File
        samples, sr = librosa.load(loc)
        show = (File[6 if original else 0] == '0')
        strokes = isolator(samples, sr, n_fft, hop_length, before, after, prom, show)
        # print(f'Length strokes: {len(strokes)}')
        if show:
            print(f'Length strokes: {len(strokes)}')
        label = [curr_labels[i]] * len(strokes)
        data_dict['Key'] += label
        data_dict['File'] += strokes

    df = pd.DataFrame(data_dict)
    mapper = {}
    counter = 0
    for key in df['Key']:
        if key not in mapper:
            mapper[key] = counter
            counter += 1
    df.replace({'Key': mapper}, inplace=True)
    return df

