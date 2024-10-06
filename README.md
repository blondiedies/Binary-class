# Binary classifier for testing keyboard key audio presses.

Based on: https://www.geeksforgeeks.org/audio-classification-using-spectrograms/

Using audio from dataset: https://www.kaggle.com/datasets/lobanovaa/keystroke-datasets/code

And from dataset: https://data.mendeley.com/datasets/bpt2hvf8n3/3

# Custom Audio

For the binary classifier with custom audio, we recorded a series of audio samples. They're available in Dataset-custom-audio.

- base-audio: contains raw audio samples of keyboard presses. Audio files contain several key presses each.
- Split-audio: contains processed audio files of keyboard presses. Each key is separated by folder; each folder contains audio files of one key press each.
- test-audio: contains processed audio files of keyboard presses with the intent of using these for testing the model.
- test-audio-not-split: contains the raw audio file for the test audio.
- test-audio-standby-files: contains unused audio files meant for testing the model.

The model accepts only .wav files. Alternatively, it can accept .m4a files as raw audio samples, to be introduced in base-audio or test-audio-not-split.