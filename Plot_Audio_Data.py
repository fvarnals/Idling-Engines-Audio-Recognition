import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt

data, sampling_rate = librosa.load('./data/Train/20.wav')
plt.figure(figsize=(12, 4))
plt.title('Engine Idling')
plt.ylabel('Amplitude')
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()
