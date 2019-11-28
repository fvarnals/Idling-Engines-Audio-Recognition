import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import numpy as np

# data, sampling_rate = librosa.load('./data/Train/20.wav')
# plt.figure(figsize=(12, 4))
# plt.title('Engine Idling')
# plt.ylabel('Amplitude')
# librosa.display.waveplot(data, sr=sampling_rate)
# plt.show()

# mfccs = pd.read_csv('./data/MFCCS0.csv')
mfccs = np.genfromtxt('./data/MFCCS_11.csv', delimiter=',')

# plt.figure()
# librosa.display.specshow(mfccs, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.savefig('mfccs.png')

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mfccs,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig('mfccs.png')
