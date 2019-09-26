**This is an on-going project in which I aim to create a program which can accurately classify common urban sounds, and detect idling engines in the hope that it could be employed in tackling offending vehicles as part of the recent government initiatives to impose "no idling" zones.

# Audio Classifier for Idling Engine Sounds

## Setup
This project requires Python 3 and the following packages:
```
pandas
librosa
glob
matplotlib
numpy
scipy
keras
tensorflow
```

NB The Training Set used is not included in this repository due to the large file size, however it can be downloaded for free from:
https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/

## Load and Plot Waveform

- The file ```Plot_Audio_Data.py``` can be used to show a plot of the waveform for sounds in the training set. The plot below shows the waveform for an Idling Engine recording.

![Idling Engine Plot](Engine_Idling.png)

## Multi-Class Classification using Neural Network

Using a sequential model with 2 hidden layers, I was able to achieve a maximum accuracy of 51% in the cross-validation set.
However, removing one of the hidden layers gave a much improved accuracy of 71% in the cross-validation set.
Moving on from this, I decided to try using a convoluted neural network on a Mel Frequency Spectogram of the audio clips.
