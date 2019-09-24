import os
import numpy as np
import pandas as pd
import librosa
import random

# full path of directory where data is stored
data_dir_path = '/Users/student/Projects/Engine_Idling/data'

# load labelled trainig data from csv
training_data = pd.read_csv("./data/train.csv")

def parser(row):
    # function to load data files and extract features
    file = os.path.join(data_dir_path, 'Train', str(row.ID) + '.wav')

    # extract data using kaiser_fast technique
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')

    # we extract mfcc feature from data
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    feature = mfccs
    label = row.Class

    return [feature, label]

# create features with noise label for each audio file in training set
temp = training_data.apply(parser, axis=1)
temp.columns = ['feature', 'label']
