import os
import pandas as pd
import numpy as np
import librosa

training_data = pd.read_csv("./data/train.csv")

data_dir_path = '/home/fvarnals/Projects/Idling-Engines-Audio-Recognition/data'



def create_mfccs(row):
    # function to load data files and extract features
    file = os.path.join(data_dir_path, 'Train', str(row.ID) + '.wav')

    # extract data using kaiser_fast technique
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')

    # we extract mfcc feature from data
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    feature = mfccs

    outputFile = data_dir_path + '/MFCCS_data/' + str(row.ID) + ".csv"
    file = open(outputFile, 'w+') # make file/over write existing file
    np.savetxt(file, feature, delimiter=",") #save MFCCs as .csv
    file.close() # close file

train = training_data[0:]
train.apply(create_mfccs, axis=1)
