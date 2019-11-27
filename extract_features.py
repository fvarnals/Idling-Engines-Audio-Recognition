import os
import numpy as np
import pandas as pd
import librosa
import glob
from keras.utils import np_utils
import tensorflow as tf
with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))

# full path of directory where data is stored
data_dir_path = '/home/fvarnals/Projects/Idling-Engines-Audio-Recognition/data'

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

    return pd.Series([feature, label])

# create dataframe of features with noise label for each audio file in training set
train = training_data[0:1]
val = training_data[2:3]

# apply parser to every value of pandas series (here train)
# axis=1 specifies columns
temp_train = train.apply(parser, axis=1)
temp_train.columns = ['feature', 'label']

temp_val = val.apply(parser, axis=1)
temp_val.columns = ['feature', 'label']

from sklearn.preprocessing import LabelEncoder

# compile features into X component
X = np.array(temp_train.feature.tolist())
val_x = np.array(temp_val.feature.tolist())

# compile labels into y component
y = np.array(temp_train.label.tolist())
val_y = np.array(temp_val.label.tolist())

lb = LabelEncoder()

# convert labels to one-hot vector
y = np_utils.to_categorical(lb.fit_transform(y))
val_y = np_utils.to_categorical(lb.fit_transform(val_y))

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(520, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X, y, batch_size=32, epochs=50, validation_data=(val_x, val_y))
