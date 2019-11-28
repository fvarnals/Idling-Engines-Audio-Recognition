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
    file = os.path.join(data_dir_path, 'MFCCS_data', str(row.ID) + '.csv')

    mfccs = np.genfromtxt(file)

    feature = mfccs

    label = row.Class

    return pd.Series([feature, label])

# create dataframe of features with noise label for each audio file in training set
train = training_data[0:4347]
val = training_data[4347:]

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

batch_size = 128
epochs=100
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y), verbose=1)

# plot schematic of the model
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from numpy import argmax

predictions = model.predict_classes(val_x, verbose=1)

from sklearn.metrics import confusion_matrix

y_true = np.array(temp_val.label)
y_pred = lb.inverse_transform(predictions)

# print(y_true)
# print(lb.inverse_transform(y_pred))
# confusion_matrix(y_true, y_pred,labels=['children_playing', 'street_music', 'dog_bark', 'engine_idling', 'jackhammer', 'drilling', 'gun_shot', 'car_horn', 'siren', 'air_conditioner'])

classes = ['children_playing', 'street_music', 'dog_bark', 'engine_idling', 'jackhammer', 'drilling', 'gun_shot', 'car_horn', 'siren', 'air_conditioner']

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

plot_confusion_matrix(y_true, y_pred, classes)

plt.savefig('confusion_matrix.png')
plt.close()

###################################################
# Add training data to .csv file
# import csv
# fields=[batch_size, epochs, history.history['accuracy'], history.history['val_accuracy']]
# with open(r'training.csv', 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(fields)
