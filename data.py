import pandas as pd
from collections import Counter

class Data(object):

    def __init__(self, file_path, csv_data_file_path):
        self.training_data = pd.read_csv(csv_data_file_path)


    def samples_per_label(self):
        # get sound labels from csv
        labels = self.training_data.Class
        print(len(labels) - 1)

        # count instances per label
        label_counts = Counter(labels)
        print(label_counts)


data = Data('/home/fvarnals/Projects/Idling-Engines-Audio-Recognition/data', "./data/train.csv")

data.samples_per_label()
