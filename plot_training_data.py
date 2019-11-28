import csv
import matplotlib.pyplot as plt
import numpy as np

with open('training.csv') as f:
    reader = csv.DictReader(f, delimiter=';')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    for row in reader:
        # accuracy = row['accuracy'].split(",")
        accuracy = [float(s) for s in row['accuracy'].split(',')]
        val_accuracy = [float(s) for s in row['val_accuracy'].split(',')]
        batch_size = (row['batch_size'])
        n_epochs = (row['epochs'])
        plt.plot(accuracy, label=f'Train batch_size = {batch_size}, epochs = {n_epochs}')
        plt.plot(val_accuracy, label=f'Test batch_size = {batch_size}, epochs = {n_epochs}')
        plt.yscale('linear')
        plt.legend()
        # plt.legend([f"Train batch_size = {batch_size}, epochs = {n_epochs}', 'Test'"], loc='upper left')

        # plt.plot(row['accuracy'], label=f'Train batch_size = {row["batch_size"]}, epochs = {row["epochs"]}' )
        # plt.plot(row['val_accuracy'], label=f'Test batch_size = {row["batch_size"]}, epochs = {row["epochs"]}' )
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')

plt.savefig('training.png')
