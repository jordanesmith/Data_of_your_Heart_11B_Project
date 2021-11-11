import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
import pandas as pd

from find_label_for_data import find_label
from find_data_with_certain_label_or_prediction import find_all_data_with_given_label_and_preiction_label


def show_sample_ecg(sample_to_visualise, dataset_path, **kwargs):

    Y = sio.loadmat(os.path.join(dataset_path, sample_to_visualise))['val'][0]
    X = np.arange(Y.size)
    
    if kwargs['include_label']: 
        labels_df = kwargs['labels_df']
        label = find_label(sample_to_visualise.split('.mat',2)[0], labels_df)

    if kwargs['include_prediction']: 
        predictions_df = kwargs['predictions_df']
        prediction = find_label(sample_to_visualise.split('.mat',2)[0], predictions_df)

    plt.plot(X,Y)

    title = 'Sample {}'.format(sample_to_visualise)
    # if kwargs['include_label'] and kwargs['include_prediction']:
    #     title'Sample {} with label: {} and model prediction: {}'.format(label, prediction))
    if kwargs['include_label']: 
        title += ' with label: {}'.format(label)
        if kwargs['include_prediction']:
            title += ' and model prediction: {}'.format(prediction)
    plt.title(title)
    plt.show()

dataset_path = 'physionet_datasets\\\\training2017'

path_to_labels = os.path.join(dataset_path, "REFERENCE.csv")
path_to_predictions = "Novel Neural Network\\\\answers.csv"

labels_df = pd.read_csv(path_to_labels, names=['filename', 'label'])
predictions_df = pd.read_csv(path_to_predictions, names=['filename', 'label'])

requested_label, requested_prediction = sys.argv[1:]
sample_to_visualise = find_all_data_with_given_label_and_preiction_label(requested_label, requested_prediction, labels_df, predictions_df)[0] + ".mat"

kwargs = {'include_label':True, 'labels_df':labels_df, 'include_prediction':True, 'predictions_df':predictions_df}

show_sample_ecg(sample_to_visualise, dataset_path, **kwargs)