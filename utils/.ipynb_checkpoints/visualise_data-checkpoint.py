import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
import pandas as pd
import wfdb

# from find_label_for_data import get_label_from_dx_code
# from find_data_with_certain_label_or_prediction import find_all_data_with_given_label_and_preiction_label


def show_sample_ecg(path_to_sample, **kwargs):

    sample_to_visualise = path_to_sample.split("\\")[-1].split('.mat',2)[0] 
    metadata_path = path_to_sample.replace('.mat', '.hea')

    record = wfdb.rdrecord(path_to_sample.split('.mat',2)[0], channels=[0])
    if '2017' in path_to_sample:
        wfdb.plot_wfdb(record=record, plot_sym=True, time_units='seconds')
    else:
        dx_code = [com_.split(' ',2)[-1] for com_ in record.comments if 'Dx' in com_][0]
        label = get_label_from_dx_code(int(dx_code))
        wfdb.plot_wfdb(record=record, plot_sym=True, time_units='seconds', title='Diagnosis: {}'.format(label))


# plotting_specific_file = False
# plotting_specific_label_combination = False

# try:
#     predictions_path, dataset_path, requested_label, requested_prediction = sys.argv[1:]
#     plotting_specific_label_combination = True
# except ValueError:
#     path_to_sample = sys.argv[1]
#     plotting_specific_file = True
#     print(path_to_sample)

# if plotting_specific_label_combination:
#     path_to_labels = os.path.join(dataset_path, "REFERENCE.csv")
#     labels_df = pd.read_csv(path_to_labels, names=['filename', 'label'])
#     predictions_df = pd.read_csv(predictions_path, names=['filename', 'label'])
#     sample_to_visualise = find_all_data_with_given_label_and_preiction_label(requested_label, requested_prediction, labels_df, predictions_df)[0] + ".mat"
#     kwargs = {'include_label':True, 'labels_df':labels_df, 'include_prediction':True, 'predictions_df':predictions_df}
#     show_sample_ecg(os.path.join(sample_to_visualise, dataset_path), **kwargs)

# elif plotting_specific_file:
#     kwargs = {'include_label':False, 'labels_df':None, 'include_prediction':False, 'predictions_df':None}
#     show_sample_ecg(path_to_sample, **kwargs)