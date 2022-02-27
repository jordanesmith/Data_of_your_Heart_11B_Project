import numpy as np 
from tqdm import tqdm
import os
import sys
import wfdb

def find_maximum_time_length_of_sample_in_dataset(dataset_path):

    # dataset_path = os.listdir(sys.argv[1])
    all_samples = [file_ for file_ in os.listdir(dataset_path) if file_.endswith('.hea')]
    all_times = []
    for sample in tqdm(all_samples):
        path_to_sample = os.path.join(dataset_path, sample)
        if "2017" in dataset_path:
            record = wfdb.rdrecord(path_to_sample.split('.hea',2)[0], channels=[0])
        elif "2020" in dataset_path:
            record = wfdb.rdrecord(path_to_sample.split('.hea',2)[0], channels=[1])
        total_time_of_record = record.sig_len/record.fs
        all_times.append(total_time_of_record)

    print(max(all_times)) 
    f = open("metadata.txt", "a")
    f.write("maximum_recording_length: {}".format(max(all_times)))
    f.close()

    return max(all_times)

# find_maximum_length_of_sample_in_dataset("physionet_datasets\\training2020\\Training_WFDB")