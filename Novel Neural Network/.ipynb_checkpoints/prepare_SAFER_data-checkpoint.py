import shutil
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from utils.preprocessing import baseline_als, butter_lowpass, butter_lowpass_filter


# find paths to files to be preprocessed

print('finding names of all test files ...')
all_files = []
for root, dirs, files in os.walk("F:\\DATA\\JSmith_SAFER_20220310\\raw_data"):
    for file in files:
        if(file.endswith(".dat")):
            all_files.append(os.path.join(root,file.split(".dat",2)[0]))
print('all files found :)')

path_to_metadata_files = "F:\\DATA\\JSmith_SAFER_20220310\\metadata"
with open(os.path.join(path_to_metadata_files, "feas1"), "rb") as file:   
    meas_id_list_feas1 = pickle.load(file)
    
feas1_filepaths_labelled = []
for filepath in tqdm(all_files):
    if "Feas1" in filepath:
        if int(filepath.split("saferF1_",2)[1]) in meas_id_list_feas1:
            feas1_filepaths_labelled.append(filepath)

af_df = pd.read_csv("F:\\DATA\\JSmith_SAFER_20220310\\raw_data\\Feas1\\files_with_af.csv", names=["filename"])
filenames_with_af = af_df.filename.tolist()


#define filter parameters
order = 6
fs = 30.0  # sampling frequency

# feas 1
feas1_path_to_folder = "F:\\DATA\\JSmith_SAFER_20220310\\preprocessed_labelled_data\\Feas1"
feas1_already_processed_files = [files for root, dirs, files in os.walk(feas1_path_to_folder)][0]
feas1_processed_file_as_string = ''.join(item for item in feas1_already_processed_files)

for test_file in tqdm(feas1_filepaths_labelled):
    
    # check not repeating process for already used file
    test_file_name = test_file.split("\\")[-1]    
    
    if test_file_name not in feas1_processed_file_as_string:
        
        #extract the sample 
        record = wfdb.rdrecord(test_file)
        ecg_signal = record.p_signal.T[0]
        
        #preprocess
        signal_no_wander = ecg_signal - baseline_als(ecg_signal)
        signal_smoothed = butter_lowpass_filter(signal_no_wander, 1.58, fs, order)
        signal_offset = signal_smoothed - np.min(signal_smoothed)
        signal_squashed = 2 * signal_offset / np.max(signal_offset)
        signal_normalised = signal_squashed - np.mean(signal_squashed)
                        
        #save this preprocessed sample
        filename = test_file.split("\\")[-1]
        npy_path = os.path.join(feas1_path_to_folder, filename)
        try:
            np.save(npy_path, signal_normalised)
        except:
            pass