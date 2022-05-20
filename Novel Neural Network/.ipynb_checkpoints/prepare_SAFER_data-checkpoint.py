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
with open(os.path.join(path_to_metadata_files, "feas2"), "rb") as file:   
    meas_id_list_feas2 = pickle.load(file)
    
feas2_filepaths_labelled = []
for filepath in tqdm(all_files):
    if "Feas2" in filepath:
        if int(filepath.split("saferF2_",2)[1]) in meas_id_list_feas2:
            feas2_filepaths_labelled.append(filepath)

# af_df = pd.read_csv("F:\\DATA\\JSmith_SAFER_20220310\\raw_data\\Feas2\\files_with_af.csv", names=["filename"])
# filenames_with_af = af_df.filename.tolist()


#define filter parameters
order = 6
fs = 30.0  # sampling frequency

# feas 2
feas2_path_to_folder = "F:\\DATA\\JSmith_SAFER_20220310\\preprocessed_labelled_data\\Feas2"
feas2_already_processed_files = [files for root, dirs, files in os.walk(feas2_path_to_folder)][0]
feas2_processed_file_as_string = ''.join(item for item in feas2_already_processed_files)

for test_file in tqdm(feas2_filepaths_labelled):
    
    # check not repeating process for already used file
    test_file_name = test_file.split("\\")[-1]    
    
    if test_file_name not in feas2_processed_file_as_string:
        
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
        npy_path = os.path.join(feas2_path_to_folder, filename)
        try:
            np.save(npy_path, signal_normalised)
        except:
            pass