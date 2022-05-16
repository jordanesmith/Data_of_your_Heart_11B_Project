""" conda env tf-NNN-build"""
"""python challenge.py ..\2_data\physionet_datasets\dev .mat csv_on_interupt <--- example cmd line """
"""                    [    dataset_relative_path     ] 
                                                        [filetypes present]
                                                             [if csv_on_interupt is included, a keyboard interupt will lead to 
                                                              a csv with all predictions made up until this point being saved, 
                                                              otherwise they will be found in the .txt file]
"""


### Import libraries
import scipy.io;import sys;import numpy as np;import tensorflow as tf; import os; import pandas as pd
from tqdm import tqdm
import math 
import wfdb 
import matplotlib.pyplot as plt
from utils.preprocessing import baseline_als, butter_lowpass_filter
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, lfilter, freqz



### parse the command line arguments
try:
    dataset_path = sys.argv[1]
except IndexError:
    dataset_path = input("********** No dataset path given, type it in below:    ********** \n")
dataset_name = dataset_path.replace('\\', '_')
dataset_name = dataset_name.replace(":","")

    
try: 
    filetypes = sys.argv[2:]
    if "csv_on_interupt" in filetypes: filetypes.remove("csv_on_interupt")
    if len(filetypes) == 0: 
        filetypes = input("********** No filetype given, type it in below:    ********** \n").split()
except IndexError:
    filetypes = input("********** No filetype given, type it in below:    ********** \n").split()
        
    
### Load graphs from training on 2017 dataset
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,input_map=None,return_elements=None,name="prefix",op_dict=None,producer_op_list=None)
    sess_out = tf.Session(graph=graph)
    x_out = graph.get_tensor_by_name('prefix/InputData/X:0')
    y_out = graph.get_tensor_by_name('prefix/FullyConnected/Softmax:0')
    return(sess_out,x_out,y_out)

print('loading graphs ...')
sess,x,y = load_graph('frozen_graph.pb.min');sess1,x1,y1 = load_graph('frozen_graph21.pb');sess2,x2,y2 = load_graph('frozen_graph22.pb');sess3,x3,y3 = load_graph('frozen_graph23.pb');sess4,x4,y4 = load_graph('frozen_graph24.pb');sess5,x5,y5 = load_graph('frozen_graph11.pb');sess6,x6,y6 = load_graph('frozen_graph12.pb');sess7,x7,y7 = load_graph('frozen_graph13.pb')
print('graphs loaded :)')


### Find all files of given filetypes in the provided dataset
print('finding names of all test files ...')
all_files = []
for root, dirs, files in os.walk(dataset_path):
    filetype = filetypes[0]
    for file in files:
        if file.endswith(filetype):
            all_files.append(os.path.join(root,file.split(filetype,2)[0]))
print('all files found :)')

#TODO delete this/ find better way
# all_files = pd.read_csv("F:\\DATA\\JSmith_SAFER_20220310\\raw_data\\Feas1\\files_with_normal.csv", names=["filepath"]).filepath.to_list()



### Make predictions for all the files in the given dataset which have not had predictions made already.

# Predictions made already will be found in the following 2 locations
path_to_predictions_made_txt = "../predictions/{}answers.txt".format(dataset_name)
path_to_predictions_made_csv = "../predictions/{}answers.csv".format(dataset_name)
print("******************",path_to_predictions_made_txt)
print("******************",path_to_predictions_made_csv)
# Loop until keyboard interupt or all predictions made
try:

    # Find which files to start on 
    try:
        df_prev_pred = pd.read_csv(path_to_predictions_made_csv, names=['filename', 'label'])
        most_recent_prediction = df_prev_pred.filename.sort_values(ascending=False).iloc[0]
        index_to_start_on = np.where(np.array(all_files)==most_recent_prediction)[0][0] + 1
    except FileNotFoundError: 
        index_to_start_on = 0
    
    # Start making predictions on these files
    counter = 0
    for test_file in tqdm(all_files[index_to_start_on:]):
        counter += 1
        
        # Read waveform samples (input is in WFDB-MAT format)
        if ".mat" in filetypes:
            ecg = scipy.io.loadmat(os.path.join(dataset_path, test_file))['val'][0]
        elif ".dat" in filetypes:
            record = wfdb.rdrecord(test_file)
            ecg = record.p_signal.T[0]
            # ecg = ecg - baseline_als(ecg)
            # ecg = butter_lowpass_filter(ecg, 0.7, 30)
        
        # resample this if needed to fit the tensorshape 
        max_length = 145 #TODO extract this from metadata folder instead
        sample_freq = 500 # for zenicor device
        max_height = 1000
        ecg = ecg * 1000 / np.max(ecg)
        sample_time_length = len(ecg) / sample_freq
        # assert (sample_time_length >= 29 and sample_time_length <= 31), f"sample time length not near 30.4 seconds expected, but was {sample_time_length}"
        resample_needed = bool(sample_freq != 300)
        
        if resample_needed:
            resample_factor = 300/sample_freq
            n = ecg.size
            x_ = np.linspace(0, n, int( n * resample_factor ))
            xp = np.arange(0, n)
            yp = ecg
            ecg = np.interp(x_, xp, yp).astype(int)

        # loop over all 58 second sections, 58 seconds is requirement from the model provided in the 2017 challenge
        probabilities = np.array([[0.,0.,0.,0.],
                                 [0.,0.,0.,0.],
                                 [0.,0.,0.,0.],
                                 [0.,0.,0.,0.],
                                 [0.,0.,0.,0.],
                                 [0.,0.,0.,0.],
                                 [0.,0.,0.,0.]]) # p1,p2,p3,p4,p5,p6,p7
        
        for i in range(math.ceil(sample_time_length/58)):
            
            # extract 58 second sample
            time_range = [i*58, (i+1)*58]
            index_range = [int(index * 300) for index in time_range] # frequency is 300 hz 
            if index_range[1] > len(ecg): #if sample is less than 58 seconds long
                ecg_section = ecg
            else:
                ecg_section = ecg[index_range[0]: index_range[1]]

            # The classification algorithm
            Sxx = (ecg_section-7.51190822991)/235.404723927
            Sxx_all = np.array([Sxx[i:(i+5*300)] for i in range(0,len(Sxx)-5*300+1,300)])[:,:,None] # original sample_freq = 300
            pred_prob = np.array(sess.run(y,{x:Sxx_all})) 
            Sxx_all = np.pad(pred_prob,((0,58-len(pred_prob)),(0,0)),mode="constant") 
            p1_,p2_,p3_,p4_,p5_,p6_,p7_ = sess1.run(y1,{x1:[Sxx_all]}),sess2.run(y2,{x2:[Sxx_all]}),sess3.run(y3,{x3:[Sxx_all]}),sess4.run(y4,{x4:[Sxx_all]}),sess5.run(y5,{x5:[Sxx_all]}),sess6.run(y6,{x6:[Sxx_all]}),sess7.run(y7,{x7:[Sxx_all]})
            
            probabilities += np.vstack((p1_,p2_,p3_,p4_,p5_,p6_,p7_))
            
        p1,p2,p3,p4,p5,p6,p7 = list(np.array(probabilities)/(i+1))
        index = [1 if np.argmax(p5+p6+p7) == 1 else np.argmax(p1+p2+p3+p4)][0]
        norm_prob_arr_1 = (p5+p6+p7)/np.sum(p5+p6+p7)
        norm_prob_arr_2 = (p1+p2+p3+p4)/np.sum(p1+p2+p3+p4)
        overall_probabilities = np.round(np.mean(np.vstack((norm_prob_arr_1, norm_prob_arr_2)), 0),4)
        probability_of_AF = overall_probabilities[1] 
        print("\nN\tA\tO\t~\n{}\t{}\t{}\t{}".format(overall_probabilities[0],overall_probabilities[1],overall_probabilities[2],overall_probabilities[3]))
        print(["N","A","O","~"][index])
        
        # Write result to answers.txt
        answers_file = open(path_to_predictions_made_txt,"a")
        answers_file.write("%s,%s\n" % (test_file,["N","A","O","~"][index]))
        answers_file.close()
    
    read_file = pd.read_csv(path_to_predictions_made_txt)
    read_file.to_csv(path_to_predictions_made_csv, index=None)


except KeyboardInterrupt:
    if sys.argv[-1] == 'csv_on_interupt':
        # convert .txt to .csv
        read_file = pd.read_csv(path_to_predictions_made_txt)
        read_file.to_csv(path_to_predictions_made_csv, index=None)
        print('csv created')

# TODO check for duplicates in this array

# Navigate to the Novel Neural Network directory
# In    tf1 conda environment
# python challenge.py ..\physionet_datasets\training2020\training_WFDB csv_on_interupt