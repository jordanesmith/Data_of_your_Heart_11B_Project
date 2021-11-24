#!/usr/bin/python3
import scipy.io;import sys;import numpy as np;import tensorflow as tf; import os; import pandas as pd
from tqdm import tqdm
import math 
# import wfdb 
# wfdb cannot be used because requires newer version of python than is used for this code with tf1 needed to run the models

train_dataset_name = sys.argv[1][3:].replace('\\', '_')


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

# print('loading graphs ...')
# sess,x,y = load_graph('frozen_graph.pb.min');sess1,x1,y1 = load_graph('frozen_graph21.pb');sess2,x2,y2 = load_graph('frozen_graph22.pb');sess3,x3,y3 = load_graph('frozen_graph23.pb');sess4,x4,y4 = load_graph('frozen_graph24.pb');sess5,x5,y5 = load_graph('frozen_graph11.pb');sess6,x6,y6 = load_graph('frozen_graph12.pb');sess7,x7,y7 = load_graph('frozen_graph13.pb')
# print('graphs loaded :)')
# Loop over all files in a given test directory

print('finding names of all test files ...')
all_files = [file_ for file_ in os.listdir(sys.argv[1]) if file_.endswith('.mat')]
print('all files found :)')

counter = 0
for test_file in tqdm(all_files):
    counter += 1
    # # Read waveform samples (input is in WFDB-MAT format)
    samples = scipy.io.loadmat(os.path.join(sys.argv[1], test_file))['val'][0]
    max_length = 145 #TODO extract this from metadata folder instead
    sample_freq = 500 #TODO as above
    sample_length = len(samples)/ sample_freq
    print('sample_length: ',sample_length)


    for i in range(math.ceil(sample_length/58)):

        time_range = [i*58, (i+1)*58]
        index_range = [int(index * sample_freq) for index in time_range ]
        print(time_range, index_range)
        sample_section = samples[index_range[0], index_range[1]]


    # # # Your classification algorithm goes here...
    Sxx = (samples-7.51190822991)/235.404723927
    Sxx_all = np.array([Sxx[i:(i+5*sample_freq)] for i in range(0,len(Sxx)-5*sample_freq+1,sample_freq)])[:,:,None] #original sample_freq = 300
    pred_prob = np.array(sess.run(y,{x:Sxx_all}))



    try: 
        Sxx_all = np.pad(pred_prob,((0,max_length-len(pred_prob)),(0,0)),mode="constant") # original max_length = 58
        if counter % 5 == 0:
            print('\nThis prediction probability worked {} ******************'.format(test_file))
            print(len(pred_prob))
            print(pred_prob)

    except ValueError:
        # Clearly this occurs when the smaple ecg is too long, need to find the new maximum 
        print('\nError here', '_'*20)
        print(len(pred_prob))
        print(pred_prob)

    p1,p2,p3,p4,p5,p6,p7 = sess1.run(y1,{x1:[Sxx_all]}),sess2.run(y2,{x2:[Sxx_all]}),sess3.run(y3,{x3:[Sxx_all]}),sess4.run(y4,{x4:[Sxx_all]}),sess5.run(y5,{x5:[Sxx_all]}),sess6.run(y6,{x6:[Sxx_all]}),sess7.run(y7,{x7:[Sxx_all]})
    index = [1 if np.argmax(p5+p6+p7) == 1 else np.argmax(p1+p2+p3+p4)][0]

    # Write result to answers.txt
    answers_file = open("{}answers.txt".format(train_dataset_name),"a")
    answers_file.write("%s,%s\n" % (test_file,["N","A","O","~"][index]))
    answers_file.close()

# convert .txt to .csv
read_file = pd.read_csv("{}answers.txt".format(train_dataset_name))
read_file.to_csv("{}answers.csv".format(train_dataset_name), index=None)