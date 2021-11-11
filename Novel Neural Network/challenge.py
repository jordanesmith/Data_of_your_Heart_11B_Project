#!/usr/bin/python3
import scipy.io;import sys;import numpy as np;import tensorflow as tf; import os; import pandas as pd

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

sess,x,y = load_graph('frozen_graph.pb.min');sess1,x1,y1 = load_graph('frozen_graph21.pb');sess2,x2,y2 = load_graph('frozen_graph22.pb');sess3,x3,y3 = load_graph('frozen_graph23.pb');sess4,x4,y4 = load_graph('frozen_graph24.pb');sess5,x5,y5 = load_graph('frozen_graph11.pb');sess6,x6,y6 = load_graph('frozen_graph12.pb');sess7,x7,y7 = load_graph('frozen_graph13.pb')

# Loop over all files in a given test directory

all_files = [file_ for file_ in os.listdir(sys.argv[1]) if file_.endswith('.mat')]

for test_file in all_files:

# # Read waveform samples (input is in WFDB-MAT format)
    # record = sys.argv[1]
    print(test_file)
    samples = scipy.io.loadmat(os.path.join(sys.argv[1], test_file))['val'][0]

    # # # Your classification algorithm goes here...
    Sxx = (samples-7.51190822991)/235.404723927
    Sxx_all = np.array([Sxx[i:(i+1500)] for i in range(0,len(Sxx)-1500+1,300)])[:,:,None]
    pred_prob = np.array(sess.run(y,{x:Sxx_all}))
    Sxx_all = np.pad(pred_prob,((0,58-len(pred_prob)),(0,0)),mode="constant")

    p1,p2,p3,p4,p5,p6,p7 = sess1.run(y1,{x1:[Sxx_all]}),sess2.run(y2,{x2:[Sxx_all]}),sess3.run(y3,{x3:[Sxx_all]}),sess4.run(y4,{x4:[Sxx_all]}),sess5.run(y5,{x5:[Sxx_all]}),sess6.run(y6,{x6:[Sxx_all]}),sess7.run(y7,{x7:[Sxx_all]})
    index = [1 if np.argmax(p5+p6+p7) == 1 else np.argmax(p1+p2+p3+p4)][0]

    # Write result to answers.txt
    answers_file = open("answers.txt","a")
    answers_file.write("%s,%s\n" % (test_file,["N","A","O","~"][index]))
    answers_file.close()

# convert .txt to .csv
read_file = pd.read_csv('answers.txt')
read_file.to_csv('answers.csv', index=None)