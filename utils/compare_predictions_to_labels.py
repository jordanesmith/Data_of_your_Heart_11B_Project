import numpy as np
import pandas as pd 
import sklearn
from sklearn.metrics import classification_report
import tqdm

labels = pd.read_csv('physionet_datasets/training2017/REFERENCE.csv', names=['filename', 'label'])

answers_from_NNN_approach = pd.read_csv('Novel Neural Network/answers.csv', names=['filename', 'label'])

filename_label_groundtruth = []


for filename, answer in tqdm.tqdm(zip(answers_from_NNN_approach['filename'], answers_from_NNN_approach['label'])):
    
    ground_truth = labels.loc[labels['filename'] == filename[:-4]]['label'].to_string(index=False).strip()
    # if ground_truth != answer: print(ground_truth, answer, type(ground_truth), type(answer), len(ground_truth), len(answer))
    filename_label_groundtruth.append([filename, answer, ground_truth])

df = pd.DataFrame(filename_label_groundtruth, columns=["filename", "predicted_label", "ground_truth_label"])

print(sklearn.metrics.classification_report(df["ground_truth_label"], df["predicted_label"]))
