import numpy as np
import pandas as pd 
import sklearn
from sklearn.metrics import classification_report
import tqdm
import os
import wfdb
from find_label_for_data import get_label_from_dx_code, get_label_from_diagnosis


labels_2017 = pd.read_csv('physionet_datasets/training2017/REFERENCE.csv', names=['filename', 'label'])
answers_from_NNN_approach_2017 = pd.read_csv('Novel Neural Network/answers.csv', names=['filename', 'label'])
dataset_2017 = False

dataset_path = "physionet_datasets\\training2020\\training_WFDB" 
answers_from_NNN_approach_2020 = pd.read_csv('predictions\\physionet_datasets_training2020_training_WFDBanswers_reclassified.csv', names=['filename', 'label'])
dataset_2020 = True

if dataset_2017: answers = answers_from_NNN_approach_2017
elif dataset_2020: answers = answers_from_NNN_approach_2020 

filename_label_groundtruth = []
for filename, answer in tqdm.tqdm(zip(answers['filename'], answers['label']), total=len(answers['filename'])):
    if dataset_2017:
        ground_truth = labels_2017.loc[labels_2017['filename'] == filename]['label'].to_string(index=False).strip()
        filename_label_groundtruth.append([filename, answer, ground_truth])
    elif dataset_2020:
        path_to_hea = os.path.join(dataset_path, filename.replace('.mat', '.hea'))
        record = wfdb.rdrecord(path_to_hea.split('.hea',2)[0], channels=[1])
        dx_codes = [com_.split(' ',2)[-1] for com_ in record.comments if 'Dx' in com_][0]
        for dx_code in dx_codes.split(','):
            diagnosis = get_label_from_dx_code(int(dx_code))
            ground_truth = get_label_from_diagnosis(diagnosis)
            if ground_truth == 'N' or ground_truth == 'A':
                filename_label_groundtruth.append([filename, answer, ground_truth])        
        #TODO ignore files labelled 'O' by model because they could also be '~', see edit_csv_files for more
        

df = pd.DataFrame(filename_label_groundtruth, columns=["filename", "predicted_label", "ground_truth_label"])
print(sklearn.metrics.classification_report(df["ground_truth_label"], df["predicted_label"], labels=['N', 'A']))
