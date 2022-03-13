import numpy as np 
import pandas as pd

def find_f1_score(predictions, labels):
    """
    Calculates performance metriks for 'A' or 'N' categorisation from provided
    predictions and labels numpy arrays.
    
    returns:
        (f1_score: float, precision: float, recall: float)
    """
    
    assert predictions.shape[0] == predictions.shape[0]
    
    N = predictions.shape[0]
    
    arr_predictions_labels_only = np.zeros((N,2), dtype=str)
    counter = 0
    for filename, pred_ in predictions:
        label_ = labels[labels[:,0] == filename][0][1]
        arr_predictions_labels_only[counter, 0] = label_ 
        arr_predictions_labels_only[counter, 1] = pred_
        counter += 1
    
    tp = 0 # no. true positives
    fp = 0 # no. false positives
    fn = 0 # no. false negatives
    tn = 0 # no. true negatives
    
    for prediction_label_pair in arr_predictions_labels_only:
        pred, lab = prediction_label_pair
        if pred == "N" and lab == "N": tn += 1
        elif pred == "A" and lab == "A": tp += 1
        elif pred == "N" and lab == "A": fn += 1
        elif pred == "A" and lab == "N": fp += 1
    
    
    # Calculate accuracy metriks for AF diagnosis
    A_diagnosis_metriks_calculated = False
    try:
        precision_A = tp / (tp + fp)
        recall_A = tp / (tp + fn)
        f1_score_A = 2 / ( recall_A**(-1) + precision_A**(-1) )
        A_diagnosis_metriks_calculated = True
    except ZeroDivisionError:
        print("Performance metriks for 'A' diagnosis unsuccessful due to divide by 0 error")
    
    
    # Calculate accuracy metriks for Not AF diagnosis
    N_diagnosis_metriks_calculated = False
    try:
        precision_N = tn / (tn + fn)
        recall_N = tn / (tn + fp)
        f1_score_N = 2 / ( recall_N**(-1) + precision_N**(-1) )
        N_diagnosis_metriks_calculated = True
    except ZeroDivisionError:
        print("Performance metriks for 'N' diagnosis unsuccessful due to divide by 0 error")
        
        
    # return results 
    if A_diagnosis_metriks_calculated and N_diagnosis_metriks_calculated:
        return [("A_values:", f1_score_A, precision_A, recall_A), (("N_values:", f1_score_N, precision_N, recall_N))]
    elif A_diagnosis_metriks_calculated and not N_diagnosis_metriks_calculated:
        return [("A_values:", f1_score_A, precision_A, recall_A)]
    elif not A_diagnosis_metriks_calculated and N_diagnosis_metriks_calculated:
        return [("N_values:", f1_score_N, precision_N, recall_N)]
