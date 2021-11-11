import pandas as pd



def find_label(data_filename, labels_df):
    """Find label for given filename from a reference csv file  

    Parameters
    ----------
    path_to_labels : str
        path to label csv
    data_filename : str
        filename without .mat extension 

    Returns
    -------
    str
        Label 
    """
    

    label = labels_df.loc[labels_df['filename'] == data_filename].label.to_string(index = False)
    
    return label

    
# path_to_labels = 'physionet_datasets/training2017/REFERENCE.csv'

# labels_csv = pd.read_csv(path_to_labels, names=['filename', 'label'])

# print(find_label(path_to_labels, 'A00001'))