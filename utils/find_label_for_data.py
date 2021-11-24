import pandas as pd

def find_label_from_df(data_filename, labels_df):
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


def get_label_from_dx_code(dx_code, path_to_ct_codes_csv="SNOMED_CT_codes.csv"):
    """
    Translate ct code to diagnosis label 

    Parameters
    ----------
    dx_code : int
        
    path_to_ct_codes_csv : str, optional
        by default "SNOMED_CT_codes.csv"

    Returns
    -------
    code: str
        the diagnosis label
    """    

    df = pd.read_csv(path_to_ct_codes_csv)

    code = df.loc[df["SNOMED CT Code"] == dx_code].Dx.to_string(index=False)
    
    return code