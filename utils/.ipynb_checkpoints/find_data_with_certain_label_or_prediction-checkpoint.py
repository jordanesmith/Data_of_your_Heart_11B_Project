import pandas as pd  

def find_all_data_with_given_label_and_preiction_label(label, prediction_label, labels_df, predictions_df):

    df_data_with_given_label = labels_df.loc[labels_df.label == label]

    df_data_with_given_prediction_label = predictions_df.loc[predictions_df.label == prediction_label]

    common_files = list(set(df_data_with_given_label.filename.values.tolist()).intersection(df_data_with_given_prediction_label.filename.values.tolist()))
    common_files.sort()

    return common_files

# path_to_labels = 'physionet_datasets\\\\training2017\\\\REFERENCE.csv'
# path_to_predictions = 'Novel Neural Network\\\\answers.csv'

# print(find_all_data_with_given_label_and_preiction_label('A', 'N', labels_df=pd.read_csv(path_to_labels, names=['filename', 'label']), predictions_df=pd.read_csv(path_to_predictions, names=['filename', 'label'])))