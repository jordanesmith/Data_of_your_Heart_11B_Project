import pandas as pd
import re

def remove_pretext_from_filename_in_answers_txt(str_to_remove, path_to_answers):

    path_to_answers = path_to_answers + ".txt"

    my_file = open(path_to_answers)
    string_list = my_file.readlines()

    string_list = [re.sub(str_to_remove, '', string_line) for string_line in string_list]

    my_file = open(path_to_answers, "w")
    new_file_contents = "".join(string_list)

    my_file.write(new_file_contents)
    my_file.close()

def remove_ending_from_filename_in_answers_csv(str_to_remove, path_to_answers):

    path_to_answers = path_to_answers + ".csv"

    df_answers = pd.read_csv(path_to_answers, names=['filename', 'label'])

    df_answers.filename = df_answers.filename.apply(lambda f_: re.sub(str_to_remove, '', f_))

    df_answers.to_csv(path_to_answers)


def reclassify_from_complex_diagnosis_to_4_classifications(path_to_answers):

    df_answers = pd.read_csv(path_to_answers + ".csv", names=['filename', 'label'])

    def reclassify(label):

        if label == 'sinus rhythm' or label == 'N':
            return 'N'
        elif 'atrial fibrillation' in label or label == 'A':
            return 'A'
        else:
            return 'O' # no ~ option because don't know which label corresponds to it


    df_answers['label'] = df_answers['label'].apply(reclassify)

    df_answers.to_csv(path_to_answers + '_reclassified.csv')


path_to_answers = "Novel Neural Network/physionet_datasets_training2020_training_WFDBanswers_reclassified"

# reclassify_from_complex_diagnosis_to_4_classifications(path_to_answers)
# remove_pretext_from_filename_in_answers_txt("..\\\\physionet_datasets\\\\training2017\\\\", path_to_answers)
# remove_ending_from_filename_in_answers_csv(".mat", path_to_answers)