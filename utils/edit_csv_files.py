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

path_to_answers = "Novel Neural Network/answers"
# str_to_remove = "..\\\\physionet_datasets\\\\training2017\\\\"
# remove_pretext_from_filename_in_answers_txt(str_to_remove, path_to_answers)

def remove_ending_from_filename_in_answers_csv(str_to_remove, path_to_answers):

    path_to_answers = path_to_answers + ".csv"

    df_answers = pd.read_csv(path_to_answers, names=['filename', 'label'])

    df_answers.filename = df_answers.filename.apply(lambda f_: re.sub(str_to_remove, '', f_))

    df_answers.to_csv(path_to_answers)

str_to_remove = ".mat"

remove_ending_from_filename_in_answers_csv(str_to_remove, path_to_answers)