import os           # os.path.join
import numpy as np  # np.concatenate
import pandas as pd

import bow
import utils

#pd.get_dummies

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer  # fit_transform


DATA_DIR = "data/"

# ONE HOT ENCODE THE COMPLETE TRAINING DATASET

DATA_FILE_TEXT = os.path.join(DATA_DIR, "train_text.csv")           
DATA_FILE_KEYWORD = os.path.join(DATA_DIR, "train_keyword.csv")
DATA_FILE_TARGET = os.path.join(DATA_DIR, "train_target.csv")


def list_to_csv(list, filepath):
    """
    ARGUMENT: list, file path
    RETURN: csv file
    Convert to dataframe first, then convert dataframe to csv file
    """
    #df = pd.DataFrame(list, columns=["text_tokens"])
    df = pd.DataFrame(list)
    df.to_csv(filepath, index=False)
    

# USE LARGEST SET OF BAG OF WORDS

token_list = utils.csv_to_list_of_strings('data/tokensfinal/train_text_token6.csv')
print(token_list[0:10])
print('')

text_list = utils.csv_to_list_of_strings(DATA_FILE_TEXT)
print(text_list[0:3])
print('')

text_token_list = bow.list_of_token_lists_text(text_list, token_list)
print(text_token_list[0:3])


onehot_multi_text = MultiLabelBinarizer()

onehot_text_nolabel = onehot_multi_text.fit_transform(text_token_list)
num_row, num_col = onehot_text_nolabel.shape

print(onehot_text_nolabel)
print(onehot_text_nolabel.shape)
print('')

text_labels = onehot_multi_text.classes_
text_labels = text_labels.reshape(1, num_col)

print(text_labels)
print(text_labels.shape)
print('')

onehot_text = np.concatenate((text_labels,onehot_text_nolabel), axis=0)
print(onehot_text)
print(onehot_text.shape)


keyword_list = utils.csv_to_list_of_strings(DATA_FILE_KEYWORD)
print(keyword_list[0:10])
print('')

keyword_token_list = bow.list_of_token_lists_keyword(keyword_list)
print(keyword_token_list[0:10])


onehot_multi_keyword = MultiLabelBinarizer()
onehot_keyword_nolabel = onehot_multi_keyword.fit_transform(keyword_token_list)
num_row, num_col = onehot_keyword_nolabel.shape

print(onehot_keyword_nolabel)
print(onehot_keyword_nolabel.shape)
print('')

keyword_labels = onehot_multi_keyword.classes_
keyword_labels = keyword_labels.reshape(1, num_col)

print(keyword_labels)
print(keyword_labels.shape)
print('')

onehot_keyword = np.concatenate((keyword_labels,onehot_keyword_nolabel), axis=0)
print(onehot_keyword)
print(onehot_keyword.shape)


onehot_train = np.concatenate((onehot_keyword,onehot_text), axis=1)
print(onehot_train)
print(onehot_train.shape)

train_df = utils.np_array_to_dataframe(onehot_train)
train_target = utils.csv_to_dataframe(DATA_FILE_TARGET)
train_df_with_target = pd.concat([train_df, train_target], axis=1)

DATA_FILE_BOW = os.path.join(DATA_DIR, "bowfinal/train_bow.csv")

utils.dataframe_to_csv(train_df_with_target, DATA_FILE_BOW)


#keyword_labels = keyword_labels.reshape(0, num_col)

keyword_labels = onehot_multi_keyword.classes_
print(keyword_labels.shape)
print(keyword_labels[0:10])


    
DATA_FILE_BOW_KEYWORD = os.path.join(DATA_DIR, "bowfinal/train_bow_keyword.csv")

list_to_csv(keyword_labels, DATA_FILE_BOW_KEYWORD)