import os           # os.path.join
import numpy as np  # np.concatenate
import pandas as pd

import bow
import utils

#pd.get_dummies

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer  # fit_transform


# train_text9.csv, train_keyword9.csv are SMALL DATASETS used to test the code

DATA_DIR = "../data/"

DATA_FILE_TEXT = os.path.join(DATA_DIR, "dataset_small/train_text9.csv")           
DATA_FILE_KEYWORD = os.path.join(DATA_DIR, "dataset_small/train_keyword9.csv")


# (1) ONE HOT ENCODE TEXT** (TEST THE CODE USING SMALL DATASET)

# CHECK
s = utils.file_to_string(DATA_FILE_TEXT)
print(s)

text_list1 = utils.csv_to_list_of_lists(DATA_FILE_TEXT)
print(text_list1)

#token_list = utils.csv_to_list_of_strings('data/tokens/train_text_token100.csv')
token_list = utils.csv_to_list_of_strings('data/tokens_dataset_small/train_text_token.csv')
print(token_list[0:20])
print('')

text_list = utils.csv_to_list_of_strings(DATA_FILE_TEXT)
print(text_list)
print('')

text_token_list = bow.list_of_token_lists_text(text_list, token_list)
print(text_token_list)

onehot_multi_text = MultiLabelBinarizer()

onehot_text_nolabel = onehot_multi_text.fit_transform(text_token_list)
num_row, num_col = onehot_text_nolabel.shape

print(onehot_text_nolabel)
print(onehot_text_nolabel.shape)
#print(type(onehot_text_nolabel))
print('')

text_labels = onehot_multi_text.classes_
text_labels = text_labels.reshape(1, num_col)

print(text_labels)
print(text_labels.shape)
#print(type(text_labels))
print('')

onehot_text = np.concatenate((text_labels,onehot_text_nolabel), axis=0)
print(onehot_text)
print(onehot_text.shape)



# ONE HOT ENCODE KEYWORDS** (USING SMALL DATASET)
# CHECK

s = utils.file_to_string(DATA_FILE_KEYWORD)
print(s)
print('')

keyword_list1 = bow.clean_doc_keyword(s)
print(keyword_list1)


keyword_list = utils.csv_to_list_of_strings(DATA_FILE_KEYWORD)
print(keyword_list)
print('')

keyword_token_list = bow.list_of_token_lists_keyword(keyword_list)
print(keyword_token_list)


onehot_multi_keyword = MultiLabelBinarizer()
onehot_keyword_nolabel = onehot_multi_keyword.fit_transform(keyword_token_list)
num_row, num_col = onehot_keyword_nolabel.shape

print(onehot_keyword_nolabel)
print(onehot_keyword_nolabel.shape)
#print(type(onehot_keyword_nolabel))
print('')

keyword_labels = onehot_multi_keyword.classes_
keyword_labels = keyword_labels.reshape(1, num_col)

print(keyword_labels)
print(keyword_labels.shape)
#print(type(keyword_labels))
print('')

onehot_keyword = np.concatenate((keyword_labels,onehot_keyword_nolabel), axis=0)
print(onehot_keyword)
print(onehot_keyword.shape)


# CONCATENATE KEYWORD AND TEXT ONE HOT DATA

onehot_train = np.concatenate((onehot_keyword,onehot_text), axis=1)
print(onehot_train)
print(onehot_train.shape)


# CONVERT NUMPY ARRAY TO DATAFRAME

train_df = utils.np_array_to_dataframe(onehot_train)



DATA_FILE_BOW = os.path.join(DATA_DIR, "bow/train_bow9.csv")

utils.dataframe_to_csv(train_df, DATA_FILE_BOW)