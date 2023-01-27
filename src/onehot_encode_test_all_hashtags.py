import os           # os.path.join
import numpy as np  # np.concatenate
import pandas as pd

import bow
import utils

#pd.get_dummies

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer  # fit_transform

DATA_DIR = "../data/"

# ONE HOT ENCODE THE TESTING DATA (NO TARGET LABELS)

DATA_FILE_TEXT = os.path.join(DATA_DIR, "test_text.csv")           
DATA_FILE_KEYWORD = os.path.join(DATA_DIR, "test_keyword.csv")
DATA_FILE_TARGET = os.path.join(DATA_DIR, "test_target.csv")

# USE LARGEST SET OF BAG OF WORDS

token_list = utils.csv_to_list_of_strings('data/tokensfinal/train_text_token10hashuser3.csv')
print(token_list[0:10])
print(len(token_list))
print('')

keyword_list = utils.csv_to_list_of_strings('data/bowfinal10/train_bow_keyword.csv')
print(keyword_list[0:10])
print(len(keyword_list))


text_list = utils.csv_to_list_of_strings(DATA_FILE_TEXT)
print(text_list[0:3])
print('')

text_token_list1 = bow.list_of_token_lists_text(text_list, token_list)
print(text_token_list1[0:3])
print(len(text_token_list1))
print('')

token_list1 = [token_list]
text_token_list = token_list1 + text_token_list1 
print(text_token_list[1:3])
print(len(text_token_list))


onehot_multi_text = MultiLabelBinarizer()

onehot_text_nolabel = onehot_multi_text.fit_transform(text_token_list)
num_row, num_col = onehot_text_nolabel.shape

print(onehot_text_nolabel)
print(onehot_text_nolabel.shape)
print('')

onehot_text_nolabel = np.delete(onehot_text_nolabel, (0), axis=0)
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


'''
text_list = utils.csv_to_list_of_strings(DATA_FILE_TEXT)
print(text_list[0:3])
print('')

text_token_list1 = bow.list_of_token_lists_text(text_list, token_list)
print(text_token_list1[0:3])
print(len(text_token_list1))
print('')

token_list1 = [token_list]
text_token_list = token_list1 + text_token_list1 
print(text_token_list[1:3])
print(len(text_token_list))
'''



keyword_list_test = utils.csv_to_list_of_strings(DATA_FILE_KEYWORD)
print(keyword_list_test[0:10])
print(len(keyword_list_test))
print('')


keyword_token_list1 = bow.list_of_token_lists_keyword(keyword_list_test)
print(keyword_token_list1[0:10])
print(len(keyword_token_list1))
print('')

keyword_list1 = [keyword_list]
print(len(keyword_list))
print(len(keyword_list1))
print('')

keyword_token_list = keyword_list1 + keyword_token_list1 
print(keyword_token_list[1:3])
print(len(keyword_token_list))


onehot_multi_keyword = MultiLabelBinarizer()
onehot_keyword_nolabel = onehot_multi_keyword.fit_transform(keyword_token_list)
num_row, num_col = onehot_keyword_nolabel.shape

print(onehot_keyword_nolabel)
print(onehot_keyword_nolabel.shape)
print('')


onehot_keyword_nolabel = np.delete(onehot_keyword_nolabel, (0), axis=0)
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


onehot_test = np.concatenate((onehot_keyword,onehot_text), axis=1)
print(onehot_test)
print(onehot_test.shape)


test_df = utils.np_array_to_dataframe(onehot_test)


# SAVE

DATA_FILE_BOW = os.path.join(DATA_DIR, "bowfinal10/test_bow.csv")

utils.dataframe_to_csv(test_df, DATA_FILE_BOW)