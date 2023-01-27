import time
import os
import pandas as pd

from sklearn import model_selection
from sklearn import metrics

from xgboost import XGBClassifier

import matplotlib
from matplotlib import pyplot
matplotlib.use('Agg')

import utils


DATA_DIR = "../data/"
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "bowfinal/train_bow.csv")
DATA_FILE_TEST = os.path.join(DATA_DIR, "bowfinal/test_bow.csv")

df_train = pd.read_csv(DATA_FILE_TRAIN)
c = len(df_train.columns)

array_train = df_train.values       # np array

print(array_train)
print(type(array_train))


x = array_train[:,0:c-1]   # feature values
y = array_train[:,c-1]     # targets

x_train = x
y_train = y

print(x.shape)
print(y.shape)
print('')
print(x_train.shape)
print(y_train.shape)
print('')
print(x_train)
print(y_train)


num_folds = 5
seed = 11
scoring = 'f1'

model_xgb = XGBClassifier(colsample_bylevel=0.5, 
                          colsample_bytree=0.5, 
                          gamma=0, 
                          learning_rate=0.1, 
                          max_depth=8, 
                          min_child_weight=1.0, 
                          n_estimators=500, 
                          subsample=1)


start = time.time()
kfold = model_selection.KFold(n_splits=num_folds, random_state=seed, shuffle=True)
cv_results = model_selection.cross_val_score(model_xgb, x_train, y_train, cv=kfold, scoring=scoring)
elapsed_time = time.time() - start
    
#results.append(cv_results)
#names.append(name)
    
# print name, mean F1, standard deviation of accuracy, time taken
print(f'F1 (mean, std): \t {cv_results.mean()} \t {cv_results.std()} \t Time: {elapsed_time}')


# TRAIN

model_xgb.fit(x_train, y_train)

#predict_x = model_xgb.predict(x_validation)
#predict_round = [round(value) for value in predict_x]
#accuracy_x = metrics.accuracy_score(y_validation, predict_x)
#print(f'XGBoost Accuracy: {accuracy_x}')


# KAGGLE TEST SET

df_test = utils.csv_to_dataframe(DATA_FILE_TEST)

column_labels = list(range(0, c-1))
df_test.columns = column_labels

array_test = df_test.values
x_test = array_test[:,0:c]   # feature values
print(type(x_test))
print(x_test.shape)


# PREDICTIONS

predict_xgb = model_xgb.predict(x_test)
predict_xgb_round = [round(value) for value in predict_xgb]

predictions_list = predict_xgb_round

print(type(predictions_list))
print(len(predictions_list))
print(predictions_list[0:10])


# SUBMISSION FILE
DATA_FILE_TEST_ID = os.path.join(DATA_DIR, "test_id.csv")

test_id_list = utils.csv_to_list_of_strings(DATA_FILE_TEST_ID)

print(len(test_id_list))
print(test_id_list[0:10])

df_test_id = utils.csv_to_dataframe(DATA_FILE_TEST_ID)

#predictions_list = predict_xgb_round

df_test_predict = pd.DataFrame({'col':predictions_list})
df_test_predict.columns = ['target']

df_submit = pd.concat([df_test_id, df_test_predict], axis=1)

DATA_FILE_SUBMIT = os.path.join(DATA_DIR, "submission.csv")

utils.dataframe_to_csv(df_submit, DATA_FILE_SUBMIT)