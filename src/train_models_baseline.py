import pandas as pd
import os
import time
import utils

import numpy as np

from sklearn import linear_model
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import svm
from sklearn import ensemble

from sklearn import model_selection as ms
from sklearn import metrics


DATA_DIR = "../data/"
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "bowfinal/train_bow.csv")
DATA_FILE_TEST = os.path.join(DATA_DIR, "bowfinal/test_bow.csv")


df = utils.csv_to_dataframe(DATA_FILE_TRAIN)

c = len(df.columns)

print(len(df.columns))
print(len(df.index))

column_labels = list(range(0, c))
df.columns = column_labels


array = df.values
x = array[:,0:c-1]   # feature values
y = array[:,c-1]     # targets

#validation_size = 0.0000001
#seed = 11
#x_train, x_validation, y_train, y_validation = ms.train_test_split(x, y, test_size=validation_size, random_state=seed)

x_train = x
y_train = y

print(x.shape)
print(y.shape)
print('')
print(x_train.shape)
#print(x_validation.shape)
print('')
print(y_train.shape)
#print(y_validation.shape)

print(type(x_train))
print(type(y_train))


# all using default hyperparameters

models =[]
models.append(('LogReg', linear_model.LogisticRegression(C=0.3)))
models.append(('GaussNB', naive_bayes.GaussianNB()))
#models.append(('KNN', neighbors.KNeighborsClassifier()))
#models.append(('CART', tree.DecisionTreeClassifier()))
#models.append(('SVM', svm.SVC(gamma='auto')))
#models.append(('RandomF100', ensemble.RandomForestClassifier(n_estimators=100)))
#models.append(('RandomF200', ensemble.RandomForestClassifier(n_estimators=200)))
#models.append(('AdaBoost', ensemble.AdaBoostClassifier()))
#models.append(('GradBoost', ensemble.GradientBoostingClassifier()))


num_folds = 5
seed = 11
scoring = 'f1'

results = []
names = []

for name, model in models:
    start = time.time()
    kfold = ms.KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = ms.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    elapsed_time = time.time() - start
    
    results.append(cv_results)
    names.append(name)
    
    # print name, mean F1, standard deviation of accuracy, time taken
    print(f'{name}: \t {cv_results.mean()} \t {cv_results.std()} \t {elapsed_time}')
    

# TRAIN MODEL ON ENTIRE TRAINING DATASET
    
model_logreg = linear_model.LogisticRegression(C=0.3)
model_logreg.fit(x_train, y_train)

#predictions = model_logreg.predict(x_validation)
#score = metrics.accuracy_score(y_validation, predictions)
#matrix = metrics.confusion_matrix(y_validation, predictions)
#report = metrics.classification_report(y_validation, predictions)

#print('Logistic Regression')
#print(score)
#print(matrix)
#print(report)


# KAGGLE TEST SET

df_test = utils.csv_to_dataframe(DATA_FILE_TEST)

column_labels = list(range(0, c-1))
df_test.columns = column_labels

array_test = df_test.values
x_test = array_test[:,0:c]   # feature values
print(type(x_test))
print(x_test.shape)


predictions = model_logreg.predict(x_test)
predictions_list = np.array(predictions).tolist()

print(type(predictions_list))
print(len(predictions_list))
print(predictions_list[0:10])


DATA_FILE_TEST_ID = os.path.join(DATA_DIR, "test_id.csv")

test_id_list = utils.csv_to_list_of_strings(DATA_FILE_TEST_ID)

print(len(test_id_list))
print(test_id_list[0:10])


df_test_id = utils.csv_to_dataframe(DATA_FILE_TEST_ID)


df_test_predict = pd.DataFrame({'col':predictions_list})
df_test_predict.columns = ['target']

df_submit = pd.concat([df_test_id, df_test_predict], axis=1)


DATA_FILE_SUBMIT = os.path.join(DATA_DIR, "submission.csv")

utils.dataframe_to_csv(df_submit, DATA_FILE_SUBMIT)