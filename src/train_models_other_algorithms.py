import pandas as pd
import os
import time
import utils
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection as ms

from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import BernoulliNB 
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble



DATA_DIR = "../data/"
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "bowfinal/train_bow.csv")
DATA_FILE_TEST = os.path.join(DATA_DIR, "bowfinal/test_bow.csv")

df = utils.csv_to_dataframe(DATA_FILE_TRAIN)

c = len(df.columns)

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


print(type(x_train))
print(type(y_train))


# EVALUATION

num_folds = 5
seed = 11
scoring = 'f1'

kfold = ms.KFold(n_splits=num_folds, random_state=seed, shuffle=True)


# NAIVE BAYES MULTINOMIAL

start = time.time()

parameters = {'alpha':[10,35,36]}
multinb = MultinomialNB()
MultiNBModel = (GridSearchCV(multinb, parameters)).fit(x_train, y_train)
scoreMultiNB = cross_val_score(MultiNBModel,  x_train, y_train, cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoreMultiNB.mean(), "Parameters: ", MultiNBModel.best_params_, "Time Elapsed: ",  elapsed_time)


# NAIVE BAYES BERNOULLI

start = time.time()

parameters = {'alpha':[1.5]}
bernnb = BernoulliNB()
BernNBModel = (GridSearchCV(bernnb, parameters)).fit(x_train, y_train)
scoreBernNB = cross_val_score(BernNBModel,  x_train, y_train, cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoreBernNB.mean(), "Parameters: ", BernNBModel.best_params_, "Time Elapsed: ",  elapsed_time)


# LINEAR SVC

start = time.time()

parameters = { 'C':[0.01, 0.1, 1, 10], 'loss':('hinge', 'squared_hinge'),
              'multi_class':('ovr', 'crammer_singer') }
linearsvc = LinearSVC()
LinearSVCModel = (GridSearchCV(linearsvc, parameters)).fit(x_train, y_train)
scoreLinearSVC = cross_val_score(LinearSVCModel,  x_train, y_train, cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoreLinearSVC.mean(), "Parameters: ", LinearSVCModel.best_params_, "Time Elapsed: ",  elapsed_time)


# KNN

start = time.time()

parameters = {'n_neighbors':[5, 7, 11]}
knn = neighbors.KNeighborsClassifier()
KNNModel = (GridSearchCV(knn, parameters)).fit(x_train, y_train)
scoreKNN = cross_val_score(KNNModel,  x_train, y_train,cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoreKNN.mean(), "Parameters: ", KNNModel.best_params_, "Time Elapsed: ",  elapsed_time)


# DECISION TREE

start = time.time()

parameters = {'min_samples_split':[2, 3 ,4]}
decisiontc =  tree.DecisionTreeClassifier()
DecisionTreeCModel = (GridSearchCV(decisiontc, parameters)).fit(x_train, y_train)
scoreDecisionTreeC = cross_val_score(DecisionTreeCModel,  x_train, y_train,cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoreDecisionTreeC.mean(), "Parameters: ", DecisionTreeCModel.best_params_, "Time Elapsed: ",  elapsed_time)


# RANDOM FOREST

start = time.time()

randomfc1 =  ensemble.RandomForestClassifier(n_estimators  = 500)
randomForestC1Model = randomfc1.fit(x_train, y_train)
scorerandomForestC1 = cross_val_score(randomForestC1Model,  x_train, y_train,cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scorerandomForestC1.mean(), "Time Elapsed: ",  elapsed_time)


# ADABOOST
start = time.time()

parameters = {'n_estimators':[50,75, 100]}
adaboost = ensemble.AdaBoostClassifier()
adaBoostModel = (GridSearchCV(adaboost, parameters)).fit(x_train, y_train)
scoreadaBoost = cross_val_score(adaBoostModel,  x_train, y_train,cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoreadaBoost.mean(), "Parameters: ", adaBoostModel.best_params_, "Time Elapsed: ",  elapsed_time)


# GRADIENT BOOSTING

start = time.time()

parameters = {'learning_rate':[1, 1.5,2], 'n_estimators':[50, 75, 100]}
gradientboostc = ensemble.GradientBoostingClassifier()
gradientBoostCModel = (GridSearchCV(gradientboostc, parameters)).fit(x_train, y_train)
scoregradientBoostC = cross_val_score(gradientBoostCModel,  x_train, y_train,cv= kfold, scoring = scoring)

elapsed_time = time.time() - start

print("F1 Score: ",  scoregradientBoostC.mean(), "Parameters: ", gradientBoostCModel.best_params_,
      "Time Elapsed: ",  elapsed_time)



# KAGGLE TEST SET
df_test = utils.csv_to_dataframe(DATA_FILE_TEST)

column_labels = list(range(0, c-1))
df_test.columns = column_labels

array_test = df_test.values
x_test = array_test[:,0:c]   # feature values
print(type(x_test))
print(x_test.shape)



# NAIVE BAYES BERNOULLI PREDICTIONS

bowBernModel =  BernoulliNB(alpha =  3.75)
bowBernModel.fit(x_train, y_train)

predictions = bowBernModel.predict(x_test)
predictions_list = np.array(predictions).tolist()

print(type(predictions_list))
print(len(predictions_list))
print(predictions_list[0:10])


# SUBMISSION FILE

DATA_FILE_TEST_ID = os.path.join(DATA_DIR, "test_id.csv")

test_id_list = utils.csv_to_list_of_strings(DATA_FILE_TEST_ID)

print(len(test_id_list))
print(test_id_list[0:10])

df_test_id = utils.csv_to_dataframe(DATA_FILE_TEST_ID)

df_test_predict = pd.DataFrame({'col':predictions_list})
df_test_predict.columns = ['target']

df_submit = pd.concat([df_test_id, df_test_predict], axis=1)

DATA_FILE_SUBMIT = os.path.join(DATA_DIR, "submissionBernBow375.csv")

utils.dataframe_to_csv(df_submit, DATA_FILE_SUBMIT)