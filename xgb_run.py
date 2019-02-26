import math
from IPython import display
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib #for saving the trained model

######

# timing helper functions

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

#####
print("loading the data...")
data = pd.read_csv('PS_20174392719_1491204439457_log.csv')

#####

print("preparing the data...")

norm_step = np.array(data['step'])
norm_amount = np.array(data['amount'])
norm_newbalanceOrig = np.array(data['newbalanceOrig'])
norm_newbalanceDest = np.array(data['newbalanceDest'])

norm_step = StandardScaler().fit_transform(norm_step.reshape(-1,1))
norm_amount = StandardScaler().fit_transform(norm_step.reshape(-1,1))
norm_newbalanceOrig = StandardScaler().fit_transform(norm_newbalanceOrig.reshape(-1,1))
norm_newbalanceDest = StandardScaler().fit_transform(norm_newbalanceDest.reshape(-1,1))

data['norm_step'] = norm_step
data['norm_amount'] = norm_amount
data['norm_newbalanceOrig'] = norm_newbalanceOrig
data['norm_newbalanceDest'] = norm_newbalanceDest

# encode the transaction type
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
cols_to_transform = ['type']
type_hash = pd.get_dummies(data=data['type'])
#data['type_hash'] = type_hash

data = pd.concat([data, type_hash], axis=1)

del data['step']
del data['amount']
del data['newbalanceOrig']
del data['newbalanceDest']
del data['nameOrig']
del data['nameDest']
del data['type']

#####

#randomize the data to be sure not to have any pathological ordering effects that might impact on gradient descent.
data = data.reindex(
    np.random.permutation(data.index))

#####

# split the features and target vector for resampling
X = data.loc[:, data.columns != 'isFraud']
y = data.loc[:, data.columns == 'isFraud']

#####

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#####

X_train_resampled = pickle.load(open( "X_train_resampled.pkl", "rb" ))
y_train_resampled = pickle.load(open( "y_train_resampled.pkl", "rb" ))


####

reg = xgb.XGBRegressor(n_estimators=25, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=0.9, random_state=8418, eval_metric='rmse')


print('fitting model...')
tic()
reg.fit(X_train_resampled, y_train_resampled)

toc()
y_pred = reg.predict(X_test)

#save the model
joblib.dump(reg, 'xgb_model.pkl')

ac = accuracy_score(y_test,reg.predict(X_test))
print('Accuracy is: ',ac)

print('done!')