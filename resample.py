import math
from IPython import display
import numpy as np
import pandas as pd
import pickle

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

cc = 4

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

# set up parallel cluster using ipyparallel engines.
# ipcluster start
import ipyparallel as ipp
from ipyparallel import require
c = ipp.Client(profile='default')
v = c[:]
#print("engines running:",v)
print("engines running:",len(v))

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

from imblearn.over_sampling import ADASYN#SMOTE
sm = ADASYN(n_jobs=cc)

# split the features and target vector for resampling
X = data.loc[:, data.columns != 'isFraud']
y = data.loc[:, data.columns == 'isFraud']

#####

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#resmaple just the training data and leave the test data untouched

print("resampling...")
tic()
X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)
toc()
# this step takes a bit of time so I pickled the output

print("saving the data...")
pickle.dump(X_train_resampled, open( "X_train_resampled.pkl", "wb" ))
pickle.dump(y_train_resampled, open( "y_train_resampled.pkl", "wb" ))

#####

print("done!!")