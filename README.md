# Kaggle bank fraud dataset

Machine learning model predicting fraudulent bank transactions in the Kaggle bank fraud dataset.

The Kaggle dataset is available here: https://www.kaggle.com/ntnu-testimon/paysim1

This is a work in progress notebook for me to learn to use Tensorflow to implement neural network models.

## SciKit Learn

I also explored a more traditional ML approach using scikit learn and documented this in scratch-scikit. 

I have managed to achieve 98.5% accuracy using XGBoost which, at time of writing, I can't beat with other methods such as linear regression and stochastic gradient descent. 

Training time is a few minutes on a 2017 MacBook Pro and the pickled model file is ~2mb, making this (in my opinion) an efficient solution.