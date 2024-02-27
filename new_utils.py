"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np
from typing import Type, Dict
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
)

def scale_data(X):
    #X = X.astype(float)
    X = (X - X.min()) / (X.max() - X.min())
    return X


def conf_mat_accuracy(matrix):
    """
    Calculate accuracy from a confusion matrix.
    """
    TruePositive = matrix[1, 1]  
    TrueNegative = matrix[0, 0]  
    total_samples = matrix.sum()
    accuracy = (TruePositive+ TrueNegative) / total_samples
    return accuracy

def remove_90_9s(X: NDArray[np.floating], y: NDArray[np.int32]):
 
    nine_idx = (y == 9)

    X_90 = X[nine_idx, :]
    y_90 = y[nine_idx]

    X_90=X_90[:int((X_90.shape[0])*0.1),:]
    y_90=y_90[:int((y_90.shape[0])*0.1)]
    
    none_nine= (y!=9)
    X_non_9 = X[none_nine, :]
    y_non_9 = y[none_nine]
    
    fin_X=np.concatenate((X_non_9,X_90),axis=0)
    fin_y=np.concatenate((y_non_9,y_90),axis=0)
    
    return fin_X, fin_y


def convert_7_0(X: NDArray[np.floating], y: NDArray[np.int32]):
   id_7=(y==7)
   id_0=(y==0)
   y[id_7]=0

   return X,y

def convert_9_1(X: NDArray[np.floating], y: NDArray[np.int32]):
   id_9=(y==9)
   id_1=(y==1)
   y[id_9]=1

   return X,y

def train_simple_classifier_with_cv(
    *,
    Xtrain: NDArray[np.floating],
    ytrain: NDArray[np.int32],
    clf: BaseEstimator,
    cv: KFold = KFold,
):
    """
    Train a simple classifier using k-vold cross-validation.

    Parameters:
        - X: Features dataset.
        - y: Labels.
        - cv_class: The cross-validation class to use.
        - estimator_class: The training classifier class to use.
        - n_splits: Number of splits for cross-validation.
        - print_results: Whether to print the results.

    Returns:
        - A dictionary with mean and std of accuracy and fit time.
    """
    scores = cross_validate(clf, Xtrain, ytrain, cv=cv, scoring='accuracy')
    return scores
