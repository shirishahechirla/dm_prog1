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
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
)

#1-B
def scale_data(X_bi = NDArray[np.floating]):
    # Check if all elements are floating-point numbers and within the range [0, 1]
    if not issubclass(X_bi.dtype.type, np.floating) or (X_bi < 0).any() or (X_bi > 1).any():
        return False

    return True

#1-B
def scale_data_1(y_bi = NDArray[np.int32]):
    # Check if the elements in y are integers or not
    if not issubclass(y_bi.dtype.type, np.int32):
        return False
    
    return True

def print_cv_result_dict_test(cv_dict: Dict):
    for key, array in cv_dict.items():
        if key not in ['fit_time', 'score_time']:
            print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")

def load_mnist_dataset(
    nb_samples=None,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """
    Load the MNIST dataset.

    nb_samples: number of samples to save. Useful for code testing.
    The homework requires you to use the full dataset.

    Returns:
        X, y
        #X_train, y_train, X_test, y_test
    """

    try:
        # Are the datasets already loaded?
        print("... Is MNIST dataset local?")
        X: NDArray[np.floating] = np.load("mnist_X.npy")
        y: NDArray[np.int32] = np.load("mnist_y.npy", allow_pickle=True)
    except Exception as e:
        # Download the datasets
        print(f"load_mnist_dataset, exception {e}, Download file")
        X, y = datasets.fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )
        X = X.astype(float)
        y = y.astype(int)

    y = y.astype(np.int32)
    X: NDArray[np.floating] = X
    y: NDArray[np.int32] = y

    if nb_samples is not None and nb_samples < X.shape[0]:
        X = X[0:nb_samples, :]
        y = y[0:nb_samples]

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    np.save("mnist_X.npy", X)
    np.save("mnist_y.npy", y)
    return X, y

def prepare_custom_data(ntrain, ntest, normalize: bool = True):
    # Check in case the data is already on the computer.
    X, y = load_mnist_dataset()

    # won't work well unless X is greater or equal to zero
    if normalize:
        X = X / X.max()

    y = y.astype(np.int32)
    Xtrain = X[0:ntrain, :]
    ytrain = y[0:ntrain]
    Xtest = X[ntrain:ntrain+ntest]
    ytest = y[ntrain:ntrain+ntest]
    return Xtrain, ytrain, Xtest, ytest

def filter_imbalanced_7_9s(X, y):
    # Filter out only 7s and 9s
    seven_nine_idx = (y == 7) | (y == 9)
    X_binary = X[seven_nine_idx]
    y_binary = y[seven_nine_idx]

    # Convert 7s to 0s and 9s to 1s
    y_binary = np.where(y_binary == 7, 0, 1)

    # Remove 90% of 9s
    nines_idx = np.where(y_binary == 1)[0]
    remove_n = int(len(nines_idx) * 0.9)  # 90% to remove
    np.random.shuffle(nines_idx)
    remove_idx = nines_idx[:remove_n]

    # Keep only the desired indices
    keep_idx = np.setdiff1d(np.arange(len(X_binary)), remove_idx)
    X_imbalanced = X_binary[keep_idx]
    y_imbalanced = y_binary[keep_idx]

    return X_imbalanced, y_imbalanced
