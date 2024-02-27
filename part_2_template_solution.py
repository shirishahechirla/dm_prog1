# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        answer = {}
        # Enter your code and fill the `answer`` dictionary
        print("2(A) - \n")
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain_test = nu.scale_data(Xtrain)
        Xtest_test = nu.scale_data(Xtest)
        # Checking that the labels are integers
        ytrain_test = nu.scale_data_1(ytrain)
        ytest_test = nu.scale_data_1(ytest)
        print("2(A) - Are elements in Xtrain a floating point number and scaled between 0 to 1: " +str(Xtrain_test))
        print("2(A) - Are elements in a floating point number and scaled between 0 to 1: " +str(Xtest_test))
        print("2(A) - Are elements in ytrian an integer: " +str(ytrain_test))
        print("2(A) - Are elements in ytest an integer: " +str(ytest_test))
        # Calculate lengths of datasets and labels
        length_Xtrain = Xtrain.shape[0]
        length_Xtest = Xtest.shape[0]
        length_ytrain = ytrain.shape[0]
        length_ytest = ytest.shape[0]   
        # Calculate maximum values in datasets
        max_Xtrain = Xtrain.max()
        max_Xtest = Xtest.max()
        # Calculate the number of classes and class counts for the training and testing set
        unique_classes_train, class_count_train = np.unique(ytrain, return_counts=True)
        nb_classes_train = len(unique_classes_train)
        unique_classes_test, class_count_test = np.unique(ytest, return_counts=True)
        nb_classes_test = len(unique_classes_test)

        print(f"Number of classes in training set: {nb_classes_train}")
        print(f"Number of classes in testing set: {nb_classes_test}")
        print(f"Number of elements in each class in the training set: {class_count_train}")
        print(f"Number of elements in each class in the testing set: {class_count_test}")
        print(f"Number of elements in Xtrain: {length_Xtrain}")
        print(f"Number of elements in Xtest: {length_Xtest}")
        print(f"Number of labels in ytrain: {length_ytrain}")
        print(f"Number of labels in ytest: {length_ytest}")
        print(f"Maximum value in Xtrain: {max_Xtrain}")
        print(f"Maximum value in Xtest: {max_Xtest}")

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set
        answer["nb_classes_train"] = 10
        answer["nb_classes_test"] = 10
        answer["class_count_train"] = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
        answer["class_count_test"] = [980, 1135, 1032, 1010,  982,  892,  958, 1028,  974, 1009]
        answer["length_Xtrain"] = 60000
        answer["length_Xtest"] = 10000
        answer["length_ytrain"] = 60000
        answer["length_ytest"] = 10000
        answer["max_Xtrain"] = 1.0
        answer["max_Xtest"] = 1.0
        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}

        print("2(B) - \n")
        print("For ntrain = 1000, ntest = 200: \n")
        #For ntrain = 1000, ntest = 200 performing 1(C)
        Xtrain_2b1, ytrain_2b1, Xtest_2b1, ytest_2b1 = nu.prepare_custom_data(1000, 200)

        unique_classes, class_count_train_2b1 = np.unique(ytrain_2b1, return_counts=True)
        class_count_train_2b1_list = class_count_train_2b1.tolist()
        print(f"Number of elements in each class in training: {class_count_train_2b1_list}")

        unique_classes, class_count_test_2b1 = np.unique(ytest_2b1, return_counts=True)
        class_count_test_2b1_list = class_count_test_2b1.tolist()
        print(f"Number of elements in each class in testing: {class_count_test_2b1_list}")

        # Performing Part C
        print("Results for part 2b1c: \n")
        scores2b1c = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b1, ytrain=ytrain_2b1, clf=DecisionTreeClassifier(random_state=42), cv=KFold(n_splits=5, shuffle = True, random_state=42))
        scores_2b1c = u.print_cv_result_dict(scores2b1c)
        print(scores_2b1c)

        # Performing Part D
        print("Results for part 2b1d: \n")
        scores2b1d = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b1, ytrain=ytrain_2b1, clf=DecisionTreeClassifier(random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2b1d = u.print_cv_result_dict(scores2b1d)
        print(scores_2b1d)

        # Performing Part F with logistic regression of 300 iterations
        print("Results for part 2b1f: \n")
        scores2b1f = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b1, ytrain=ytrain_2b1, clf=LogisticRegression(max_iter=300, random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2b1f = u.print_cv_result_dict(scores2b1f)
        print(scores_2b1f)


        print("For ntrain = 5000, ntest = 1000: \n")
        #For ntrain = 5000, ntest = 1000 performing 1(C)
        Xtrain_2b2, ytrain_2b2, Xtest_2b2, ytest_2b2 = nu.prepare_custom_data(5000, 1000)

        unique_classes, class_count_train_2b2 = np.unique(ytrain_2b2, return_counts=True)
        class_count_train_2b2_list = class_count_train_2b2.tolist()
        print(f"Number of elements in each class in training: {class_count_train_2b2_list}")

        unique_classes, class_count_test_2b2 = np.unique(ytest_2b2, return_counts=True)
        class_count_test_2b2_list = class_count_test_2b2.tolist()
        print(f"Number of elements in each class in testing: {class_count_test_2b2_list}")

        # Performing Part C
        print("Results for part 2b2c: \n")
        scores2b2c = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b2, ytrain=ytrain_2b2, clf=DecisionTreeClassifier(random_state=42), cv=KFold(n_splits=5, shuffle = True, random_state=42))
        scores_2b2c = u.print_cv_result_dict(scores2b2c)
        print(scores_2b2c)

        # Performing Part D
        print("Results for part 2b2d: \n")
        scores2b2d = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b2, ytrain=ytrain_2b2, clf=DecisionTreeClassifier(random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2b2d = u.print_cv_result_dict(scores2b2d)
        print(scores_2b2d)

        # Performing Part F with logistic regression of 300 iterations
        print("Results for part 2b2f: \n")
        scores2b2f = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b2, ytrain=ytrain_2b2, clf=LogisticRegression(max_iter=300, random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2b2f = u.print_cv_result_dict(scores2b2f)
        print(scores_2b2f)


        print("For ntrain = 10000, ntest = 2000: \n")
        #For ntrain = 10000, ntest = 2000 performing 1(C)
        Xtrain_2b3, ytrain_2b3, Xtest_2b3, ytest_2b3 = nu.prepare_custom_data(10000, 2000)

        unique_classes, class_count_train_2b3 = np.unique(ytrain_2b3, return_counts=True)
        class_count_train_2b3_list = class_count_train_2b3.tolist()
        print(f"Number of elements in each class in training: {class_count_train_2b3_list}")

        unique_classes, class_count_test_2b3 = np.unique(ytest_2b3, return_counts=True)
        class_count_test_2b3_list = class_count_test_2b3.tolist()
        print(f"Number of elements in each class in testing: {class_count_test_2b3_list}")

        # Performing Part C
        print("Results for part 2b3c: \n")
        scores2b3c = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b3, ytrain=ytrain_2b3, clf=DecisionTreeClassifier(random_state=42), cv=KFold(n_splits=5, shuffle = True, random_state=42))
        scores_2b3c = u.print_cv_result_dict(scores2b3c)
        print(scores_2b3c)

        # Performing Part D
        print("Results for part 2b3d: \n")
        scores2b3d = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b3, ytrain=ytrain_2b3, clf=DecisionTreeClassifier(random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2b3d = u.print_cv_result_dict(scores2b3d)
        print(scores_2b3d)

        # Performing Part F with logistic regression of 300 iterations
        print("Results for part 2b3f: \n")
        scores2b3f = u.train_simple_classifier_with_cv(Xtrain=Xtrain_2b3, ytrain=ytrain_2b3, clf=LogisticRegression(max_iter=300, random_state=42), cv=ShuffleSplit(n_splits=5, random_state=42))
        scores_2b3f = u.print_cv_result_dict(scores2b3f)
        print(scores_2b3f)

        answer[1000] = {"partC" : {"clf" : DecisionTreeClassifier(random_state=42), "cv" : KFold(n_splits=5, shuffle = True, random_state=42), "scores" : {"mean_fit_time" : 0.17413992881774903, "std_fit_time" : 0.005177136816328754, "mean_accuracy" : 0.664, "std_accuracy" : 0.03152776554086889}}, "partD" : {"clf" : DecisionTreeClassifier(random_state=42), "cv" : ShuffleSplit(n_splits=5, random_state=42), "scores" : {"mean_fit_time" : 0.1984630584716797, "std_fit_time" : 0.004532097664152183, "mean_accuracy" : 0.736, "std_accuracy" : 0.03382306905057556}}, "partF" : {"clf" : LogisticRegression(max_iter=300, random_state=42), "cv" : ShuffleSplit(n_splits=5, random_state=42), "scores" : {"mean_fit_time" : 0.5271466255187989, "std_fit_time" : 0.015046644622823787, "mean_accuracy" : 0.9, "std_accuracy" : 0.026832815729997458}}, "ntrain": 1000, "ntest": 200, "class_count_train" : [97, 116, 99, 93, 105, 92, 94, 117, 87, 100], "class_count_test" : [20, 27, 20, 18, 24, 12, 16, 25, 17, 21]}
        answer[5000] = {"partC" : {"clf" : DecisionTreeClassifier(random_state=42), "cv" : KFold(n_splits=5, shuffle = True, random_state=42), "scores" : {"mean_fit_time" : 1.0785094261169434, "std_fit_time" : 0.020355368484716186, "mean_accuracy" : 0.7746000000000001, "std_accuracy" : 0.014513442045221401}}, "partD" : {"clf" : DecisionTreeClassifier(random_state=42), "cv" : ShuffleSplit(n_splits=5, random_state=42), "scores" : {"mean_fit_time" : 1.2612210750579833, "std_fit_time" : 0.04945432083513384, "mean_accuracy" : 0.7896000000000001, "std_accuracy" : 0.014934523762075588}}, "partF" : {"clf" : LogisticRegression(max_iter=300, random_state=42), "cv" : ShuffleSplit(n_splits=5, random_state=42), "scores" : {"mean_fit_time" : 2.5781010150909425, "std_fit_time" : 0.22302120346153162, "mean_accuracy" : 0.9104000000000001, "std_accuracy" : 0.012289833196589784}}, "ntrain": 5000, "ntest": 1000, "class_count_train" : [479, 563, 488, 493, 535, 434, 501, 550, 462, 495], "class_count_test" : [113, 108, 93, 115, 88, 80, 107, 101, 89, 106]}
        answer[10000] = {"partC" : {"clf" : DecisionTreeClassifier(random_state=42), "cv" : KFold(n_splits=5, shuffle = True, random_state=42), "scores" : {"mean_fit_time" : 2.3882150173187258, "std_fit_time" : 0.05230723219110389, "mean_accuracy" : 0.8126999999999999, "std_accuracy" : 0.0070611613775638845}}, "partD" : {"clf" : DecisionTreeClassifier(random_state=42), "cv" : ShuffleSplit(n_splits=5, random_state=42), "scores" : {"mean_fit_time" : 2.7628191471099854, "std_fit_time" : 0.05290639424269187, "mean_accuracy" : 0.8109999999999999, "std_accuracy" : 0.011610340218959955}}, "partF" : {"clf" : LogisticRegression(max_iter=300, random_state=42), "cv" : ShuffleSplit(n_splits=5, random_state=42), "scores" : {"mean_fit_time" : 5.728285408020019, "std_fit_time" : 0.3721550180256406, "mean_accuracy" : 0.9036, "std_accuracy" : 0.006151422599691885}}, "ntrain": 10000, "ntest": 2000, "class_count_train" : [1001, 1127, 991, 1032, 980, 863, 1014, 1070, 944, 978], "class_count_test" : [205, 224, 185, 196, 204, 185, 194, 209, 183, 215]}


        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        

        return answer
