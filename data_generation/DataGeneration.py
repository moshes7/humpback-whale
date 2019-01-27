from __future__ import print_function, division, absolute_import

__author__ = 'Moshe Shilemay'
__license__ = 'MIT'
__email__ = "moshes777@gmail.com"
'''
    Last modified: 27.01.209
    Python Version: 3.6
'''

import numpy as np
from sklearn.utils import check_random_state

class DataGeneration:

    def __init__(self):
        pass

    @staticmethod
    def stratifiedSplit(y, test_size=0.2, random_state=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        test_size : float, optionl
            Float in the interval [0,1], controls on what percentage of the data
            will be used for test set.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        """

        # get random state
        random_state = check_random_state(random_state)

        # find unique classes
        classes, y_indices, y_inverse, y_counts = np.unique(y, return_index=True, return_inverse=True, return_counts=True)

        # split to train and test
        train = []
        test = []
        for ind, count in zip(y_indices, y_counts): # iterate over all classes

            # number of elements to be added to test set
            Ntest = np.round(test_size * count)

            if count == 1: # if there is only 1 example in class - add it to training set
                train.append(ind)
            else:

                # sample test indices without replacement
                ind_test = random_state.choice(ind, size=Ntest, replace=False)

                # get train indices
                ind_train = np.setdiff1d(ind, ind_test)

                # add indices to train and test lists
                train.append(ind_train)
                test.append(ind_test)

        return train, test