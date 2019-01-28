from __future__ import print_function, division, absolute_import
import numpy as np
from data_generation.DataGeneration import DataGeneration

# simple script to develop and verify implementation of DataGeneration.stratifiedSplit()

# create labels
y = np.array([ 1,
               2, 2,
               3, 3, 3,
               4, 4, 4, 4,
               5, 5, 5, 5, 5,
               ])

test_size=0.2
random_state=42

train_ind, test_ind = DataGeneration.stratifiedSplit(y, test_size=test_size, random_state=random_state)

print('train set: {}'.format(y[train_ind]))
print('test set: {}'.format(y[test_ind]))

print('Done!')
