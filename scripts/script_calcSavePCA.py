from __future__ import print_function, division, absolute_import

__author__ = 'Moshe Shilemay'
__license__ = 'MIT'
__email__ = "moshes777@gmail.com"
'''
    Last modified: 27.01.209
    Python Version: 3.6
'''

import numpy as np
import os
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import model_selection
from sklearn import preprocessing
import pickle
from data_generation.DataGeneration import DataGeneration

# ------------ parameters ------------
# featuresFile = r'../Results/script_featuresExtraction/features_3_examples.p'
featuresFile = r'../Results/script_featuresExtraction/features_All_examples.p'
resultsDir = '../Results/script_calcSavePCA'
NMaxIds = None #10 # maximum number of ids to load, if None - all ids will be used
NComponentsPCA = 4096 # number of pca compnents
testSize = 0.2 # test set percentage
ipcaFlag = True # if True, incremental pca is used, otherwise regular pca is used
ipcaBatchSize = 100 # number of features in each ipca iteration

os.makedirs(resultsDir, exist_ok=True) # create results dir if not exist

# ------------ load features ------------

counter = 0
featsList = []
idsList = []
with open(featuresFile, 'rb') as fid:
    try:
        while True:
            id, feats = pickle.load(fid)
            idsList.extend(np.repeat(id, len(feats)))
            featsList.extend(feats)
            counter += 1
            print('reading features {}'.format(counter))
            if (NMaxIds is not None) and (counter >= NMaxIds):
                break
    except EOFError:
        pass

# convert lists to ndarrays
print('converting lists to ndarrays ...')
feats = np.asarray(featsList)
ids = np.asarray(idsList)
print('done')

# ------------ Pre-Processing ------------

# normalize each feature to have zero mean and unit variance
print('whitening features ...')
# feats = (feats - feats.mean(axis=1)[:,np.newaxis]) / feats.std(axis=1)[:,np.newaxis]
scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True)
feats = scaler.fit_transform(feats)
print('done')

# ------------ Split to Train and Validation Sets ------------

print('spliting data to train and val ...')
trainInd, valInd = DataGeneration.stratifiedSplit(y=ids, test_size=testSize, random_state=42)
print('done')

# ------------ PCA ------------

# get NComponentsPCA principle components using PCA - using train data only
print('fitting pca to training data...')
if ipcaFlag:
    # incremental PCA - calculate PCA incrementaly in bathces
    # FIXME: ValueError: Number of input features has changed from 100 to 4096 between calls to partial_fit! Try setting n_components to a fixed value.
    pca = IncrementalPCA(n_components=NComponentsPCA, batch_size=ipcaBatchSize)
    pca.fit(feats[trainInd])
else:
    # regular PCA - loads all data into memory
    pca = PCA(n_components=NComponentsPCA)
    pca.fit(feats[trainInd])
print('done')

print('{} componenets PCA - explained variance = {}'.format(NComponentsPCA, pca.explained_variance_ratio_[0]))

# transform
print('transforming train and val data according to pca...')
trainPCA = pca.transform(feats[trainInd])
valPCA = pca.transform(feats[valInd])
print('done')

# save pca
print('saving pca data ...')
fileName = os.path.join(resultsDir, 'pca_{}_trainExamples_{}_components.p'.format(trainPCA.shape[0], trainPCA.shape[-1]))
with open(fileName, 'wb') as fid:
    pickle.dump([pca, trainPCA, valPCA, trainInd, valInd], fid)
print('done')

print('Done!')
