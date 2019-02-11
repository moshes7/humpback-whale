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
featuresFileList = [
                    r'../Results/script_featuresExtraction/features_All_examples_without_new_whale.p',
                    # r'../Results/script_featuresExtraction/features_new_whale_examples.p'
                    ]
resultsDir = '../Results/script_calcSavePCA'
sfx =  '_without_new_whale'
# sfx =  '_new_whale_examples'#
NMaxIds = None #10 # maximum number of ids to load, if None - all ids will be used
NComponentsPCA = 4096 # number of pca compnents, should be equal or greater than NMaxIds
testSize = 0.2 # test set percentage
ipcaFlag = True # if True, incremental pca is used, otherwise regular pca is used
ipcaBatchSize = NComponentsPCA #10 # number of features in each ipca iteration, should be equal or grater than NComponentsPCA

os.makedirs(resultsDir, exist_ok=True) # create results dir if not exist

# ------------ load features ------------

counter = 0
featsList = []
idsList = []

for featuresFile in featuresFileList: # iterate over all features files

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

# verify that number of principal components is not greater than number of examples
# verify that the following holds: (see: https://github.com/scikit-learn/scikit-learn/issues/6452)
#   n_samples > n_components > batch_size
if len(trainInd) < NComponentsPCA:
    NComponentsPCA = len(trainInd)

if ipcaBatchSize < NComponentsPCA:
    NComponentsPCA = ipcaBatchSize

# get NComponentsPCA principle components using PCA - using train data only
print('fitting pca to training data...')
if ipcaFlag:
    # incremental PCA - calculate PCA incrementaly in bathces
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
# fileName = os.path.join(resultsDir, 'pca_{}_trainExamples_{}_components{}.p'.format(trainPCA.shape[0], trainPCA.shape[-1], sfx))
# with open(fileName, 'wb') as fid:
#     pickle.dump([pca, trainPCA, valPCA, trainInd, valInd], fid)

# save pca instance
fileName = os.path.join(resultsDir, 'pca_{}_trainExamples_{}_components{}_instance.p'.format(trainPCA.shape[0], trainPCA.shape[-1], sfx))
with open(fileName, 'wb') as fid:
    pickle.dump([pca], fid)

# save transformed features in batches
fileName = os.path.join(resultsDir, 'pca_{}_trainExamples_{}_components{}_features.p'.format(trainPCA.shape[0], trainPCA.shape[-1], sfx))
with open(fileName, 'wb') as fid:
    pickle.dump([trainPCA, valPCA, trainInd, valInd], fid)

# # save transformed train features in batches
# fileName = 'pca_{}_trainExamples_{}_components{}_featuresTrain.p'.format(trainPCA.shape[0], trainPCA.shape[-1], sfx)
# fid = open(os.path.join(resultsDir, fileName), 'wb')
#
# batch_size = 100
# n_examples = trainPCA.shape[0]
# n_batches = int(n_examples / batch_size)
# n_residual = n_examples - batch_size * n_batches
#
# for n in np.arange(n_batches):
#     ind = np.arange(n*batch_size, (n+1)*batch_size)
#     pickle.dump([trainPCA[ind, ...], trainInd[ind, ...]], fid)
#
# if n_residual > 0:
#     ind = np.arange(batch_size * n_batches, n_examples)
#     pickle.dump([trainPCA[ind, ...], trainInd[ind, ...]], fid)
#
# fid.close()

print('done')

print('Done!')
