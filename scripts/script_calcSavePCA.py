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
from sklearn.decomposition import PCA
from sklearn import model_selection
import pickle

# ------------ parameters ------------
# featuresFile = r'../Results/script_featuresExtraction/features_3_examples.p'
featuresFile = r'../Results/script_featuresExtraction/features_All_examples.p'
resultsDir = '../Results/script_clusterByTrainFeatrues'
NComponentsPCA = 4096 # number of pca compnents

os.makedirs(resultsDir, exist_ok=True) # create results dir if not exist

# ------------ load features ------------

counter = 0
featsList = []
idsList = []
with open(featuresFile, 'rb') as fid:
    try:
        while True:
            id, feats = pickle.load(fid)
            idsList.append(np.repeat(id, len(feats)))
            featsList.append(feats)
            counter += 1
            print('reading features {}'.format(counter))
    except EOFError:
        pass

# flatten lists
print('flattening lists ...')
featsList = [item for sublist in featsList for item in sublist]
idsList = [item for sublist in idsList for item in sublist]
print('done')

# convert lists to ndarrays
print('converting lists to ndarrays ...')
feats = np.asarray(featsList)
ids = np.asarray(idsList)
print('done')

# ------------ Pre-Processing ------------

# normalize each feature to have zero mean and unit variance
print('whitening features ...')
feats = (feats - feats.mean(axis=1)[:,np.newaxis]) / feats.std(axis=1)[:,np.newaxis]
print('done')

# ------------ Split to Train and Validation Sets ------------

print('spliting data to train and val ...')
sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainInd, valInd = next(sss.split(X=feats, y=ids))
print('done')

# ------------ PCA ------------

# get NComponentsPCA principle components using PCA - using train data only
print('fitting pca to training data...')
pca = PCA(n_components=NComponentsPCA)
# principalComponents = pca.fit_transform(feats[trainInd])
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
fileName = os.path.join(resultsDir, 'pca_{}_components.p'.format(valPCA.shape[-1]))
with open(fileName, 'wb') as fid:
    pickle.dump([pca, trainPCA, valPCA, trainInd, valInd], fid)
print('done')

print('Done!')
