from __future__ import print_function, division, absolute_import

__author__ = 'Moshe Shilemay'
__license__ = 'MIT'
__email__ = "moshes777@gmail.com"
'''
    Last modified: 25.01.209
    Python Version: 3.6
'''

import pretrainedmodels # see https://github.com/Cadene/pretrained-models.pytorch#xception
import torch
import pretrainedmodels.utils as utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from PIL import Image
from sklearn.decomposition import PCA
import pickle

# ------------ parameters ------------
dataDir = r'C:\Users\Moshe\Sync\Data\humpback-whale\train'
labelsFile = r'C:\Users\Moshe\Sync\Data\humpback-whale\train.csv'
resultsDir = '../Results/script_featuresExtraction'
# idsToCheck = ['w_f48451c', 'w_c3d896a', 'w_20df2c5', 'w_dd88965', 'w_64404ac'] # None
# idsToCheck = ['w_f48451c', 'w_20df2c5', 'w_dd88965'] # None
idsToCheck = None
sfx =  '_All_examples'# ''
saveFeatures = True # if True features will be saved to file
display = False

os.makedirs(resultsDir, exist_ok=True) # create results dir if not exist

# ------------ load labels ------------

# load labels
df = pd.read_csv(labelsFile)
print(df.info())

# delete rows with 'new_whale' id
df = df.loc[df['Id'] != 'new_whale']
print(df.info())

# find unique ids
idUnique = df['Id'].unique().tolist()

# filter list according to idsToCheck
if idsToCheck is not None:
    idUnique = [id for id in idUnique if id in idsToCheck]

# ------------ load network ------------

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load Xception model pre-trained on imagenet
model = pretrainedmodels.models.xception(num_classes=1000, pretrained='imagenet').to(device)
model.eval() # set model mode to eval

# define image loading and pre-processing functions
load_img = utils.LoadImage(space='L') # image loading function. space='L' - convert images to grayscale
tf_img = utils.TransformImage(model) # image pre-processing function: transformations depending on the model rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)

# ------------ extract features ------------

if saveFeatures:
    featFileName = 'features{}.p'.format(sfx)
    fid = open(os.path.join(resultsDir, featFileName), 'wb')

featDict = {}
counter = 0
for id in idUnique: # iterate over unique ids

    counter += 1
    print('extracting features of id {}: {}/{}'.format(id ,counter, len(idUnique)))

    # find all images with this id
    idsCurrent = df.loc[df['Id'] == id]
    imageList = idsCurrent['Image'].tolist()

    featList = []
    idList = []
    for imageName in imageList: # iterate over all examples with the same id

        # load image
        imgFile = os.path.join(dataDir, imageName)
        img = load_img(imgFile) # grayscale image

        # duplicate graysacle img to create 3 color channels
        imgN = np.array(img)
        imgN = np.stack([img, img ,img], axis=-1)
        img = Image.fromarray(imgN)

        # transform image
        imgT = tf_img(img)

        # add dimension
        imgT = imgT.unsqueeze(0)  # 3x299x299 -> 1x3x299x299

        # create Tensor variable
        input = torch.autograd.Variable(imgT, requires_grad=False).to(device)

        # load image to device (gpu/cpu)
        input = input.to(device)

        # TODO: stack images of same id to a single batch

        # extract features
        output_features = model.features(input)  # 1x14x14x2048 size may differ

        # convert to ndarray
        if device.type == 'cuda':
            feat = output_features.cpu().detach().numpy()
        else:
            feat = output_features.detach().numpy()

        featList.append(feat.squeeze())
        idList.append(id)

    # convert list to ndarray
    featN = np.asarray(featList) # convert list to ndarray
    featN = np.reshape(featN, (featN.shape[0],-1)) # reshape to 1 feature vector per img

    if saveFeatures:
        pickle.dump([id, featN], fid)

    if display: # do not save features if not displaying them in order to save internal memory
        # save features vector
        featDict[id] = featN

# close features file
if saveFeatures:
    fid.close()

if display:

    # some processing before PCA - all features should be located in one array

    # convert dict of lists to list of lists
    featsList = []
    idsList = []
    for id, feat in featDict.items():
        featsList.append(feat)
        idsList.append(np.repeat(id, feat.shape[0]))

    # flatten lists
    featsList = [item for sublist in featsList for item in sublist]
    idsList = [item for sublist in idsList for item in sublist]

    # convert lists to ndarrays
    feats = np.asarray(featsList)
    ids = np.asarray(idsList)

    # get 2 principle components using PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(feats)

    # TODO: use 3 components - should use 3D scatterplot, see: https://matplotlib.org/gallery/mplot3d/scatter3d.html
    # TODO: add tSNE visualization

    # display
    fontSize = 20
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=fontSize)
    ax.set_ylabel('Principal Component 2', fontsize=fontSize)
    strTitle = '2 Components PCA{}'.format(sfx)
    ax.set_title(strTitle, fontsize=fontSize)

    colors = cm.rainbow(np.linspace(0, 1, len(idUnique)))

    for n, id in enumerate(idUnique):
        ind = np.where(ids == id)[0]
        x = principalComponents[ind, 0]
        y = principalComponents[ind, 1]
        ax.scatter(x, y, c=colors[n, 0:3], s=50)

    ax.legend(idUnique, fontsize=fontSize)
    ax.grid()
    plt.show(block=False); plt.pause(1e-2)

    # enlarge figure
    try:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.pause(1e-2)
    except:
        pass

    plt.savefig(os.path.join(resultsDir, '{}.jpg'.format(strTitle)), bbox_inches='tight')

print('Done!')
