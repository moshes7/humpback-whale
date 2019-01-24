from __future__ import print_function, division, absolute_import

import pretrainedmodels # see https://github.com/Cadene/pretrained-models.pytorch#xception
import torch
import pretrainedmodels.utils as utils

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# print the available pretrained models
print(pretrainedmodels.model_names)

# print the available pretrained settings for a chosen model
print(pretrainedmodels.pretrained_settings['xception'])

# load a pretrained models from imagenet
# option 1
# model_name = 'xception'
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# option 2
model = pretrainedmodels.models.xception(num_classes=1000, pretrained='imagenet').to(device)
model.eval()

# load an image and do a complete forward pass
load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model)

path_img = './data/cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor, requires_grad=False).to(device)

output_logits = model(input) # 1x1000

# extract features (beware this API is not available for all networks)
output_features = model.features(input) # 1x14x14x2048 size may differ
output_logits = model.logits(output_features) # 1x1000

# convert to ndarray
if device.type == 'cuda':
    feat = output_features.cpu().detach().numpy()
else:
    feat = output_features.detach().numpy()

print(feat.shape)
feat = feat.squeeze()
print(feat.shape)
print(feat.size)
