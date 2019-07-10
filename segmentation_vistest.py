# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import cv2
import PIL.Image as Image

# In[2]:
PATH = os.path.abspath('results')

SOURCE_IMAGES = [os.path.join(PATH, "day2night_inno_cyclegan/test_latest/images")]

images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    
images.sort()

print("Datasets used: {}\n".format(SOURCE_IMAGES))
    
labels = []

for i in range(0, len(images), 2):
    labels.append(images[i].replace("results/day2night_inno_cyclegan/test_latest/images", "datasets/day2night_inno/labels").replace("_fake_B",""))

print(len(images))
print(len(labels))

# In[]
from PIL import Image

def get_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img
#    return img[16:,8:632]

def get_label(path):
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    return img
#    return img[16:,8:632]

# In[]:
from keras.utils import to_categorical

num_classes = 3
input_shape = (256, 640, 3)
#input_shape = (240, 624, 3)

# In[ ]:
# # Define model
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet, PSPNet

backbone = 'resnet18'
#backbone = 'seresnet101'
preprocessing_fn = get_preprocessing(backbone)

model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax',  encoder_weights='imagenet')
#model = PSPNet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax',  encoder_weights='imagenet')

weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 13-47-28.hdf5"

model.load_weights(weights_path)

print("Model summary:")
model.summary()

model._make_predict_function()

# In[ ]:
from keras import optimizers
from segmentation_models.losses import dice_loss, cce_dice_loss
from segmentation_models.metrics import dice_score
from losses import dice_coef_multiclass_loss

learning_rate = 1e-4
optimizer = optimizers.Adam(learning_rate)

losses = [dice_coef_multiclass_loss]
metrics = ['categorical_accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[]:
i = 140
x = get_image(images[2*i+1])

y_true = get_label(labels[i])

x2 = np.expand_dims(x, axis = 0)
x2 = preprocessing_fn(x2)

y_pred = model.predict(x2)
y_pred = np.argmax(y_pred, axis = -1)
y_pred = np.squeeze(y_pred)

fig, axes = plt.subplots(nrows = 3, ncols = 1)
axes[0].imshow(x)
axes[1].imshow(y_true)
axes[2].imshow(y_pred)
fig.tight_layout()
