# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
import numpy as np
import cv2
import PIL.Image as Image
import pickle
from keras import optimizers
from losses import dice_coef_multiclass_loss, dice_coef_multiclass
from metrics_for_paper import mAccuracy, mPrecision, mRecall, mIU, mF1
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet
from tqdm import tqdm
from glob import glob

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# READ IMAGES AND MASKS
# In[2]:
PATH = os.path.abspath('datasets')

losstype = 'mydice'
vis = False

imgs_dir = "images"

SOURCE_IMAGES = [os.path.join(PATH, "day2night_inno/", imgs_dir)]

day_images = []

for si in SOURCE_IMAGES:
    day_images.extend(glob(os.path.join(si, "*.png")))
    
day_images.sort()
    
day_labels = []

for i in range(len(day_images)):
    day_labels.append(day_images[i].replace(imgs_dir, "labels"))
    
imgs_dir = "trainB"
    
SOURCE_IMAGES = [os.path.join(PATH, "day2night_inno/", imgs_dir)]

night_images = []

for si in SOURCE_IMAGES:
    night_images.extend(glob(os.path.join(si, "*.png")))

night_images.sort()
    
night_labels = []

for i in range(len(night_images)):
    night_labels.append(night_images[i].replace(imgs_dir, "labels"))

print("Total day images count: {}".format(len(day_images)))
print("Total day labels count: {}\n".format(len(day_labels)))
print("Total night images count: {}".format(len(night_images)))
print("Total night labels count: {}\n".format(len(night_labels)))

# In[]
input_shape = (256, 640, 3)
num_classes = 3
vis = True
numvis = 50
mode = 'night'

def get_image(path):
    img = Image.open(path)
    img = img.resize((input_shape[1], input_shape[0]))
    img = np.array(img) 
    return img    

if mode == 'day':
    images = day_images
    labels = day_labels
elif mode == 'night':
    images = night_images
    labels = night_labels

# In[ ]:
backbone = 'resnet18'
preprocessing_fn = get_preprocessing(backbone)
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')

weights = []

################### DICE LOSS from segmentation_models AUGMODE0
weights_path = "weights/segmentation_linknet_resnet18/2019-05-22 09-10-59.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 12-36-48.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 14-33-14.hdf5"
weights.append(weights_path)

################### DICE LOSS from segmentation_models AUGMODE1
weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 15-06-19.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 07-49-16.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 12-35-18.hdf5"
weights.append(weights_path)

################### DICE LOSS from segmentation_models AUGMODE2
weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 15-06-07.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 13-57-52.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 13-58-59.hdf5"
weights.append(weights_path)

################### DICE LOSS from segmentation_models AUGMODE3
weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 13-47-28.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 08-22-50.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 07-48-43.hdf5"
weights.append(weights_path)

################### DICE LOSS from segmentation_models AUGMODE4
weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 08-23-52.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 09-59-55.hdf5"
weights.append(weights_path)
weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 10-00-46.hdf5"
weights.append(weights_path)

# In[ ]:
learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = [dice_coef_multiclass_loss]
metrics = [dice_coef_multiclass]

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

# In[]:
dlina = len(images)

wc = 0

for w in tqdm(weights):
    
    model.load_weights(w)
    model._make_predict_function()
    model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

    mAccuracy_ = 0
    mPrecision_ = 0
    mRecall_ = 0
    mIU_ = 0
    mF1_ = 0
    
    for i in tqdm(range(dlina)):
        
        x = get_image(images[i])
        
        if vis and i%numvis==0:
            x_vis = x.copy()
            
        x = np.expand_dims(x, axis = 0)
        x = preprocessing_fn(x)
        
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis = -1)
        y_pred = np.squeeze(y_pred)
    
        if vis and i%numvis==0:
            y_pred_vis = y_pred.astype(np.uint8)
            y_pred_vis *= 255//2
            overlay_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(y_pred_vis,cv2.COLORMAP_OCEAN),1,0)
            
        y_true = get_image(labels[i])
        y_true = y_true.astype('int64')  
    
        if vis and i%numvis==0:
            y_true_vis = y_true.astype(np.uint8)
            y_true_vis *= 255//2
            overlay_true = cv2.addWeighted(x_vis,1,cv2.applyColorMap(y_true_vis,cv2.COLORMAP_OCEAN),1,0)
            tosave = Image.fromarray(np.vstack((overlay_true, overlay_pred)))
            tosave.save("results/segmentation_linknet_resnet18/vis/{}_{}_{}".format(mode, w.split('/')[-1][:-5], images[i].split('/')[-1]))
        
        mAccuracy_ += mAccuracy(y_pred, y_true)/dlina
        mPrecision_ += mPrecision(y_pred, y_true)/dlina
        mRecall_ += mRecall(y_pred, y_true)/dlina
        mIU_ += mIU(y_pred, y_true)/dlina
        mF1_ += mF1(y_pred, y_true)/dlina
        
    print("accuracy: {}".format(mAccuracy_))
    print("precision: {}".format(mPrecision_))
    print("recall: {}".format(mRecall_))
    print("iu: {}".format(mIU_))
    print("f1: {}".format(mF1_))
    
    with open('results/segmentation_linknet_resnet18/{}_{}.pkl'.format(mode, wc), 'wb') as f:
        pickle.dump([mode, wc, mAccuracy_, mPrecision_, mRecall_, mIU_, mF1_], f)
    
    wc += 1