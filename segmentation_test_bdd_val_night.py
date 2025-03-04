# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
import numpy as np
import cv2
import PIL.Image as Image
import pickle
from keras import optimizers
from sklearn.model_selection import train_test_split
from losses import dice_coef_multiclass_loss, dice_coef_multiclass
from metrics_for_paper import mAccuracy, mPrecision, mRecall, mIU, mF1
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Unet
from tqdm import tqdm

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# READ IMAGES AND MASKS
# In[2]:
with open('bdd_day_night_list.pkl', 'rb') as f:
    day_images = pickle.load(f)
    night_images = pickle.load(f)
        
day_images.sort()
night_images.sort()
    
day_labels = [di.replace("images","drivable_maps")[:-4] + "_drivable_id.png" for di in day_images]
night_labels = [ni.replace("images","drivable_maps")[:-4] + "_drivable_id.png" for ni in night_images]
    
print("Total day images count: {}".format(len(day_images)))
print("Total day labels count: {}\n".format(len(day_labels)))
print("Total night images count: {}".format(len(night_images)))
print("Total night labels count: {}\n".format(len(night_labels)))

# In[]: Prepare for testing
test_size = 0.8
random_state = 1

night_images_val, night_images_test, night_labels_val, night_labels_test = train_test_split(night_images, night_labels, test_size=test_size, random_state=random_state)

print("Training images count: {}".format(len(day_images)))
print("Training labels count: {}\n".format(len(day_labels)))
print("Validation images count: {}".format(len(night_images_val)))
print("Validation labels count: {}\n".format(len(night_labels_val)))
print("Testing images count: {}".format(len(night_images_test)))
print("Testing labels count: {}\n".format(len(night_labels_test)))

# In[]
input_shape = (384, 640, 3)
num_classes = 3
vis = True
numvis = 5000
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
    images = night_images_test
    labels = night_labels_test

# In[ ]:
backbone = 'seresnext101'
preprocessing_fn = get_preprocessing(backbone)
model = Unet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')

weights = []

################### PSPNET SERESNEXT101 DICE LOSS from segmentation_models AUGMODE0
weights_path = "weights/segmentation_bdd_val_night/2019-07-20 21-52-43.hdf5"
weights.append(weights_path)

################### PSPNET SERESNEXT101 DICE LOSS from segmentation_models AUGMODE1
weights_path = "weights/segmentation_bdd_val_night/2019-07-20 21-52-59.hdf5"
weights.append(weights_path)

################### PSPNET SERESNEXT101 DICE LOSS from segmentation_models AUGMODE2
weights_path = "weights/segmentation_bdd_val_night/2019-07-22 19-43-34.hdf5"
weights.append(weights_path)

################### PSPNET SERESNEXT101 DICE LOSS from segmentation_models AUGMODE3
weights_path = "weights/segmentation_bdd_val_night/2019-07-28 22-13-27.hdf5"
weights.append(weights_path)

################### PSPNET SERESNEXT101 DICE LOSS from segmentation_models AUGMODE4
weights_path = "weights/segmentation_bdd_val_night/2019-08-01 06-51-04.hdf5"
weights.append(weights_path)

# In[ ]:
learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = [dice_coef_multiclass_loss]
metrics = [dice_coef_multiclass]

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

# In[]:
dlina = len(images)

for w in tqdm(weights):
    
    model.load_weights(w)
    model.compile(optimizer = optimizer, loss = losses, metrics = metrics)
    model._make_predict_function()

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
            tosave.save("results/segmentation_bdd_val_night/vis/{}_{}_{}".format(mode, w.split('/')[-1][:-5], images[i].split('/')[-1]))
        
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
    
    with open('results/segmentation_bdd_val_night/{}_{}.pkl'.format(mode, w.split('/')[-1][:-5]), 'wb') as f:
        pickle.dump([mode, mAccuracy_, mPrecision_, mRecall_, mIU_, mF1_], f)