# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import cv2
import PIL.Image as Image

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# READ IMAGES AND MASKS
# In[2]:
PATH = os.path.abspath('datasets')

losstype = 'mydice'
vis = False

imgs_dir = "images"
visdir = 'train/'

imgs_dir = "trainB"
visdir = 'test/'

SOURCE_IMAGES = [os.path.join(PATH, "day2night_inno/", imgs_dir)]

images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    
labels = []

for i in range(len(images)):
    labels.append(images[i].replace(imgs_dir, "labels"))

print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
labels.sort()

print(len(images))
print(len(labels))

# In[]
def get_image(path):
    
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    
    return img

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

num_classes = 3
           
# In[ ]:
from segmentation_models.backbones import get_preprocessing

backbone = 'resnet18'

preprocessing_fn = get_preprocessing(backbone)

# In[ ]:
# # Define model
from segmentation_models import Linknet

model = Linknet(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')

################### CCE
#weights_path = "weights/segmentation/CCE/2019-03-03 22-00-24.hdf5" # for 447 day images of innopolis in 2018 bare training 
#visdir+='bare/'

#weights_path = "weights/segmentation/CCE/2019-03-05 10-41-19.hdf5" # for 447 day images of innopolis in 2018 GANed training 
#visdir+='gan/'

#weights_path = "weights/segmentation/CCE/2019-03-26 08-51-43.hdf5" # for 447 day images of innopolis in 2018 albumentated training 
#visdir+='alb/'

#weights_path = "weights/segmentation/CCE/2019-03-26 10-04-18.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### DICE
#weights_path = "weights/segmentation/dice/2019-03-27 07-55-59.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### JACCARD
#weights_path = "weights/segmentation/jaccard/2019-03-27 13-59-56.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### MY IOU
#weights_path = "weights/segmentation/myiou/2019-03-28 16-19-29.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### FOCAL
#weights_path = "weights/segmentation/focal/2019-03-29 14-44-37.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### My dice: +FP OK
#weights_path = "weights/segmentation/2019-03-30 11-24-56.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### My dice: +0.1FP WORSE than prev
#weights_path = "weights/segmentation/2019-03-31 15-53-53.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### My dice: +100FP BAD
#weights_path = "weights/segmentation/2019-03-31 18-16-31.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### My dice: +0.5*(fp+fn)
#weights_path = "weights/segmentation/2019-04-01 07-57-37.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'

################### (DICE + IOU)/2:
#weights_path = "weights/segmentation/2019-04-01 11-46-54.hdf5" # for 447 day images of innopolis in 2018 albumentated + GANed training 
#visdir+='mix/'




###################
###################
###################




################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE0 1
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-22 09-10-59.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE0 2
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 12-36-48.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE0 3
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 14-33-14.hdf5"




################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE1 1
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 15-06-19.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE1 2
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 07-49-16.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE1 3
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 12-35-18.hdf5"




################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE2 1
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 15-06-07.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE2 2
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 13-57-52.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE2 3
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 13-58-59.hdf5"




################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE3 1
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-21 13-47-28.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE3 2
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 08-22-50.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE3 3
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-24 07-48-43.hdf5"




################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE4 1
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 08-23-52.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE4 2
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 09-59-55.hdf5"

################### LINKNET RESNET18 DICE LOSS from segmentation_models AUGMODE4 3
#weights_path = "weights/segmentation_linknet_resnet18/2019-05-23 10-00-46.hdf5"




model.load_weights(weights_path)

print("Model summary:")
model.summary()
model._make_predict_function()

# In[ ]:
from keras import optimizers
from metrics import tptnfpfn, mean_IU, frequency_weighted_IU, mean_accuracy, pixel_accuracy, mIU_fp_penalty
from losses import dice_coef_multiclass_loss, dice_coef_multiclass, mIU_fp_penalty_loss, focal_loss, dice_fp_penalty_loss, dice_fpfn_weighted_loss, dice_iou_loss
from keras_contrib.losses import jaccard_distance

from segmentation_models.losses import dice_loss, cce_dice_loss
from segmentation_models.metrics import dice_score

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = [dice_coef_multiclass_loss]
metrics = [dice_coef_multiclass]

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[]:
from tqdm import tqdm

meaniu = 0
freqweightediu = 0
meanacc = 0
pixacc = 0
mIU_penalized = 0
dice = 0

dlina = len(images)

for i in tqdm(range(dlina)):
    
    x = get_image(images[i])
    
    if vis:
        x_vis = x.copy()
        
    x = np.expand_dims(x, axis = 0)
    x = preprocessing_fn(x)
    
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis = -1)
    y_pred = np.squeeze(y_pred)

    if vis:
        y_pred_vis = y_pred.astype(np.uint8)
        y_pred_vis *= 255//2
        overlay_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(y_pred_vis,cv2.COLORMAP_OCEAN),1,0)
        
    y_true = get_image(labels[i])
    y_true = y_true.astype('int64')  

    if vis:
        y_true_vis = y_true.astype(np.uint8)
        y_true_vis *= 255//2
        overlay_true = cv2.addWeighted(x_vis,1,cv2.applyColorMap(y_true_vis,cv2.COLORMAP_OCEAN),1,0)
        tosave = Image.fromarray(np.vstack((overlay_true, overlay_pred)))
        tosave.save("results/{}/vis/{}{}".format(losstype, visdir, images[i].split('/')[-1]))
        
    meaniu += mean_IU(y_pred, y_true)/dlina
    freqweightediu += frequency_weighted_IU(y_pred, y_true)/dlina
    meanacc += mean_accuracy(y_pred, y_true)/dlina
    pixacc += pixel_accuracy(y_pred, y_true)/dlina
       
    mIU_penalized += mIU_fp_penalty(y_pred, y_true)/dlina
    
    y_true = to_categorical(y_true, num_classes=num_classes)
    
    dice += model.evaluate(x=x, y=np.expand_dims(y_true,axis=0))[-1]/dlina
    
print("mean_IU: {}".format(meaniu))
print("frequency weighted IU: {}".format(freqweightediu))
print("mean accuracy: {}".format(meanacc))
print("pixel accuracy: {}".format(pixacc))
print("mIU_penalized_fp_no_background: {}".format(mIU_penalized))
print("dice: {}".format(dice))

# In[]:
#i = 13
#x = get_image(images[i])
#y = get_image(labels[i])
#fig, axes = plt.subplots(nrows = 3, ncols = 1)
#axes[0].imshow(x)
#axes[1].imshow(np.argmax(y_true[i], axis = -1))
#axes[2].imshow(np.argmax(y_pred[i], axis = -1))
#fig.tight_layout()
