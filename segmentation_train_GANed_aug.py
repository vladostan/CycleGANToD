# coding: utf-8

# In[1]:
import os
from glob import glob
import numpy as np
import datetime

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[]
log = True
verbose = 2

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

# Print stdout to file
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/segmentation/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if log:
    sys.stdout = Logger()

#sys.stdout = open('logs/{}'.format(loggername), 'w')

print('Date and time: {}\n'.format(loggername))

# READ IMAGES AND MASKS
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

def get_label(path):
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    return img

# In[]:
class_weights = np.array([0.16626288, 1.,         1.46289384]) # for 447 day images of innopolis in 2018 

# In[]: AUGMENTATIONS
doaug = True

if doaug:
    print("DOING AUGMENTATION")
else:
    print("NO AUGMENTATIONS")

if doaug:
    from albumentations import (
        OneOf,
        Blur,
        RandomGamma,
        HueSaturationValue,
        RGBShift,
        RandomBrightness,
        RandomContrast,
        MedianBlur,
        CLAHE
    )
    
    aug = OneOf([
            Blur(blur_limit=5, p=1.),
            RandomGamma(gamma_limit=(50, 150), p=1.),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
            RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
            RandomBrightness(limit=.25, p=1.),
            RandomContrast(limit=.25, p=1.),
            MedianBlur(blur_limit=5, p=1.),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
            ], p=1.)
    
    def augment(image, aug=aug):
    
        augmented = aug(image=image)
        image_augmented = augmented['image']
        
        return image_augmented
    
# In[]: AUGMENTATIONS
print("DOING GAN")

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

num_classes = 3

def custom_generator(images_path, labels_path, preprocessing_fn = None, doaug = False, batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        if validation or not doaug:
	        x_batch = np.zeros((batch_size, 256, 640, 3))
	        y_batch = np.zeros((batch_size, 256, 640))
        else:
            x_batch = np.zeros((4*batch_size, 256, 640, 3))
            y_batch = np.zeros((4*batch_size, 256, 640))
        
        for b in range(batch_size):
            
            if i == len(labels_path):
                i = 0
                
            x = get_image(images_path[2*i+1])
            y = get_label(labels_path[i])
            
            if validation or not doaug:
                x_batch[b] = x
                y_batch[b] = y
            else:
                x2 = get_image(images_path[2*i])
                x3 = augment(x)
                x4 = augment(x2)
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                x_batch[2*b+2] = x3
                x_batch[2*b+3] = x4
                y_batch[2*b] = y
                y_batch[2*b+1] = y
                y_batch[2*b+2] = y
                y_batch[2*b+3] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
        
#        print(x_batch.shape)
#        print(y_batch.shape)
    
        yield (x_batch, y_batch)
           
# In[ ]:
from segmentation_models.backbones import get_preprocessing

batch_size = 1

backbone = 'resnet18'

preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images, 
                             labels_path = labels, 
                             preprocessing_fn = preprocessing_fn, 
                             doaug = doaug,
                             batch_size = batch_size)

# In[ ]:
# # Define model
from segmentation_models import Linknet

model = Linknet(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers
from losses import dice_coef_multiclass_loss, mIU_fp_penalty_loss, focal_loss
from keras_contrib.losses import jaccard_distance

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

#losses = ['categorical_crossentropy']

losses = [focal_loss]
metrics = ['categorical_accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[ ]:
from keras import callbacks
from callbacks import TelegramCallback

config = {
    'token': '720029625:AAGG5aS46wOliEIs0HmUFgg8koN_ScI3AIY',   # paste your bot token
    'telegram_id': 218977821,                                   # paste your telegram_id
}
tg_callback = TelegramCallback(config)

tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 4, verbose = 1, min_lr = 1e-7)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)
clbacks = [reduce_lr, early_stopper, tensor_board, tg_callback]

if log:
    csv_logger = callbacks.CSVLogger('logs/segmentation/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/segmentation/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)

print("Callbacks: {}\n".format(clbacks))

# In[ ]:
steps_per_epoch = len(images)//batch_size
epochs = 1000

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = clbacks,
    class_weight = class_weights
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))
