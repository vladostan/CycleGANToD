# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[]
log = True

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
print("DOING GAN")

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

num_classes = 3

def custom_generator(images_path, labels_path, preprocessing_fn = None, batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        if validation:
	        x_batch = np.zeros((batch_size, 256, 640, 3))
	        y_batch = np.zeros((batch_size, 256, 640))
        else:
            x_batch = np.zeros((2*batch_size, 256, 640, 3))
            y_batch = np.zeros((2*batch_size, 256, 640))
        
        for b in range(batch_size):
            
            if i == len(images_path):
                i = 0
                
            x = get_image(images_path[2*i+1])
            y = get_label(labels_path[i])
            
            if validation:
                x_batch[b] = x
                y_batch[b] = y
            else:
                x2 = get_image(images_path[2*i])
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                y_batch[2*b] = y
                y_batch[2*b+1] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
    
        yield (x_batch, y_batch)
           
# In[ ]:
from segmentation_models.backbones import get_preprocessing

batch_size = 1

backbone = 'resnet18'

preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images, 
                             labels_path = labels, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

# In[ ]:
# # Define model
from segmentation_models import Linknet

model = Linknet(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = ['categorical_crossentropy']
metrics = ['categorical_accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)


# In[ ]:
from keras import callbacks

model_checkpoint = callbacks.ModelCheckpoint('weights/segmentation/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
#tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 4, verbose = 1, min_lr = 1e-7)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)
clbacks = [model_checkpoint, reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/segmentation/{}.log'.format(loggername))
    clbacks.append(csv_logger)

print("Callbacks: {}\n".format(clbacks))


# In[ ]:
steps_per_epoch = len(images)//batch_size
epochs = 1000
verbose = 2

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