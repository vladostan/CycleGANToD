# coding: utf-8

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# In[]: Imports
from collections import Counter
import pickle
import keras
import matplotlib.pylab as plt
from tqdm import tqdm
import numpy as np
import datetime
import sys
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
import classification_models.keras_applications as ka
from keras import optimizers, callbacks
import imgaug.augmenters as iaa
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

# In[]: Parameters
log = True
verbose = 2
aug_mode = 0

resume_from = False
#weights_path = "weights/segmentation_bdd/.hdf5"

batch_size_init = 8
batch_factor = [1,2,2,3,4]
batch_size = batch_size_init//batch_factor[aug_mode]
test_size = 0.2
random_state = 28
classes = ['city street', 'highway', 'residential', 'parking lot', 'tunnel']
num_classes = len(classes)
#input_shape = (720, 1280, 3)
input_shape = (331, 331, 3)

class_weight_counting = False

# In[]: Logger
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/classification_bdd_val_day/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

if log:
    sys.stdout = Logger()

print('Date and time: {}\n'.format(loggername))

# READ IMAGES AND MASKS
# In[2]:
with open('bdd_day_night_scenes.pkl', 'rb') as f:
    day_scenes = pickle.load(f)
    
day_scenes = {key:val for key, val in day_scenes.items() if val in classes}
        
cyclegan_images = ['results/day2night_bdd_cyclegan/test_latest/images/' + di.split('/')[-2] + '_' + di.split('/')[-1].split('.')[0] + '_fake_B.jpg' for di in day_scenes.keys()]
    
print("Total images count: {}".format(len(day_scenes)))

# In[]: Read images and labels from file
#def get_image(path):
#    img = Image.open(path)
#    img = img.resize((input_shape[1], input_shape[0]))
#    img = np.array(img) 
#    return img    

def get_image(path):
    img = Image.open(path)
    img = img.resize((588, 331))
    img = np.array(img) 
    return img[:,128:459]

print("Images shape: {}".format(get_image(next(iter(day_scenes.keys()))).shape))
print("CycleGAN images shape: {}\n".format(get_image(cyclegan_images[0]).shape))

print("Images dtype: {}".format(get_image(next(iter(day_scenes.keys()))).dtype))
print("CycleGAN images dtype: {}\n".format(get_image(cyclegan_images[0]).dtype))

# In[]: Prepare for training
print("Train:test split = {}:{}\n".format(1-test_size, test_size))

images_train, images_val, labels_train, labels_val = train_test_split(list(day_scenes.keys()), list(day_scenes.values()), test_size=test_size, random_state=random_state)
cyclegan_images_train, cyclegan_images_val = train_test_split(cyclegan_images, test_size=test_size, random_state=random_state)

print("Training images count: {}".format(len(images_train)))
print("Training labels count: {}\n".format(len(labels_train)))
print("Validation images count: {}".format(len(images_val)))
print("Validation labels count: {}\n".format(len(labels_val)))

# In[]:
if class_weight_counting:    
    cw = Counter(labels_train)
    cw = np.array(list(cw.values()))
    
    if sum(cw) == len(labels_train):
        print("Class weights calculated successfully:")
        class_weights = np.median(cw/sum(cw))/(cw/sum(cw))
        for cntr,i in enumerate(class_weights):
            print("Class {} = {}".format(cntr, i))
    else:
        print("Class weights calculation failed")
else:
    class_weights = np.array([0.25812362, 0.5737214, 1., 22.27906977, 88.7037037])  

# In[]: AUGMENTATIONS    
def random_float(low, high):
    return np.random.random()*(high-low) + low
    
if aug_mode == 1:
    print("DOING CLASSICAL AUGMENTATION")  
elif aug_mode == 2:
    print("DOING CYCLEGAN AUGMENTATION")
elif aug_mode == 3:
    print("DOING MIXED AUGMENTATION")
elif aug_mode == 4:
    print("DOING COMBO AUGMENTATION")
else:
    print("NO AUGMENTATIONS")

if aug_mode == 1 or aug_mode == 3 or aug_mode == 4:
            
    def augment(image):
        
        mul = random_float(0.1, 0.5)
        add = np.random.randint(-100,-50)
        gamma = random_float(2,3)
        
        aug = iaa.OneOf([
                iaa.Multiply(mul = mul),
                iaa.Add(value = add),
                iaa.GammaContrast(gamma=gamma)
                ])
    
        image_augmented = aug.augment_image(image)
        
        return image_augmented

if aug_mode == 4:
    
    def augment_hard(image):
        
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
    
        augmented = aug(image=image)
        image_augmented = augmented['image']
        
        return image_augmented

# In[]: CUSTOM GENERATORS
def train_generator(images_path, labels_path, cyclegan_images = None, aug_mode = 0, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_factor[aug_mode]*batch_size, input_shape[0], input_shape[1], 3))
        y_batch = np.zeros((batch_factor[aug_mode]*batch_size, num_classes), dtype=np.int64)
        
        for b in range(batch_size):
            
            if i == len(labels_path):
                i = 0
                
            x = get_image(images_path[i])
            
            if labels_path[i] == classes[0]:
                y = 0
            elif labels_path[i] == classes[1]:
                y = 1
            elif labels_path[i] == classes[2]:
                y = 2
            elif labels_path[i] == classes[3]:
                y = 3
            elif labels_path[i] == classes[4]:
                y = 4
                
            y = to_categorical(y, num_classes=num_classes)
            
            x_batch[batch_factor[aug_mode]*b] = x
            y_batch[batch_factor[aug_mode]*b] = y

            if aug_mode == 1:
                x2 = augment(x)
                x_batch[batch_factor[aug_mode]*b+1] = x2
                y_batch[batch_factor[aug_mode]*b+1] = y
            elif aug_mode == 2:
                x2 = get_image(cyclegan_images[i])
                x_batch[batch_factor[aug_mode]*b+1] = x2
                y_batch[batch_factor[aug_mode]*b+1] = y 
            elif aug_mode == 3:
                x2 = augment(x)
                x3 = get_image(cyclegan_images[i])
                x_batch[batch_factor[aug_mode]*b+1] = x2
                x_batch[batch_factor[aug_mode]*b+2] = x3
                y_batch[batch_factor[aug_mode]*b+1] = y
                y_batch[batch_factor[aug_mode]*b+2] = y
            elif aug_mode == 4:
                x2 = augment(x)
                x3 = augment_hard(x)
                x4 = get_image(cyclegan_images[i])
                x_batch[batch_factor[aug_mode]*b+1] = x2
                x_batch[batch_factor[aug_mode]*b+2] = x3
                x_batch[batch_factor[aug_mode]*b+3] = x4
                y_batch[batch_factor[aug_mode]*b+1] = y
                y_batch[batch_factor[aug_mode]*b+2] = y
                y_batch[batch_factor[aug_mode]*b+3] = y
                
            i += 1
            
        x_batch = ka.nasnet.preprocess_input(x_batch)
    
        yield (x_batch, y_batch)

def val_generator(images_path, labels_path, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
        y_batch = np.zeros((batch_size, num_classes), dtype=np.int64)

        for b in range(batch_size):
            
            if i == len(labels_path):
                i = 0
                
            x = get_image(images_path[i])

            if labels_path[i] == classes[0]:
                y = 0
            elif labels_path[i] == classes[1]:
                y = 1
            elif labels_path[i] == classes[2]:
                y = 2
            elif labels_path[i] == classes[3]:
                y = 3
            elif labels_path[i] == classes[4]:
                y = 4
                
            y = to_categorical(y, num_classes=num_classes)
            
            x_batch[b] = x
            y_batch[b] = y
                
            i += 1
            
        x_batch = ka.nasnet.preprocess_input(x_batch)
    
        yield (x_batch, y_batch)
           
# In[ ]:
train_gen = train_generator(images_path = images_train, 
                             labels_path = labels_train,
                             cyclegan_images = cyclegan_images_train,
                             aug_mode = aug_mode,
                             batch_size = batch_size)

val_gen = val_generator(images_path = images_val, 
                         labels_path = labels_val,
                         batch_size = batch_size_init)

# In[ ]:
# # Define model
base_model = ka.nasnet.NASNetLarge(input_shape=input_shape, weights='imagenet', classes=num_classes, include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

if resume_from:
	print("Loading weights: {}".format(weights_path))
	model.load_weights(weights_path)

print("Model summary:")
model.summary()

# In[ ]:
learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = ['categorical_crossentropy']
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[]:
import tensorflow as tf
from keras import backend as K

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#K.set_session(get_tf_session())

# In[ ]:
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 4, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor='val_loss', patience = 8, verbose = 1)
clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/classification_bdd_val_day/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/classification_bdd_val_day/{}.hdf5'.format(loggername), monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = True)
    tensor_board = callbacks.TensorBoard(log_dir='./tblogs/classification_bdd_val_day')
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)
    clbacks.append(tensor_board)

print("Callbacks: {}\n".format(clbacks))

# In[ ]:
steps_per_epoch = len(labels_train)//batch_size
validation_steps = len(labels_val)//batch_size
epochs = 1000

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = clbacks,
    validation_data = val_gen,
    validation_steps = validation_steps,
    class_weight = class_weights
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))