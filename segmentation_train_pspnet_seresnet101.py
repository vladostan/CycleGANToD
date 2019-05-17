# coding: utf-8

# In[1]:
import os
from glob import glob
import numpy as np
import datetime

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]
log = True
aug_mode = 0
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
        self.log = open('logs/segmentation_pspnet_seresnet101/{}.txt'.format(loggername), 'w')

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
    return img[16:,8:632]

def get_label(path):
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    return img[16:,8:632]

# In[]:
class_weights = np.array([0.16626288, 1.,         1.46289384]) # for 447 day images of innopolis in 2018 

# In[]: AUGMENTATIONS
if aug_mode == 1:
    print("DOING CLASSICAL AUGMENTATION")  
elif aug_mode == 2:
    print("DOING CYCLEGAN AUGMENTATION")
elif aug_mode == 3:
    print("DOING MIXED AUGMENTATION")
else:
    print("NO AUGMENTATIONS")

if aug_mode == 1 or aug_mode == 3:
    import imgaug.augmenters as iaa
    
    def random_float(low, high):
        return np.random.random()*(high-low) + low
    
    mul = random_float(0.1, 0.5)
    add = np.random.randint(-100,-50)
    gamma = random_float(2,3)
    
    aug = iaa.OneOf([
            iaa.Multiply(mul = mul),
            iaa.Add(value = add),
            iaa.GammaContrast(gamma=gamma)
            ])
            
    def augment(image, aug=aug):
    
        image_augmented = aug.augment_image(image)
        
        return image_augmented

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

num_classes = 3
input_shape = (240, 624, 3)

batch_factor = [1,2,2,3]

def custom_generator(images_path, labels_path, preprocessing_fn = None, aug_mode = aug_mode, batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        if validation or aug_mode == 0:
	        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
	        y_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        else:
            x_batch = np.zeros((batch_factor[aug_mode]*batch_size, input_shape[0], input_shape[1], 3))
            y_batch = np.zeros((batch_factor[aug_mode]*batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(labels_path):
                i = 0
                
            x = get_image(images_path[2*i+1])
            y = get_label(labels_path[i])
            
            if validation or aug_mode == 0:
                x_batch[b] = x
                y_batch[b] = y
            elif aug_mode == 1:
                x2 = augment(x)
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                y_batch[2*b] = y
                y_batch[2*b+1] = y
            elif aug_mode == 2:
                x2 = get_image(images_path[2*i])
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                y_batch[2*b] = y
                y_batch[2*b+1] = y 
            else:
                x2 = augment(x)
                x3 = get_image(images_path[2*i])
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                x_batch[2*b+2] = x3
                y_batch[2*b] = y
                y_batch[2*b+1] = y
                y_batch[2*b+2] = y
                
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

backbone = 'seresnet101'

preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images, 
                             labels_path = labels, 
                             preprocessing_fn = preprocessing_fn, 
                             aug_mode = aug_mode,
                             batch_size = batch_size)

# In[ ]:
# # Define model
from segmentation_models import PSPNet

model = PSPNet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax',  encoder_weights='imagenet')

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers
from segmentation_models.losses import dice_loss, cce_dice_loss
from segmentation_models.metrics import dice_score

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = [dice_loss]
metrics = [dice_score]

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

reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)
clbacks = [reduce_lr, early_stopper, tg_callback]

if log:
    csv_logger = callbacks.CSVLogger('logs/segmentation_pspnet_seresnet101/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/segmentation_pspnet_seresnet101/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    tensor_board = callbacks.TensorBoard(log_dir='./tblogs/segmentation_pspnet_seresnet101')
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)
    clbacks.append(tensor_board)

print("Callbacks: {}\n".format(clbacks))

# In[ ]:
steps_per_epoch = len(labels)//batch_size
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