# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# READ IMAGES AND MASKS
# In[2]:
PATH = os.path.abspath('datasets')

imgs_dir = "images"

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
from PIL import Image

def get_image(path):
    
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    
    return img

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

num_classes = 3
#
#def custom_generator(images_path, labels_path, batch_size = 1):
#    
#    i = 0
#    
#    while True:
#        
#        x_batch = np.zeros((batch_size, 256, 640, 3))
#        
#        for b in range(batch_size):
#            
#            if i == len(images_path):
#                i = 0
#                
#            x = get_image(images_path[i])            
#            x_batch[b] = x
#                
#            i += 1
#            
#        x_batch = preprocessing_fn(x_batch)
#
#        yield x_batch
           
# In[ ]:
from segmentation_models.backbones import get_preprocessing

#batch_size = 1

backbone = 'resnet18'

preprocessing_fn = get_preprocessing(backbone)

#test_gen = custom_generator(images_path = images, 
#                             labels_path = labels,
#                             batch_size = batch_size)

# In[ ]:
# # Define model
from segmentation_models import Linknet

model = Linknet(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')

weights_path = "weights/segmentation/2019-03-03 22-00-24.hdf5" # for 447 day images of innopolis in 2018 bare training 

model.load_weights(weights_path)

print("Model summary:")
model.summary()
model._make_predict_function()

# In[ ]:
from keras import optimizers

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = ['categorical_crossentropy']
metrics = ['categorical_accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

#model.compile(optimizer = Adam(lr=learning_rate, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
#              loss = losses, metrics = metrics)
#model.compile(optimizer = Adam(lr=learning_rate, epsilon=1e-8, decay=1e-6), loss = losses, metrics = metrics)
model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[ ]:
#steps = len(images)//batch_size
#verbose = 1
#
#print("Steps: {}".format(steps))
#
#print("Starting testing...\n")
#y_pred = model.predict_generator(
#    generator = test_gen,
#    steps = steps,
#    verbose = verbose
#)
#
#print("Finished testing\n")

# In[]:
#y_pred = to_categorical(np.argmax(y_pred, axis=-1))
#y_pred = y_pred.astype(np.int64)
#
#y_true = []
#
#for l in labels:
#    y_true.append(get_image(l))
#    
#y_true = to_categorical(y_true, num_classes=num_classes)
#y_true = y_true.astype('int64')

# In[]:
from metrics import tptnfpfn, mean_IU, frequency_weighted_IU, mean_accuracy, pixel_accuracy
from tqdm import tqdm

meaniu = 0
freqweightediu = 0
meanacc = 0
pixacc = 0
mIU_penalized = 0
cce = 0

dlina = len(images)

for i in tqdm(range(dlina)):
    
    x = get_image(images[i])
    x = np.expand_dims(x, axis = 0)
    x = preprocessing_fn(x)
    
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis = -1)
    y_pred = np.squeeze(y_pred)
#    y_pred = to_categorical(y_pred, num_classes=num_classes)
    
    y_true = get_image(labels[i])
    y_true = y_true.astype('int64')
    
    meaniu += mean_IU(y_pred, y_true)/dlina
    freqweightediu += frequency_weighted_IU(y_pred, y_true)/dlina
    meanacc += mean_accuracy(y_pred, y_true)/dlina
    pixacc += pixel_accuracy(y_pred, y_true)/dlina
    
    mIU_solo = 0
    
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=num_classes)[...,cl]
        true_labels = to_categorical(y_true, num_classes=num_classes)[...,cl]
        mIU_solo += tptnfpfn(pred_labels, true_labels)/2
        
    mIU_penalized += mIU_solo/dlina
    
    y_true = to_categorical(y_true, num_classes=num_classes)
    cce += model.evaluate(x=x, y=np.expand_dims(y_true,axis=0))[-1]/dlina
    
print("mean_IU: {}".format(meaniu))
print("frequency weighted IU: {}".format(freqweightediu))
print("mean accuracy: {}".format(meanacc))
print("pixel accuracy: {}".format(pixacc))
print("mIU_penalized_fp_no_background: {}".format(mIU_penalized))
print("CCE: {}".format(cce))

# In[]:
#i = 13
#x = get_image(images[i])
#y = get_image(labels[i])
#fig, axes = plt.subplots(nrows = 3, ncols = 1)
#axes[0].imshow(x)
#axes[1].imshow(np.argmax(y_true[i], axis = -1))
#axes[2].imshow(np.argmax(y_pred[i], axis = -1))
#fig.tight_layout()
