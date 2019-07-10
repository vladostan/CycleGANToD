#!/usr/bin/env python
# coding: utf-8

# In[28]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import webcolors
from PIL import Image
import cv2
import tqdm

from pylab import rcParams
rcParams['figure.figsize'] = 15, 30


# In[21]:
PATH = os.path.abspath('../../datasets/bdd/bdd100k/images')
SOURCE_IMAGES = [os.path.join(PATH, "100k/train")]

# filename = 'af2d8f1f-f36ec56e'
# img = 

# In[22]:
images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.jpg")))
    
len(images)


# In[23]:


# img = cv2.imread('/home/kenny/Desktop/bdd/images/10k/test/bfbe2ad2-68fb1d68.jpg',0)

img_gray = cv2.imread(images[0], 0)
plt.imshow(img_gray, cmap='gray')


# In[24]:


# hist = cv2.calcHist([img], [0], mask = None, histSize = [256], ranges = [0,256])
# plt.plot(hist)


# In[25]:


avg_color = np.average(np.average(img_gray, axis=0), axis=0)
avg_color


# In[32]:


from tqdm import tqdm

d = 0
n = 0
num_images = 100

for im in tqdm(images):
    img_gray = cv2.imread(im, 0)
    avg_color = np.average(np.average(img_gray, axis=0), axis=0)
    pil_img = Image.open(im)
    if avg_color < 256/6 and n < num_images:
#         pil_img.save('data/night/tr{}.jpg'.format(n))
        n += 1
    elif avg_color > 256/2 and d < num_images:
#         pil_img.save('data/day/tr{}.jpg'.format(d))
        d += 1
        
    if n == num_images and d == num_images:
        break

