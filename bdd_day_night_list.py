#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:43:44 2019

@author: kenny
"""

import os
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

# In[]:
jpgpath = "../datasets/bdd100k/images/"

#pngpath = "../datasets/bdd100k/drivable_maps/labels/"

images = [y for x in os.walk(jpgpath) for y in glob(os.path.join(x[0], '*.jpg'))]

#labels = [y for x in os.walk(pngpath) for y in glob(os.path.join(x[0], '*.png'))]

# In[]:
day_images = []
night_images = []

for im in tqdm(images):
    img_gray = cv2.imread(im, 0)
    avg_color = np.average(np.average(img_gray, axis=0), axis=0)
    if avg_color < 256/6:
        night_images.append(im)
        print("NIGHT: {}".format(len(night_images)))
    elif avg_color > 256/3:
        day_images.append(im)
        print("DAY: {}".format(len(day_images)))

# In[]:
with open('bdd_day_night_list.pkl', 'wb') as f:
    pickle.dump(day_images, f)
    pickle.dump(night_images, f)

# In[]:
i = np.random.randint(0,len(day_images))
img = cv2.imread(day_images[i])
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# In[]:
i = np.random.randint(0,len(night_images))
img = cv2.imread(night_images[i])
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# In[]:
with open('bdd_day_night_list.pkl', 'rb') as f:
    day_images = pickle.load(f)
    night_images = pickle.load(f)

# In[]:
from shutil import copyfile

for dy in tqdm(day_images):
    dst = 'datasets/day2night_bdd/testA/' + dy.split('/')[-2] + '_' + dy.split('/')[-1]
    copyfile(dy, dst)


# In[]: READING FROM JSON
import json

jpgpath = "../datasets/bdd100k/images/"
images = [y for x in os.walk(jpgpath) for y in glob(os.path.join(x[0], '*.jpg'))]

# In[]:
jsonpath = "../datasets/bdd100k_labels_release/labels/"
jsons = [y for x in os.walk(jsonpath) for y in glob(os.path.join(x[0], '*.json'))]

# In[]:
with open(jsons[0], 'rb') as f:
    val_data = json.load(f)

# In[]:
imgname = val_data[0]['name']
timeofday = val_data[0]['attributes']['timeofday']
scenes = ['city street', 'highway', 'residential', 'parking lot', 'gas stations', 'tunnel']
for vd in val_data:
#    timeofday = vd['attributes']['timeofday']
    scene = vd['attributes']['scene']
#    if timeofday == 'night':
#        n+=1
#    elif timeofday == 'daytime':
#        d+=1


# In[]:



# In[]:



# In[]:



# In[]:



# In[]:



# In[]:



# In[]:



