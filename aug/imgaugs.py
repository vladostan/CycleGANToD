# -*- coding: utf-8 -*-

import random
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import webcolors
from PIL import Image
import imgaug

from pylab import rcParams
rcParams['figure.figsize'] = 15, 30

# In[]
name = '2018-11-30-158'
path = '../datasets/day2night_inno/trainA/' + name + '.png'

img = Image.open(path)
image = np.array(img)
plt.imshow(image)


# In[]
def vis(aug, image=image):
    augmented = aug(image=image)

    image_augmented = augmented['image']

    f, ax = plt.subplots(2, 1, figsize=(15, 15))

    ax[0].imshow(image)
    ax[1].imshow(image_augmented)
    
# In[]
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

# In[]
aug = Blur(blur_limit=5, p=1.)
vis(aug)

# In[]
from imgaug import augmenters as iaa

seq = iaa.Sequential([iaa.Snowflakes()])


# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

