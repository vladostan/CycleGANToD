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
import imgaug.augmenters as iaa

def random_float(low, high):
    return np.random.random()*(high-low) + low

mul = random_float(0.1, 0.5)
add = np.random.randint(-100,-50)
gamma = random_float(2,3)

seq = iaa.Multiply(mul = mul)
seq = iaa.Add(value = add)
seq = iaa.GammaContrast(gamma=gamma)

seq = iaa.OneOf([
        iaa.Multiply(mul = mul),
        iaa.Add(value = add),
        iaa.GammaContrast(gamma=gamma)
        ])
    
image_aug = seq.augment_image(image)
plt.imshow(image_aug)

# In[]

# In[]

# In[]

# In[]

# In[]

# In[]

