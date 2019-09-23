# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

# In[]:
img = cv2.imread("/home/kenny/Desktop/00067cfb-5443fe39.jpg")
plt.imshow(img) 

# In[]:
def random_float(low, high):
    return np.random.random()*(high-low) + low

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

# In[]:
img2 = augment(img)
plt.imshow(img2)
cv2.imwrite("a.jpg", img2)

# In[]:




# In[]:




# In[]:




