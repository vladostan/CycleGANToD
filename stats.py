# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pickle
import os
from glob import glob

# In[]: 
path = "results/segmentation_linknet_resnet18/"
files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.pkl'))]

files.sort()

# In[]:
with open(files[2], 'rb') as f:
    mode, wc, mAccuracy_, mPrecision_, mRecall_, mIU_, mF1_ = pickle.load(f)
    
# In[]:
modes = []
wcs = []
mAccuracies = []
mPrecisions = []
mRecalls = []
mIUs = []
mF1s = []

for file in files:
    with open(file, 'rb') as f:
        mode, wc, mAccuracy, mPrecision, mRecall, mIU, mF1 = pickle.load(f)
    
    modes.append(mode)
    wcs.append(wc)
    mAccuracies.append(mAccuracy)
    mPrecisions.append(mPrecision)
    mRecalls.append(mRecall)
    mIUs.append(mIU)
    mF1s.append(mF1)
    
# In[]:
o = mF1s
print((o[0] + o[1] + o[7])/3)
print((o[8] + o[9] + o[10])/3)
print((o[11] + o[12] + o[13])/3)
print((o[14] + o[2] + o[3])/3)
print((o[4] + o[5] + o[6])/3)
    
# In[]:
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

x = ['Accuracy', 'Precision', 'Recall', 'IU', 'F1']

for i in range(5):
    y = [mAccuracies[i], mPrecisions[i], mRecalls[i], mIUs[i], mF1s[i]]
    plt.plot(x,y)
    plt.legend('01234')
    
for i in range(5,10):
    y = [mAccuracies[i], mPrecisions[i], mRecalls[i], mIUs[i], mF1s[i]]
    plt.plot(x,y)
    plt.legend('01234')

# In[]:
    
    
    
    

# In[]:
    
    
    
    

