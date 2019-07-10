# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pickle
import os
from glob import glob

# In[]: 
path = "results/segmentation_bdd/"
files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.pkl'))]

files.sort()

# In[]:
with open(files[0], 'rb') as f:
    mode, mAccuracy_, mPrecision_, mRecall_, mIU_, mF1_ = pickle.load(f)
    
# In[]:
modes = []
mAccuracies = []
mPrecisions = []
mRecalls = []
mIUs = []
mF1s = []

for file in files:
    with open(file, 'rb') as f:
        mode, mAccuracy, mPrecision, mRecall, mIU, mF1 = pickle.load(f)
    
    modes.append(mode)
    mAccuracies.append(mAccuracy)
    mPrecisions.append(mPrecision)
    mRecalls.append(mRecall)
    mIUs.append(mIU)
    mF1s.append(mF1)
    
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
    
    
    
    

