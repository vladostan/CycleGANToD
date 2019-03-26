# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# In[]: DATA
# Trained on Keras CCE
train_1 = {"mean_IU": 0.9208359066301264,
         "frequency weighted IU": 0.9615444003008902,
         "mean accuracy": 0.9492156178764647,
         "pixel accuracy": 0.9754851953561948,
         "mIU_penalized_fp_no_background": 0.8022608511724546,
         "CCE": 0.9754851955162057
        }

test_1 = {"mean_IU": 0.6919448857492123,
         "frequency weighted IU": 0.8301668379907033,
         "mean accuracy": 0.8361818766926736,
         "pixel accuracy": 0.8822732248613913,
         "mIU_penalized_fp_no_background": 0.43828268527656244,
         "CCE": 0.8822732244127542
        }

train_2 = {"mean_IU": 0.9472695713751482,
         "frequency weighted IU": 0.9737132627390848,
         "mean accuracy": 0.9629875315872906,
         "pixel accuracy": 0.9836384681513913,
         "mIU_penalized_fp_no_background": 0.8486950857572041,
         "CCE": 0.9836384678046983
        }

test_2 = {"mean_IU": 0.7215472521404238,
         "frequency weighted IU": 0.8549668519901354,
         "mean accuracy": 0.8277798484150793,
         "pixel accuracy": 0.9045121387768814,
         "mIU_penalized_fp_no_background": 0.48850373625507637,
         "CCE": 0.9045121384564262
        }

train_3 = {"mean_IU": 0.944216600013085,
         "frequency weighted IU": 0.9722787921052543,
         "mean accuracy": 0.9637215305299385,
         "pixel accuracy": 0.9818008730075495,
         "mIU_penalized_fp_no_background": 0.8460513033730046,
         "CCE": 0.9818008732742374
        }
    
test_3 = {"mean_IU": 0.7276355739082678,
         "frequency weighted IU": 0.8544218148735585,
         "mean accuracy": 0.8498295912042985,
         "pixel accuracy": 0.9009976622878867,
         "mIU_penalized_fp_no_background": 0.488734722590782,
         "CCE": 0.900997661134248
        }
    
train_4 = {"mean_IU": 0.962715455698229,
         "frequency weighted IU": 0.9815659775031409,
         "mean accuracy": 0.9733406733810351,
         "pixel accuracy": 0.9867523671263283,
         "mIU_penalized_fp_no_background": 0.8804213560289029,
         "CCE": 0.9867523676597032
        }

test_4 = {"mean_IU": 0.7399346311936498,
         "frequency weighted IU": 0.862772470181213,
         "mean accuracy": 0.8491591291154035,
         "pixel accuracy": 0.9071904336252522,
         "mIU_penalized_fp_no_background": 0.5108234144345777,
         "CCE": 0.9071904347147994
        }
  
train = [train_1, train_2, train_3, train_4]
test = [test_1, test_2, test_3, test_4]
# In[]:
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

for i in range(len(train)):
    plt.plot(train[i].values())
    
x = train[0].keys()
xi = [i for i in range(0, len(x))]
plt.xticks(xi, x)
plt.legend(["bare", "CycleGANed", "albumentations", "mixed"])
plt.title("Train")
plt.savefig("results/train.png")

# In[]:
figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

for i in range(len(test)):
    plt.plot(test[i].values())
  
plt.xticks(xi, x)
plt.legend(["bare", "CycleGANed", "albumentations", "mixed"])
plt.title("Test")
plt.savefig("results/test.png")