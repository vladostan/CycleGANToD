# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# In[]: DATA
# Trained on CCE
#train_1_cce = {"mean_IU": 0.9208359066301264,
#         "frequency weighted IU": 0.9615444003008902,
#         "mean accuracy": 0.9492156178764647,
#         "pixel accuracy": 0.9754851953561948,
#         "mIU_penalized_fp_no_background": 0.8022608511724546,
#         "dice": 0.9132714864299929
#        }
#
#test_1_cce = {"mean_IU": 0.6919448857492123,
#         "frequency weighted IU": 0.8301668379907033,
#         "mean accuracy": 0.8361818766926736,
#         "pixel accuracy": 0.8822732248613913,
#         "mIU_penalized_fp_no_background": 0.43828268527656244,
#         "dice": 0.6850184365826586
#        }
#
#train_2_cce = {"mean_IU": 0.9472695713751482,
#         "frequency weighted IU": 0.9737132627390848,
#         "mean accuracy": 0.9629875315872906,
#         "pixel accuracy": 0.9836384681513913,
#         "mIU_penalized_fp_no_background": 0.8486950857572041,
#         "dice": 0.9424347420466829
#        }
#
#test_2_cce = {"mean_IU": 0.7215472521404238,
#         "frequency weighted IU": 0.8549668519901354,
#         "mean accuracy": 0.8277798484150793,
#         "pixel accuracy": 0.9045121387768814,
#         "mIU_penalized_fp_no_background": 0.48850373625507637,
#         "dice": 0.7140497394426853
#        }
#
#train_3_cce = {"mean_IU": 0.944216600013085,
#         "frequency weighted IU": 0.9722787921052543,
#         "mean accuracy": 0.9637215305299385,
#         "pixel accuracy": 0.9818008730075495,
#         "mIU_penalized_fp_no_background": 0.8460513033730046,
#         "dice": 0.9361771105453208
#        }
#    
#test_3_cce = {"mean_IU": 0.7276355739082678,
#         "frequency weighted IU": 0.8544218148735585,
#         "mean accuracy": 0.8498295912042985,
#         "pixel accuracy": 0.9009976622878867,
#         "mIU_penalized_fp_no_background": 0.488734722590782,
#         "dice": 0.720352483773103
#        }
#    
#train_4_cce = {"mean_IU": 0.962715455698229,
#         "frequency weighted IU": 0.9815659775031409,
#         "mean accuracy": 0.9733406733810351,
#         "pixel accuracy": 0.9867523671263283,
#         "mIU_penalized_fp_no_background": 0.8804213560289029,
#         "dice": 0.9522818611888941
#        }
#
#test_4_cce = {"mean_IU": 0.7399346311936498,
#         "frequency weighted IU": 0.862772470181213,
#         "mean accuracy": 0.8491591291154035,
#         "pixel accuracy": 0.9071904336252522,
#         "mIU_penalized_fp_no_background": 0.5108234144345777,
#         "dice": 0.7312126281960389
#        }
#  
#train_cce = [train_1_cce, train_2_cce, train_3_cce, train_4_cce]
#test_cce = [test_1_cce, test_2_cce, test_3_cce, test_4_cce]

# In[]: 
# Trained on Dice
train_4_dice = {"mean_IU": 0.9788234388542933,
        "frequency weighted IU": 0.9877865361823307,
        "mean accuracy": 0.9869401700345356,
        "pixel accuracy": 0.9932978209766512,
        "mIU_penalized_fp_no_background": 0.9217704306745046,
        "dice": 0.9823468521537402
        }

test_4_dice = {"mean_IU": 0.7507758124000424,
        "frequency weighted IU": 0.8704001097474142,
        "mean accuracy": 0.8500628521900183,
        "pixel accuracy": 0.9171118623466904,
        "mIU_penalized_fp_no_background": 0.5287808864459332,
        "dice": 0.7559706630806127
        }

# In[]: 
# Trained on Jaccard
#train_4_jaccard = {"mean_IU": 0.9664029703708996,
#        "frequency weighted IU": 0.9827603780286285,
#        "mean accuracy": 0.976665559774132,
#        "pixel accuracy": 0.9881099419305786,
#        "mIU_penalized_fp_no_background": 0.8845113505842928,
#        "dice": 0.9587634254155283 
#        }
#
#test_4_jaccard = {"mean_IU": 0.7464148505614313,
#        "frequency weighted IU": 0.8660823350863838,
#        "mean accuracy": 0.848681401054398,
#        "pixel accuracy": 0.9100283058740757,
#        "mIU_penalized_fp_no_background": 0.5223429339429536,
#        "dice": 0.7378697856603937 
#        }
    
# In[]: 
# Trained on 
#train_4_ = {"mean_IU": ,
#        "frequency weighted IU": ,
#        "mean accuracy": ,
#        "pixel accuracy": ,
#        "mIU_penalized_fp_no_background": ,
#        "dice": 
#        }
#
#test_4_ = {"mean_IU": ,
#        "frequency weighted IU": ,
#        "mean accuracy": ,
#        "pixel accuracy": ,
#        "mIU_penalized_fp_no_background": ,
#        "dice": 
#        }

# In[]:
#from matplotlib.pyplot import figure
#figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
#
#for i in range(len(train_cce)):
#    plt.plot(train_cce[i].values())
#    
#x = train_cce[0].keys()
#xi = [i for i in range(0, len(x))]
#plt.xticks(xi, x)
#plt.legend(["bare", "CycleGANed", "albumentations", "mixed"])
#plt.title("Train")
#plt.savefig("results/train.png")

# In[]:
#figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
#
#for i in range(len(test_cce)):
#    plt.plot(test_cce[i].values())
#  
#plt.xticks(xi, x)
#plt.legend(["bare", "CycleGANed", "albumentations", "mixed"])
#plt.title("Test")
#plt.savefig("results/test.png")

# In[]: CCE vs DICE mixed augs TRAIN
#figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
#
#plt.plot(train_4_cce.values())
#plt.plot(train_4_jaccard.values())
#plt.plot(train_4_dice.values())
#  
#plt.xticks(xi, x)
#plt.legend(["CCE", "Jaccard", "Dice"])
#plt.title("Train: Mixed augmentation for different losses")
#plt.savefig("results/train.png")

# In[]: CCE vs DICE vs Jaccard mixed augs TEST
#figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
#
#plt.plot(test_4_cce.values())
#plt.plot(test_4_jaccard.values())
#plt.plot(test_4_dice.values())
#  
#plt.xticks(xi, x)
#plt.legend(["CCE", "Jaccard", "Dice"])
#plt.title("Test: Mixed augmentation for different losses")
#plt.savefig("results/test.png")