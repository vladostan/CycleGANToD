# -*- coding: utf-8 -*-

import json
import pickle
from tqdm import tqdm

# In[]:
train_file = '../datasets/bdd100k_labels_release/labels/bdd100k_labels_images_train.json'
val_file = '../datasets/bdd100k_labels_release/labels/bdd100k_labels_images_val.json'

# In[]:
with open('bdd_day_night_list.pkl', 'rb') as f:
    day_images = pickle.load(f)
    night_images = pickle.load(f)
    
# In[]:

with open(train_file) as json_file:
    train_data = json.load(json_file)

with open(val_file) as json_file:
    val_data = json.load(json_file)
    
# In[]:
day_scenes = {}
night_scenes = {}

for di in tqdm(day_images):
    nobreak = True
    for td in train_data:
        name = td['name']
        if 'train/' + name in di:
            scene = td['attributes']['scene']
            day_scenes[di] = scene
            nobreak = False
            break
    for vd in val_data:
        name = vd['name']
        if 'val/' + name in di:
            scene = vd['attributes']['scene']
            day_scenes[di] = scene
            nobreak = False
            break
    if nobreak:
        print(di)
        
#../datasets/bdd100k/images/train/60595c14-113d4b8c.jpg
        
#for ni in tqdm(night_images):
#    for td in train_data:
#        name = td['name']
#        if 'train/' + name in ni:
#            scene = td['attributes']['scene']
#            night_scenes[ni] = scene
#            break
#    for vd in val_data:
#        name = vd['name']
#        if 'val/' + name in ni:
#            scene = vd['attributes']['scene']
#            night_scenes[ni] = scene
#            break
            
#with open('bdd_day_night_scenes.pkl', 'wb') as f:
#    pickle.dump(day_scenes, f)
#    pickle.dump(night_scenes, f)

# In[]:
#with open('bdd_day_night_scenes.pkl', 'rb') as f:
#    day_scenes = pickle.load(f)
#    night_scenes = pickle.load(f)

# In[]:




# In[]:




# In[]:




# In[]:




