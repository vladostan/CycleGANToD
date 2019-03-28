# -*- coding: utf-8 -*-

import keras.backend as K
from metrics import tptnfpfn
from keras.utils import to_categorical

def dice_coef_binary(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_binary_loss(y_true, y_pred):
    return 1-dice_coef_binary(y_true, y_pred)

def dice_coef_multiclass(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 3 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_multiclass_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_multiclass(y_true, y_pred)

def mIU_fp_penalty(y_true, y_pred, smooth=1e-7):
    
    y_true_f = K.flatten(y_true[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    
    tp = K.sum(y_true_f * y_pred_f, axis=-1)
    fp = K.sum((1.-y_true_f)*y_pred_f, axis=-1)
    fn = K.sum(y_true_f*(1.-y_pred_f), axis=-1)
    
    return K.mean((tp / (tp + 2.*fp + fn + smooth)))

def mIU_fp_penalty_loss(y_true, y_pred):
    
    return 1 - mIU_fp_penalty(y_true, y_pred)
    

def iou_loss_score(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    print(intersection)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    print(union)
    iou = (intersection + smooth) / ( union + smooth)
    return iou
