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

def tptnfpfn_keras(pred_labels, true_labels):
    
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
#    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    
    # true positive / (true positive + false positive + false negative)

#    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    
    if TP == 0 and FP == 0 and FN == 0:
        return 1.
    
    IU = TP/(TP+2*FP+FN)

def mIU_fp_penalty_loss(y_pred, y_true):
    
    mIU_solo = 0
    
    for cl in range(1,3):
        pred_labels = to_categorical(y_pred, num_classes=3)[...,cl]
        true_labels = to_categorical(y_true, num_classes=3)[...,cl]
        mIU_solo += tptnfpfn(pred_labels, true_labels)/2
        
    return mIU_solo

def iou_loss_score(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    print(intersection)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    print(union)
    iou = (intersection + smooth) / ( union + smooth)
    return iou
