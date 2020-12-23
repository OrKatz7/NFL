
import numpy as np
import pandas as pd
import glob
import cv2
import os
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
import functools
import torch



def get_training_augmentation(y=128,x=128):
    train_transform = [
                       albu.RandomBrightnessContrast(p=0.5),
                       albu.HorizontalFlip(p=0.5),
                       albu.VerticalFlip(p=0.5),
                       albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.3, border_mode = cv2.BORDER_REPLICATE),
                       albu.OneOf([
                           albu.CenterCrop(x,y),
                           albu.RandomCrop(x,y),    
                       ],p=1.0),
                       ]
    return albu.Compose(train_transform)


formatted_settings = {
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],}


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0
    return x

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation(y=224,x=224):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(x, y),albu.CenterCrop(x,y)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

class Dataset2D(Dataset):
    def __init__(self,df,jpeg_dir,transforms = albu.Compose([albu.HorizontalFlip()]),preprocessing=None,size=256,mode='val'):

    def __getitem__(self, idx):

        return im_id,frame,img,label #img CxHxW, label 1

    def __len__(self):
        return len(self.df)
