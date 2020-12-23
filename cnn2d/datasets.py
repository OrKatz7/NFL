import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return torch.tensor(x.transpose(0, 3, 1, 2).astype('float32'))

class DatasetRetriever(Dataset):
    """
    3D Dataset Retriever for NFL 2020 
    """
    def __init__(self, 
                 df = None,
                 seq_len = 50,
                 crop_size = 128,
                 image_size = 150,
                 transforms = None,
                 image_root='train_images',
                 test=False,
                 add_channel=True):
        super().__init__()

        self.df = df
        self.len_df = self.df.seq_index_1.nunique()
        self.image_root = image_root
        self.test = test
        self.seq_len = seq_len
        self.add_channel = add_channel
        self.image_size = image_size
        self.crop_size = crop_size
        self.transforms = transforms

    def __getitem__(self, index: int):
        
        paths = self.df[self.df.seq_index_1==index]
        if random.choice([True, False]):
            sequence = to_tensor(np.stack([self.load_image(paths.iloc[i]) for i in range(len(paths))]))
        else:
            sequence = to_tensor(np.stack([np.flip(self.load_image(paths.iloc[i]),1) for i in range(len(paths))]))
        
        s = sequence.shape
        if s[0]<self.seq_len:
            sequence = torch.cat((sequence, torch.zeros(self.seq_len-s[0], s[1], s[2],s[3])),0)
        
        frames = torch.tensor(paths.frame.to_numpy().reshape(-1,1))
        frames = torch.cat([frames, torch.zeros(self.seq_len-s[0],1)],0)
        
        if not self.test:
            label = torch.tensor(paths.impact.to_numpy().reshape(-1,1))
            label = torch.cat([label, torch.zeros(self.seq_len-s[0],1)],0)
            
            return sequence, label, frames
        return sequence, frames
        
    def __len__(self) -> int:
        return self.len_df

    def load_image(self, row):
            img = cv2.imread(self.get_path(row))
            masked_image = np.zeros((img.shape[0],img.shape[1],4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            s = img.shape
            h_center = (2*row.left+row.width)//2
            v_center = (2*row.top+row.height)//2
            h_center = max(self.image_size//2,h_center)
            v_center = max(self.image_size//2,v_center)
            h_center = min(s[1]-self.image_size//2,h_center)
            v_center = min(s[0]-self.image_size//2,v_center)
            
            # Random Cropping
            left = h_center - random.randint(self.crop_size//2,self.image_size//2)
            top = v_center - random.randint(self.crop_size//2,self.image_size//2)
            right = left + self.crop_size
            bottom = top + self.crop_size
            
            img = self.transforms(image=img)
            masked_image[:,:,:3] = img['image']
            
            # Add Noise to mask - To do
            masked_image[row.top:row.top+row.height,row.left:row.left+row.width,3]=1
    
            img = masked_image[top:bottom,left:right,:]
            assert img.shape[0]==self.crop_size
            assert img.shape[1]==self.crop_size
            assert img.shape[2]==4
               
            img /= 255.0
            
            if self.add_channel:
                return img
            
            return img[:,:,:3]
            
    def get_path(self, row):
        return row.image_path