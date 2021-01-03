import sys


import torch
import os
from datetime import datetime
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
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm
import shutil as sh

PATH_DF = '/kaggle/input/nfl-impact-detection/train_labels.csv'
ALL_DATA = -1 #change to 0 if you want to train only images with impact
video_dir = '/kaggle/input/nfl-impact-detection/train'
out_dir = 'train_images'
#!mkdir -p $out_dir

def mk_images(video_name, video_labels, video_dir, out_dir, only_with_impact=True):
    video_path=f"{video_dir}/{video_name}"
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    if only_with_impact:
        boxes_all = video_labels.query("video == @video_name")
        print(video_path, boxes_all[boxes_all.impact > 1.0].shape[0])
    else:
        print(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if frame ==0:
            print(img.shape)
        if not it_worked:
            break
        frame += 1
        if only_with_impact:
            boxes = video_labels.query("video == @video_name and frame == @frame")
            boxes_with_impact = boxes[boxes.impact > 1.0]
            if boxes_with_impact.shape[0] == 0:
                continue
        img_name = f"{video_name}_frame{frame}"
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        _ = cv2.imwrite(image_path, img)
        
        
video_labels = pd.read_csv(PATH_DF).fillna(0)
video_labels_with_impact = video_labels[video_labels['impact'] > 0]
for row in tqdm(video_labels_with_impact[['video','frame','label']].values):
    frames = np.array([-4,-3,-2,-1,1,2,3,4])+row[1]
    video_labels.loc[(video_labels['video'] == row[0]) 
                                 & (video_labels['frame'].isin(frames))
                                 & (video_labels['label'] == row[2]), 'impact'] = 1
video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(str) + '.png'
video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > ALL_DATA].reset_index(drop=True)
video_labels['impact'] = video_labels['impact'].astype(int)+1
video_labels['x'] = video_labels['left']
video_labels['y'] = video_labels['top']
video_labels['w'] = video_labels['width']
video_labels['h'] = video_labels['height']

np.random.seed(0)
video_names = np.random.permutation(video_labels.video.unique())
valid_video_len = int(len(video_names)*0.2)
video_valid = video_names[:valid_video_len]
video_train = video_names[valid_video_len:]
images_valid = video_labels[ video_labels.video.isin(video_valid)].image_name.unique()
images_train = video_labels[~video_labels.video.isin(video_valid)].image_name.unique()

uniq_video = video_labels.video.unique()

from tqdm.auto import tqdm
for video_name in tqdm(uniq_video):
    mk_images(video_name, video_labels, video_dir, out_dir)

df = video_labels.copy()
df['x_center'] = df['left'] + df['width']/2
df['y_center'] = df['top'] + df['height']/2
df = df[['image_name','left', 'top', 'width', 'height','x_center','y_center','impact']]

df.columns = ['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']
index = list(set(df.image_id))

source = 'train'
if True:
    for fold in [0]:
        for name,mini in tqdm(df.groupby('image_id')):
            name = name.split(".png")[0]
            if name in images_valid:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)
            with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h']].astype(float)
                row['y_center'] = row['y_center']/720.0
                row['h'] = row['h']/720.0
                row['x_center'] = row['x_center']/1280.0
                row['w'] = row['w']/1280.0
                row = row.values.astype(str)
                for j in range(len(row)):
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))
            sh.move("train_images/{}.png".format(name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))


        
        


