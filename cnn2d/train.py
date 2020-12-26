
import argparse

import os
import sys
from efficientnet_pytorch import EfficientNet
import glob
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
import functools
import torch
from tqdm.auto import tqdm
import models
from data import *
from config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_utils import trainer
from utils import split_train_val
import os
from apex import amp
import pretrainedmodels
from torch import nn
import random
import torchvision

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='NFL trian cnn')
    my_parser.add_argument('--config',type=str)
    my_parser.add_argument('--fold',type=str)
    args = my_parser.parse_args()
    if True:
        set_seed(1234)
        model_config = eval(args.config)
        os.environ["CUDA_VISIBLE_DEVICES"]=model_config.gpu
        orig_stdout = sys.stdout
        f = open(f'log/{model_config.model_name}_fold{args.fold}.txt', 'w')
        sys.stdout = f
        t_df,v_df = split_train_val(data_config,fold=args.fold,FOLD_NUM=args.folds)
        df = pd.read_csv("../"+data_config.train_csv_path)
        t_df = df[df.fold!=args.fold]
        df1 = t_df[t_df.impact>0]
        df0 = t_df[t_df.impact<1]
        df0 = df0[df0.frame<100]
        df0 = df0[df0.frame>20]
        t_df = pd.concat([df1,df0.sample(frac=0.5)])
        t_df = t_df[(t_df.frame<100)&(t_df.frame>20)]
        v_df = df[df.fold==args.fold]
        v_df = v_df[v_df.frame<100]
        v_df = v_df[v_df.frame>20]
        train_dataset = DatasetRetriever2D(df=df,image_root=data_config.train_dir,
                                           add_channel=False,image_size=data_config.img_size_crop,
                                      transforms =get_training_augmentation() )#,mode='train')

        val_dataset = DatasetRetriever2D(df=df,image_root=data_config.train_dir,
                                           add_channel=False,image_size=data_config.img_size_crop,
                                      transforms =get_training_augmentation() )#,mode='train')

        val_dataset = DatasetRetriever2D(df=v_df,image_root=data_config.jpeg_dir,
                                    transforms=get_validation_augmentation(),
                                    add_channel=False)
        train = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=model_config.WORKERS, pin_memory=True)
        val = DataLoader(val_dataset, batch_size=model_config.batch_size*1, shuffle=False, num_workers=model_config.WORKERS, pin_memory=True)
        model = EfficientNet.from_pretrained("efficientnet-b5",num_classes=model_config.classes).cuda()

        optimizer = eval(model_config.optimizer)(model.parameters(),**model_config.optimizer_parm)
        scheduler = eval(model_config.scheduler)(optimizer,**model_config.scheduler_parm)
        loss_fn = eval(model_config.loss_fn)()

        model = torch.nn.DataParallel(model)
        model_config.model_name = "efficientnet-b5" + f"_cnn_{args.fold}"
        Trainer = trainer(loss_fn,model,optimizer,scheduler,config=model_config)
        Trainer.run(train,val)
        sys.stdout = orig_stdout
        f.close()

