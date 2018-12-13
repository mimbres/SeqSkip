#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:34:25 2018

@author: mimbres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import glob, os
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader import SpotifyDataloader
import time

parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type = str, default = "./config_init_dataset.json")
parser.add_argument("-s","--save_path",type = str, default = "./save/exp1/")
parser.add_argument("-l","--load_continue_latest",type = str, default = None)
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-e","--epochs",type = int, default= 1000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
#parser.add_argument("-e","--embed_hidden_unit",type=int, default=2)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
EPOCHS = args.epochs
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

mtrain_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=True, # True, because we use part of trainset as testset
                                  data_sel=(0, 99965071),#(99965071, 124950714), # 20%를 테스트
                                  batch_size=1,
                                  shuffle=False) 

#%%
train_sessions_iter = iter(mtrain_loader)
for train_session in trange(len(train_sessions_iter), desc='varify', position=0):
    feats, labels, num_items, index = train_sessions_iter.next()
    if train_session==0:
        a0f = feats.detach().numpy()
        a0l = labels.detach().numpy()
        #tqdm.write(np.array2string(a0f))
        tqdm.write(np.array2string(a0l))
        
    if train_session==1:
        a1f = feats.numpy()
        a1l = labels.numpy()
        #tqdm.write(a1f)
        #tqdm.write(a1l)
        
    if train_session==2:
        a2f = feats.numpy()
        a2l = labels.numpy()
        #tqdm.write(a2f)
        #tqdm.write(a2l)
        
    if train_session==3:
        a3f = feats.numpy()
        a3l = labels.numpy()
        #tqdm.write(a3f)
        #tqdm.write(a2l)
        
    
    
    if train_session > 4:
        cur_feat = feats.numpy()
        cur_labels = labels.numpy()
        if sum(sum(sum(a0f-cur_feat)))==0:
            tqdm.write("found same feature matching a0 in {}th data, label diff={}".format(train_session, sum(sum(sum(a0l-cur_labels)))))
            
        if sum(sum(sum(a1f-cur_feat)))==0:
            tqdm.write("found same feature matching a1 in {}th data, label diff={}".format(train_session, sum(sum(sum(a1l-cur_labels)))))
            
        if sum(sum(sum(a2f-cur_feat)))==0:
            tqdm.write("found same feature matching a2 in {}th data, label diff={}".format(train_session, sum(sum(sum(a2l-cur_labels)))))
            
        if sum(sum(sum(a3f-cur_feat)))==0:
            tqdm.write("found same feature matching a3 in {}th data, label diff={}".format(train_session, sum(sum(sum(a3l-cur_labels)))))
    
    #time.sleep(0.001)

    
    
    
    
    
    
    