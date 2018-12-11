#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:45:08 2018

@author: mimbres
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader import SpotifyDataloader

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-c","--config",type = str, default = "./config_init_dataset.json")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-s","--min_support_num_per_class",type = int, default = 1)
parser.add_argument("-q","--min_query_num_per_class",type = int, default = 0)
parser.add_argument("-e","--epochs",type = int, default= 1000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
#parser.add_argument("-e","--embed_hidden_unit",type=int, default=2)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
MIN_SUPPORT_NUM_PER_CLASS = args.min_support_num_per_class
MIN_QUERY_NUM_PER_CLASS = args.min_query_num_per_class
EPOCHS = args.epochs
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

# Trainset stats: 2072002577 items from 124950714 sessions
print('Loading data...')
mtrain_loader = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=True,
                                  data_sel=(0, 99965071), # 80% 트레인
                                  batch_size=1,
                                  shuffle=False) # shuffle은 True로 해야됨 나중에... 

mtest_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=True, # True, because we use part of trainset as testset
                                  data_sel=(99965071, 124950714), # 20%를 테스트
                                  batch_size=1,
                                  shuffle=False) 


# Init neural net

 # optimizer
 
#relation_net_optim#

for epoch in trange(EPOCHS, desc='epochs', position=0):

    sessions_iter = iter(mtrain_loader)
    for session in trange(len(session_iter), desc='sessions', position=1):        
        #tqdm.write('sef')
        feats, labels, num_items, index = sessions_iter.next()
        feats, labels = Variable(feats).cuda(GPU), Variable(labels).cuda(GPU) 
    
        # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...
        num_support = int(num_items/2) # 
        num_query   = int(num_items) - num_support
        
        x_support = feats[:, :num_support, :] # bx7x70 (batch x sample x dim)
        x_support = torch.cat((x_support, labels[:,:num_support,:]), dim=2) # bx7x73, we use skip labels as feature, too.
        x_query   = feats[:, num_support:num_items, :] # bx8x70
        
        # - tile tensors for coding relations
        x_support_ext = x_support.repeat(1, num_query, 1) # bx56x73
        x_query_ext   = x_query.repeat(1, num_support, 1) # bx56x70
        # - generate relation pairs
        x_relation_pairs = torch.cat((x_support_ext, x_query_ext),2) # bx56x143
        
        # - generate relation ground-truth
        y_support_ext = labels[:, :num_support, 1].repeat(1, num_query) # bx7 --> bx56
        y_query_ext   = labels[:, num_support:num_items, 1].repeat(1, num_support) # bx56
        y_relation = (y_support_ext==y_query_ext).float()  # bx56x1
        
        # Forward Propagation
        
        # Calcultate MSE loss
        #loss = F.mse_loss(x_relation_pairs[:,:,1], y_relation)