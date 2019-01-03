#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 14:50:41 2018

@author: mimbres
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.backends import cudnn
import numpy as np
import glob, os
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader import SpotifyDataloader
from utils.eval import evaluate
from blocks.highway_dil_conv import HighwayDCBlock
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type=str, default="./config_init_dataset.json")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_seq1H_genlog128/")
parser.add_argument("-l","--load_continue_latest",type=str, default=None)
parser.add_argument("-glu","--use_glu", type=bool, default=False)
parser.add_argument("-w","--class_num",type=int, default = 2)
parser.add_argument("-e","--epochs",type=int, default= 15)
parser.add_argument("-lr","--learning_rate", type=float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type=int, default = 2048)
parser.add_argument("-tsb","--test_batch_size", type=int, default = 1024)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
USE_GLU    = args.use_glu
INPUT_DIM = 72 if USE_SUPLOG else 31

CLASS_NUM = args.class_num
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
TR_BATCH_SZ = args.train_batch_size
TS_BATCH_SZ = args.test_batch_size
GPU = args.gpu

# Model-save directory
MODEL_SAVE_PATH = args.save_path
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


hist_trloss = list()
hist_tracc  = list()
hist_vloss  = list()
hist_vacc   = list()

hist_trloss_qlog = list()
hist_trloss_skip = list()
hist_vloss_qlog =  list()
hist_vloss_skip =  list()
np.set_printoptions(precision=3)

class SeqFeatEnc(nn.Module):  
    def __init__(self, input_dim, e_ch, #d_ch=256,
                 #h_io_chs=[256, 256, 256, 256, 256, 256, 256],
                 d_ch,
                 h_io_chs=[1,1,1,1,1,1,1],
                 h_k_szs=[2,2,2,2,2,1,1],
                 h_dils=[1,2,4,8,16,1,1],
#                 h_dils=[1,2,4,1,2,4,1,2,4,1,1,1,1],  #이것도 Receptive Field가 20인데 왜 안되는걸까??????
                 use_glu=False):
        super(SeqFeatEnc, self).__init__()
        h_io_chs[:] = [n * d_ch for n in h_io_chs]
        # Layers:
        self.mlp = nn.Sequential(nn.Conv1d(input_dim,e_ch,1),
                                 nn.ReLU(),
                                 nn.Conv1d(e_ch,d_ch,1))
        self.h_block = HighwayDCBlock(h_io_chs, h_k_szs, h_dils, causality=True, use_glu=use_glu)
        return None
 
    def forward(self, x):
        # Input={{x_sup,x_que};{label_sup,label_que}}  BxC*T (Bx(29+1)*20), audio feat dim=29, label dim=1, n_sup+n_que=20
        # Input bx30x20
        x = self.mlp(x) # bx128*20
        x = self.h_block(x) #bx256*20, 여기서 attention 쓰려면 split 128,128
        return x#x[:,:128,:]
        
class SeqClassifier(nn.Module):
    def __init__(self, input_ch, e_ch,
                 h_io_chs=[1,1,1,1,1,1,1],
                 h_k_szs=[2,2,2,2,2,1,1],
                 h_dils=[1,2,4,8,16,1,1],
                 use_glu=False):
        super(SeqClassifier, self).__init__()
        h_io_chs[:] = [n * e_ch for n in h_io_chs]
        self.front_1x1 = nn.Conv1d(input_ch, e_ch,1)
        self.h_block = HighwayDCBlock(h_io_chs, h_k_szs, h_dils, causality=True, use_glu=use_glu)
        self.last_1x1  = nn.Sequential(nn.Conv1d(e_ch,e_ch,1), nn.ReLU(),
                                       nn.Conv1d(e_ch,e_ch,1), nn.ReLU())    
        self.classifier = nn.Sequential(nn.Conv1d(e_ch,e_ch,1), nn.ReLU(),
                                        nn.Conv1d(e_ch,e_ch,1))#nn.Conv1d(e_ch,1,1))
    def forward(self, x): # Input:bx256*20
        x = self.front_1x1(x) # bx128*20
        x = self.h_block(x)   # bx128*20
        x = self.last_1x1(x)  # bx64*20
        return self.classifier(x).squeeze(1) # bx20

     

class SeqModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, e_ch=128, d_ch=128, use_glu=USE_GLU):
        super(SeqModel, self).__init__()
        self.enc = SeqFeatEnc(input_dim=input_dim, e_ch=e_ch, d_ch=d_ch, use_glu=use_glu)
        self.clf = SeqClassifier(input_ch=d_ch, e_ch=e_ch, use_glu=use_glu)
        self.qlog_classifier = nn.Sequential(nn.Conv1d(e_ch,e_ch,1), nn.ReLU(),
                                        nn.Conv1d(e_ch,41,1))#nn.Conv1d(e_ch,1,1))
        self.skip_classifier = nn.Sequential(nn.Conv1d(e_ch,e_ch,1), nn.ReLU(),
                                        nn.Conv1d(e_ch,1,1))#nn.Conv1d(e_ch,1,1))
        
    def forward(self, x):
        x = self.enc(x) # bx128x20
        x = self.clf(x) # bx128x20
        #x_qlog, x_skip = x[:,:41,:], x[:,41,:] 
        x_qlog = self.qlog_classifier(x) # bx41*20
        x_skip = self.skip_classifier(x).squeeze(1) # bx20
        return x_qlog, x_skip 