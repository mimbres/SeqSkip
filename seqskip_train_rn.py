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
                                  shuffle=True) # shuffle은 True로 해야됨 나중에... 

mtest_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=True, # True, because we use part of trainset as testset
                                  data_sel=(99965071, 100065071),#(99965071, 124950714), # 20%를 테스트
                                  batch_size=1,
                                  shuffle=False) 



# Feature encoder:
class MLP(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_sz, hidden_sz) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sz, output_sz)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Model: Relation Nets
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_sz):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Linear(input_sz, 256), # 56x1x515 -> 56x1x256
                        nn.LayerNorm(256),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Linear(256, 64),
                        nn.LayerNorm(64),
                        nn.ReLU())
        self.fc1 = nn.Linear(64,8)
        self.fc2 = nn.Linear(8,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
        
# Init neural net
FeatEnc = MLP(input_sz=70, hidden_sz=512, output_sz=256).cuda(GPU)
RN      = RelationNetwork(input_sz=515).cuda(GPU)

FeatEnc_optim = torch.optim.Adam(FeatEnc.parameters(), lr=LEARNING_RATE)
RN_optim      = torch.optim.Adam(RN.parameters(), lr=LEARNING_RATE)

FeatEnc_scheduler = StepLR(FeatEnc_optim, step_size=100000, gamma=0.2)
RN_scheduler = StepLR(RN_optim, step_size=100000, gamma=0.2)

 
#relation_net_optim#
#%%
hist_trloss = list()
hist_vloss = list()
hist_vacc    = list()


def validate():
    tqdm.write("Validation...")
    total_vloss    = 0
    total_corrects = 0
    total_query    = 0
    val_sessions_iter = iter(mtest_loader)
    
    for val_session in trange(len(val_sessions_iter), desc='val-sessions', position=2):
        FeatEnc.eval(); RN.eval()
        feats, labels, num_items, index = val_sessions_iter.next()
        feats, labels = Variable(feats).cuda(GPU), Variable(labels).cuda(GPU) 
        num_support = int(num_items/2) # 
        num_query   = int(num_items) - num_support
        
        x_support = feats[:, :num_support, :].permute(1,0,2) # 7x1x70
        x_query   = feats[:, num_support:num_items, :].permute(1,0,2) # 8x1*70 (batch x ch x dim)
        x_feat_support = FeatEnc(x_support) # 7x1x256
        x_feat_query   = FeatEnc(x_query)   # 8x1x256
        x_feat_support = torch.cat((x_feat_support, labels[:, :num_support, :].view(-1,1,3)), 2) #7x1x259
        x_feat_support_ext = x_feat_support.unsqueeze(0).repeat(num_query,1,1,1) # 8x7x1*259
        x_feat_query_ext   = x_feat_query.unsqueeze(0).repeat(num_support,1,1,1) # 7x8x1*256
        x_feat_query_ext = torch.transpose(x_feat_query_ext,0,1) # 8x7x1*256
        x_feat_relation_pairs = torch.cat((x_feat_support_ext, x_feat_query_ext),3) # 8x7x1*515
        x_feat_relation_pairs = x_feat_relation_pairs.view(num_support*num_query, 1, -1) # 56x1*515
        
        y_support_ext = labels[:, :num_support, 1].view(-1).repeat(num_query) # 1x7->1x56
        y_query_ext   = labels[:, num_support:num_items, 1].repeat(num_support,1) 
        y_query_ext   = torch.transpose(y_query_ext,0,1).contiguous().view(-1) # 1x8->1x56
        y_relation = (y_support_ext==y_query_ext).float().view(-1,1)  # 56x1
        y_hat_relation = RN(x_feat_relation_pairs) # 56x1
        
        loss = F.mse_loss(input=y_hat_relation, target=y_relation)
        total_vloss += loss.item()
        
        sim_score = torch.FloatTensor(np.zeros((num_support*num_query,2))).cuda(GPU)
        sim_score[:,0] = y_hat_relation.view(-1) * (y_support_ext == 0).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 1).float()
        sim_score[:,1] = y_hat_relation.view(-1) * (y_support_ext == 1).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 0).float()
        sim_score = sim_score.view(num_query,-1,2) #8x7x2 (que x sup x class)
        y_query = labels[:, num_support:num_items, 1].long().cpu() # 8
        total_corrects += np.sum((torch.argmax(sim_score.sum(1),1).cpu() == y_query).numpy())  
        total_query += num_query
        if (val_session+1)%2500 == 0:
            tqdm.write("val_session:{0:}  loss:{1:.6f}  acc:{2:.4f}".format(val_session,loss.item(), total_corrects/total_query))
        
    hist_vloss.append(total_vloss/val_session)
    hist_vacc.append(total_corrects/total_query)


for epoch in trange(EPOCHS, desc='epochs', position=0):
    
    tqdm.write('Train...')
    sessions_iter = iter(mtrain_loader)
    for session in trange(len(sessions_iter), desc='sessions', position=1):
        
        FeatEnc.train(); RN.train()
        
        feats, labels, num_items, index = sessions_iter.next()
        feats, labels = Variable(feats).cuda(GPU), Variable(labels).cuda(GPU) 
    
        # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...
        num_support = int(num_items/2) # 
        num_query   = int(num_items) - num_support
        
        x_support = feats[:, :num_support, :].permute(1,0,2) # 7x1x70
        x_query   = feats[:, num_support:num_items, :].permute(1,0,2) # 8x1*70 (batch x ch x dim)
        
        # - feature encoder
        x_feat_support = FeatEnc(x_support) # 7x1x256
        x_feat_query   = FeatEnc(x_query)   # 8x1x256
        # - concat support labels
        x_feat_support = torch.cat((x_feat_support, labels[:, :num_support, :].view(-1,1,3)), 2) #7x1x259
        
        # - tile tensors for coding relations
        x_feat_support_ext = x_feat_support.unsqueeze(0).repeat(num_query,1,1,1) # 8x7x1*259
        x_feat_query_ext   = x_feat_query.unsqueeze(0).repeat(num_support,1,1,1) # 7x8x1*256
        x_feat_query_ext = torch.transpose(x_feat_query_ext,0,1) # 8x7x1*256
        # - generate relation pairs
        x_feat_relation_pairs = torch.cat((x_feat_support_ext, x_feat_query_ext),3) # 56x1*515
        x_feat_relation_pairs = x_feat_relation_pairs.view(num_support*num_query, 1, -1)
        # - generate relation ground-truth
        y_support_ext = labels[:, :num_support, 1].view(-1).repeat(num_query) # 1x7->1x56
        y_query_ext   = labels[:, num_support:num_items, 1].repeat(num_support,1) 
        y_query_ext   = torch.transpose(y_query_ext,0,1).contiguous().view(-1) # 1x8->1x56
        y_relation = (y_support_ext==y_query_ext).float().view(-1,1)  # 56x1
        
        # Forward Propagation
        y_hat_relation = RN(x_feat_relation_pairs) # 56x1
        # Calcultate MSE loss
        loss = F.mse_loss(input=y_hat_relation, target=y_relation)
        
        
        # Update Nets
        FeatEnc.zero_grad()
        RN.zero_grad()        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(FeatEnc.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(RN.parameters(), 0.5)
        
        FeatEnc_optim.step()
        RN_optim.step()
        
        if (session+1)%5000 == 0:
            tqdm.write("session:{0:}  loss:{1:.6f}".format(session, loss.item()))
            hist_trloss.append(loss)
        
        if (session+1)%50000 == 0:
            # Validation
            validate()
         
            

