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
import glob, os
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader import SpotifyDataloader

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

# Model-save directory
MODEL_SAVE_PATH = args.save_path
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# Trainset stats: 2072002577 items from 124950714 sessions
print('Initializing dataloader...')
mtrain_loader = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=True,
                                  data_sel=(0, 99965071), # 80% 트레인
                                  batch_size=1,
                                  shuffle=True) # shuffle은 True로 해야됨 나중에... 

mtest_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=True, # True, because we use part of trainset as testset
                                  data_sel=(99965071, 99975071),#(99965071, 124950714), # 20%를 테스트
                                  batch_size=1,
                                  shuffle=True) 



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
        #n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
        
# Init neural net
FeatEnc = MLP(input_sz=70, hidden_sz=512, output_sz=256).apply(weights_init).cuda(GPU)
RN      = RelationNetwork(input_sz=515).apply(weights_init).cuda(GPU)

FeatEnc_optim = torch.optim.Adam(FeatEnc.parameters(), lr=LEARNING_RATE)
RN_optim      = torch.optim.Adam(RN.parameters(), lr=LEARNING_RATE)

FeatEnc_scheduler = StepLR(FeatEnc_optim, step_size=100000, gamma=0.2)
RN_scheduler = StepLR(RN_optim, step_size=100000, gamma=0.2)

 
#relation_net_optim#
#%%
hist_trloss = list()
hist_tracc  = list()
hist_vloss = list()
hist_vacc    = list()


def validate():
    tqdm.write("Validation...")
    total_vloss    = 0
    total_vcorrects = 0
    total_vquery    = 0
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
        
        sim_score = torch.FloatTensor(np.zeros((num_support*num_query,2))).detach().cpu()
        sim_score[:,0] = y_hat_relation.view(-1) * (y_support_ext == 0).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 1).float()
        sim_score[:,1] = y_hat_relation.view(-1) * (y_support_ext == 1).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 0).float()
        sim_score = sim_score.view(num_query,-1,2) #8x7x2 (que x sup x class)
        y_query = labels[:, num_support:num_items, 1].long().cpu() # 8
        total_vcorrects += np.sum((torch.argmax(sim_score.sum(1),1).cpu() == y_query).numpy())  
        total_vquery += num_query
        if (val_session+1)%2000 == 0:
            tqdm.write(np.array2string(sim_score.detach().cpu().numpy()))
            tqdm.write(np.array2string(labels[:, :num_items, 1].detach().cpu().numpy().flatten()))
            tqdm.write("val_session:{0:}  vloss:{1:.6f}  vacc:{2:.4f}".format(val_session,loss.item(), total_vcorrects/total_vquery))
            
        
    hist_vloss.append(total_vloss/val_session)
    hist_vacc.append(total_vcorrects/total_vquery)
    

# Main
if args.load_continue_latest is None:
    START_EPOCH = 0
else:
    latest_fpath = max(glob.iglob(MODEL_SAVE_PATH + "check*.pth"),key=os.path.getctime)  
    checkpoint = torch.load(latest_fpath)
    tqdm.write("Loading saved model from '{0:}'... loss: {1:.6f}".format(latest_fpath,checkpoint['loss']))
    FeatEnc.load_state_dict(checkpoint['FE_state'])
    RN.load_state_dict(checkpoint['RN_state'])
    FeatEnc_optim.load_state_dict(checkpoint['FE_opt_state'])
    RN_optim.load_state_dict(checkpoint['RN_opt_state'])
    FeatEnc_scheduler.load_state_dict(checkpoint['FE_sch_state'])
    RN_scheduler.load_state_dict(checkpoint['RN_sch_state'])
    START_EPOCH = checkpoint['ep']
    
    
for epoch in trange(START_EPOCH, EPOCHS, desc='epochs', position=0):
    
    tqdm.write('Train...')
    sessions_iter = iter(mtrain_loader)
    total_corrects = 0
    total_query    = 0
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
        
        # Train acc
        sim_score = torch.FloatTensor(np.zeros((num_support*num_query,2))).detach().cpu()
        sim_score[:,0] = y_hat_relation.view(-1) * (y_support_ext == 0).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 1).float()
        sim_score[:,1] = y_hat_relation.view(-1) * (y_support_ext == 1).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 0).float()
        sim_score = sim_score.view(num_query,-1,2) #8x7x2 (que x sup x class)
        y_query = labels[:, num_support:num_items, 1].long().cpu() # 8
        total_corrects += np.sum((torch.argmax(sim_score.sum(1),1).cpu() == y_query).numpy())  
        total_query += num_query
        if (session+1)%2000 == 0:
            hist_trloss.append(loss.item())
            hist_tracc.append(total_corrects/total_query)
            tqdm.write(np.array2string(sim_score.detach().cpu().numpy()))
            tqdm.write(np.array2string(labels[:, :num_items, 1].detach().cpu().numpy().flatten()))
            tqdm.write("tr_session:{0:}  tr_loss:{1:.6f}  tr_acc:{2:.4f}".format(hist_trloss[-1], hist_tracc[-1]))
            total_corrects = 0
            total_query    = 0
            
        
        if (session+1)%10000 == 0:
            tqdm.write("session:{0:}  loss:{1:.6f}".format(session, loss.item()))
        
        if (session+1)%20000 == 0:
            # Validation
            validate()
            # Save
            torch.save({'ep': epoch, 'sess':session, 'FE_state': FeatEnc.state_dict(), 'RN_state': RN.state_dict(), 'loss': loss, 'hist_vacc': hist_vacc,
                        'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'FE_opt_state': FeatEnc_optim.state_dict(), 'RN_opt_state': RN_optim.state_dict(),
            'FE_sch_state': FeatEnc_scheduler.state_dict(), 'RN_sch_state': RN_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
         
            

