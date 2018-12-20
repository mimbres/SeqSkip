#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:45:08 2018

rnbc2 sep UserEmb(80+40)cv2

RN with batch + classifier
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
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type = str, default = "./config_init_dataset.json")
parser.add_argument("-s","--save_path",type = str, default = "./save/exp_rnbc2_sepUE256v2L_all/")
parser.add_argument("-l","--load_continue_latest",type = str, default = None)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-e","--epochs",type = int, default= 10)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type = int, default = 2048)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
CLASS_NUM = args.class_num
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
TR_BATCH_SZ = args.train_batch_size
GPU = args.gpu

# Model-save directory
MODEL_SAVE_PATH = args.save_path
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

hist_trloss = list()
hist_tracc  = list()
hist_vloss  = list()
hist_vacc   = list()
np.set_printoptions(precision=3)

#Feature encoder:
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
    def __init__(self, 
                 num_sup_max=10, num_que_max=10,
                 in_feat_sup_sz=64, in_feat_que_sz=64, 
                 in_log_sup_sz=41, in_log_que_sz=0,
                 in_label_sup_sz=3, ue_sz=128, ue_sz2=128): # 64,64,41,0,3
        super(RelationNetwork, self).__init__()
        self.num_sup_max = num_sup_max
        self.num_que_max = num_que_max
        self.in_feat_sup_sz = in_feat_sup_sz
        self.in_feat_que_sz = in_feat_que_sz
        self.in_log_sup_sz = in_log_sup_sz
        self.in_log_que_sz = in_log_que_sz
        self.in_label_sup_sz = in_label_sup_sz
        self.ue_sz = ue_sz
        self.ue_sz2 = ue_sz2
        self.layer1_input_sz = in_feat_sup_sz + in_feat_que_sz + ue_sz # SEP RELATION: we drop log and labels here.
        
        self.layer1 = nn.Sequential(
                        nn.Linear(self.layer1_input_sz, 256), # bx8x7x1*128 -> bx8x7x1*512
                        nn.LayerNorm(256),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.LayerNorm(128),
                        nn.ReLU())
        self.fc1 = nn.Linear(128+ue_sz2,64)
        self.fc2 = nn.Linear(64,1) 

        # Option: user embedding layer
        self.u_emb1 = nn.Sequential(
                        nn.Linear(108,ue_sz), # bx7x1*108 --> bx7x1*16
                        nn.LayerNorm(ue_sz),  
                        nn.ReLU(), 
                        nn.Conv1d(10,1,1,bias=False)) # 1x1Conv --> bx1x1*16
        
        self.u_emb2 = nn.Sequential(
                        nn.Linear(108,ue_sz2), # bx7x1*108 --> bx7x1*40
                        nn.LayerNorm(ue_sz2),  
                        nn.ReLU(), 
                        nn.Conv1d(10,1,1,bias=False), # 1x1Conv --> bx1x1*40
                        nn.ReLU())
        
        # Option: classifier 
        self.classifier = nn.Linear(10,1)

    def forward(self, x_sup, x_que, x_log_sup, y_log_que, label_sup):
        # x_feat_sup: bx7x1*64, x_feat_que: bx8x1*64
        # x_log_sup: bx7*41, label_sup:bx10*3
        ######### User Emb:(x_feat_sup, log_sup, label_sup)
        ue_source = torch.cat((x_sup.squeeze(2), x_log_sup, label_sup), 2) # bx7*108
        ue = self.u_emb1(ue_source) # bx1x16, 16D user embedding
    
        ######### Audio relation: (audio_sup, audio_que, 유저embedding)
        audio_relation_pairs = torch.cat((x_sup.unsqueeze(1).repeat(1,10,1,1,1),
                                    x_que.unsqueeze(2).repeat(1,1,10,1,1),
                                    ue.view(-1,1,1,1,self.ue_sz).repeat(1,10,10,1,1) ), 4) # bx8x7x1*144
        
        #             metric relation: (audio relation, 유저emb)
        out = self.layer1(audio_relation_pairs) # bx7x8x1*512
        out = self.layer2(out) # bx7x8x1*256
        
        ###
        #out = torch.cat((out, _ext_sup), 4) # bx8x7x1*264
        ue2 = self.u_emb2(ue_source).view(-1,1,1,1,self.ue_sz2).repeat(1,10,10,1,1)
        out = torch.cat((out, ue2), 4) #bx8x7x1*(256+ue_sz2)
        out = F.relu(self.fc1(out)) # -> bx7x8x1*64
        out = torch.sigmoid(self.fc2(out)) # -> bx7x8*1
        out = out.view(-1,10,10)
        out = self.classifier(out) # bx8x1
        out = out.view(-1,10) # bx8
        return out
    
        
    
#%%
def validate(mval_loader, FeatEnc, RN, eval_mode):
    tqdm.write("Validation...")
    submit = []
    gt     = []
    total_vloss    = 0
    total_vcorrects = 0
    total_vquery    = 0
    val_sessions_iter = iter(mval_loader)
    
    for val_session in trange(len(val_sessions_iter), desc='val-sessions', position=2, ascii=True):
        FeatEnc.eval(); RN.eval();        
        x_sup, x_que, x_log_sup, x_log_que, label_sup, label_que, num_items, index = val_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS
        x_sup, x_que = Variable(x_sup).cuda(GPU), Variable(x_que).cuda(GPU)
        x_log_sup, x_log_que   = Variable(x_log_sup).cuda(GPU), Variable(x_log_que).cuda(GPU)
        label_sup = Variable(label_sup).cuda(GPU)
                
        num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
        num_query   = num_items[:,1].detach().numpy().flatten()
        batch_sz    = num_items.shape[0]
 
        x_sup = x_sup.unsqueeze(2) # 1x7*29 --> 1x7x1*29
        x_que = x_que.unsqueeze(2) # 1x8*29 --> 1x8x1*29
        x_feat_sup = FeatEnc(x_sup) # 1x7x1*64     
        x_feat_que = FeatEnc(x_que) # 1x8x1*64
        
        y_hat = RN(x_feat_sup, x_feat_que, x_log_sup, x_log_que, label_sup) # bx8
        y_gt  = label_que[:,:,1]
        y_mask = np.zeros((batch_sz,10), dtype=np.float32)
        for b in np.arange(batch_sz):
            y_mask[b,:num_query[b]] = 1
        y_mask = torch.FloatTensor(y_mask).cuda(GPU)    

        loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask, target=y_gt.cuda(GPU)*y_mask)
        total_vloss += loss.item()
        
        # Decision
        y_prob = (torch.sigmoid(y_hat)*y_mask).detach().cpu().numpy()
        y_pred = ((torch.sigmoid(y_hat)>0.5).float()*y_mask).detach().cpu().long().numpy()
        
        # Eval, Submission
        _y_gt  = label_que[:,:,1].detach().cpu().numpy()
        if eval_mode is True:
            for b in np.arange(batch_sz):
                submit.append(y_pred[b,:num_query[b]].flatten())
                gt.append(_y_gt[b,:num_query[b]].flatten())
        
        # Prepare display
        sample_sup = label_sup[0,:num_support[0],1].detach().long().cpu().numpy().flatten() 
        sample_que = label_que[0,:num_query[0],1].long().numpy().flatten()
        sample_pred = y_pred[0,:num_query[0]].flatten()
        sample_prob = y_prob[0, :num_query[0]].flatten()
    
        # Acc
        total_vcorrects += np.sum((y_pred == label_que[:,:,1].long().numpy()) * y_mask.cpu().numpy())  
        total_vquery += np.sum(num_query)

        if (val_session+1)%400 == 0:
            tqdm.write("S:" + np.array2string(sample_sup) +'\n'+
                       "Q:" + np.array2string(sample_que) + '\n' +
                       "P:" + np.array2string(sample_pred) + '\n'+
                       "prob:" + np.array2string(sample_prob))
            tqdm.write("val_session:{0:}  vloss:{1:.6f}  vacc:{2:.4f}".format(val_session,total_vloss/val_session, total_vcorrects/total_vquery))
        # Restore GPU memory
        del loss, y_hat
        
     # Avg.Acc
    aacc = evaluate(submit, gt)
    tqdm.write("AACC={0:.6f}, FirstAcc={1:.6f}".format(aacc[0], aacc[1])) 
    
    hist_vloss.append(total_vloss/(val_session+1))
    hist_vacc.append(total_vcorrects/total_vquery)
    return submit
    

# Main
def main():
    # Trainset stats: 2072002577 items from 124950714 sessions
    print('Initializing dataloader...')
    mtrain_loader = SpotifyDataloader(config_fpath=args.config,
                                      mtrain_mode=True,
                                      data_sel=(0, 99965071), # 80% 트레인
                                      batch_size=TR_BATCH_SZ,
                                      shuffle=True) # shuffle은 True로 해야됨 나중에... 
    
    mval_loader  = SpotifyDataloader(config_fpath=args.config,
                                      mtrain_mode=True, # True, because we use part of trainset as testset
                                      data_sel=(3000000, 5000000),#(99965071, 124950714),#(99965071, 124950714), # 20%를 테스트
                                      batch_size=2048,
                                      shuffle=False) 
    
    # Init neural net
    #FeatEnc = MLP(input_sz=29, hidden_sz=512, output_sz=64).apply(weights_init).cuda(GPU)
    FeatEnc = MLP(input_sz=29, hidden_sz=256, output_sz=64).cuda(GPU)
    RN      = RelationNetwork().cuda(GPU)
    
    FeatEnc_optim = torch.optim.Adam(FeatEnc.parameters(), lr=LEARNING_RATE)
    RN_optim      = torch.optim.Adam(RN.parameters(), lr=LEARNING_RATE)
    
    FeatEnc_scheduler = StepLR(FeatEnc_optim, step_size=100000, gamma=0.2)
    RN_scheduler = StepLR(RN_optim, step_size=100000, gamma=0.2) 
    
    
    if args.load_continue_latest is None:
        START_EPOCH = 0  
        
    else:
        latest_fpath = max(glob.iglob(MODEL_SAVE_PATH + "check*.pth"),key=os.path.getctime)  
        checkpoint = torch.load(latest_fpath, map_location='cuda:{}'.format(GPU))
        tqdm.write("Loading saved model from '{0:}'... loss: {1:.6f}".format(latest_fpath,checkpoint['hist_trloss'][-1]))
        FeatEnc.load_state_dict(checkpoint['FE_state'])
        RN.load_state_dict(checkpoint['RN_state'])
        FeatEnc_optim.load_state_dict(checkpoint['FE_opt_state'])
        RN_optim.load_state_dict(checkpoint['RN_opt_state'])
        FeatEnc_scheduler.load_state_dict(checkpoint['FE_sch_state'])
        RN_scheduler.load_state_dict(checkpoint['RN_sch_state'])
        START_EPOCH = checkpoint['ep']
        
        
    for epoch in trange(START_EPOCH, EPOCHS, desc='epochs', position=0, ascii=True):
        
        tqdm.write('Train...')
        tr_sessions_iter = iter(mtrain_loader)
        total_corrects = 0
        total_query    = 0
        total_trloss   = 0
        for session in trange(len(tr_sessions_iter), desc='sessions', position=1, ascii=True):
            
            FeatEnc.train(); RN.train();
            x_sup, x_que, x_log_sup, x_log_que, label_sup, label_que, num_items, index = tr_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS
            x_sup, x_que = Variable(x_sup).cuda(GPU), Variable(x_que).cuda(GPU)
            x_log_sup, x_log_que   = Variable(x_log_sup).cuda(GPU), Variable(x_log_que).cuda(GPU)
            label_sup = Variable(label_sup).cuda(GPU)
            
            # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
            num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
            num_query   = num_items[:,1].detach().numpy().flatten()
            batch_sz    = num_items.shape[0]
     
            x_sup = x_sup.unsqueeze(2) # 1x7*29 --> 1x7x1*29
            x_que = x_que.unsqueeze(2) # 1x8*29 --> 1x8x1*29
            
            # - feature encoder
            x_feat_sup = FeatEnc(x_sup) # 1x7x1*64     
            x_feat_que = FeatEnc(x_que) # 1x8x1*64
            
            # - relation network
            y_hat = RN(x_feat_sup, x_feat_que, x_log_sup, x_log_que, label_sup) # bx8
            
            # Prepare ground-truth simlarity score and mask
            y_gt  = label_que[:,:,1]
            y_mask = np.zeros((batch_sz,10), dtype=np.float32)
            for b in np.arange(batch_sz):
                y_mask[b,:num_query[b]] = 1
            y_mask = torch.FloatTensor(y_mask).cuda(GPU)    
            
            # Calcultate BCE loss
            loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask, target=y_gt.cuda(GPU)*y_mask)
            total_trloss += loss.item()
        
            # Update Nets
            FeatEnc.zero_grad()
            RN.zero_grad()        
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(FeatEnc.parameters(), 0.5)
            #torch.nn.utils.clip_grad_norm_(RN.parameters(), 0.5)
            
            FeatEnc_optim.step()
            RN_optim.step()
            
            # Decision
            y_prob = (torch.sigmoid(y_hat)*y_mask).detach().cpu().numpy()
            y_pred = ((torch.sigmoid(y_hat)>0.5).float()*y_mask).detach().cpu().long().numpy()
    
            # Prepare display
            sample_sup = label_sup[0,:num_support[0],1].detach().long().cpu().numpy().flatten() 
            sample_que = label_que[0,:num_query[0],1].long().numpy().flatten()
            sample_pred = y_pred[0,:num_query[0]].flatten()
            sample_prob = y_prob[0, :num_query[0]].flatten()
        
            # Acc
            total_corrects += np.sum((y_pred == label_que[:,:,1].long().numpy()) * y_mask.cpu().numpy())  
            total_query += np.sum(num_query)
    
            # Restore GPU memory
            del loss, x_feat_sup, x_feat_que, y_hat 
    
            if (session+1)%900 == 0:
                hist_trloss.append(total_trloss/900)
                hist_tracc.append(total_corrects/total_query)
                tqdm.write("S:" + np.array2string(sample_sup) +'\n'+
                           "Q:" + np.array2string(sample_que) + '\n' +
                           "P:" + np.array2string(sample_pred) + '\n'+
                           "prob:" + np.array2string(sample_prob))
                
                tqdm.write("tr_session:{0:}  tr_loss:{1:.6f}  tr_acc:{2:.4f}".format(session, hist_trloss[-1], hist_tracc[-1]))
                total_corrects = 0
                total_query    = 0
                total_trloss   = 0
            
            if (session+1)%4000 == 0:
                 # Validation
                 validate(mval_loader, FeatEnc, RN, eval_mode=True)
                 # Save
                 torch.save({'ep': epoch, 'sess':session, 'FE_state': FeatEnc.state_dict(), 'RN_state': RN.state_dict(), 'loss': hist_trloss[-1], 'hist_vacc': hist_vacc,
                             'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'FE_opt_state': FeatEnc_optim.state_dict(), 'RN_opt_state': RN_optim.state_dict(),
                             'FE_sch_state': FeatEnc_scheduler.state_dict(), 'RN_sch_state': RN_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))

        # Validation
        validate(mval_loader, FeatEnc, RN, eval_mode=True)
        # Save
        torch.save({'ep': epoch, 'sess':session, 'FE_state': FeatEnc.state_dict(), 'RN_state': RN.state_dict(), 'loss': hist_trloss[-1], 'hist_vacc': hist_vacc,
                    'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'FE_opt_state': FeatEnc_optim.state_dict(), 'RN_opt_state': RN_optim.state_dict(),
        'FE_sch_state': FeatEnc_scheduler.state_dict(), 'RN_sch_state': RN_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        
    
if __name__ == '__main__':
    main()
                

