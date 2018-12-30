#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:44:03 2018
Student Net
S_seq1eH_e

- non-autoregressive (not feeding predicted labels)
- instance Norm.
- G: GLU version
- H: Highway-net version

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
from blocks.highway_glu_dil_conv_v2 import HighwayDCBlock
from copy import deepcopy
#from blocks.multihead_attention import MultiHeadAttention
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type=str, default="./config_init_dataset.json")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_S_seq1eH_eBCEL1sig/")
parser.add_argument("-l","--load_continue_latest",type=str, default=None)
parser.add_argument("-t","--load_teacher_net_fpath",type=str, default="./save/exp_T_seq1eH/check_8_48811.pth")
parser.add_argument("-glu","--use_glu", type=bool, default=False)
parser.add_argument("-w","--class_num",type=int, default = 2)
parser.add_argument("-e","--epochs",type=int, default= 10)
parser.add_argument("-lr","--learning_rate", type=float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type=int, default = 2048)
parser.add_argument("-tsb","--test_batch_size", type=int, default = 1024)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
FPATH_T_NET_CHECKPOINT = args.load_teacher_net_fpath
USE_GLU    = args.use_glu
INPUT_DIM = 72 

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
np.set_printoptions(precision=3)
       
class SeqEncoder(nn.Module):
    def __init__(self, input_ch, e_ch,
                 h_k_szs=[2,2,2,3,1,1], #h_k_szs=[2,2,5,1,1],
                 h_dils=[1,2,4,8,1,1],
                 causality=True,
                 use_glu=False):
        super(SeqEncoder, self).__init__()
        h_io_chs = [e_ch]*len(h_k_szs)
        self.front_1x1 = nn.Conv1d(input_ch, e_ch,1)
        self.h_block = HighwayDCBlock(h_io_chs, h_k_szs, h_dils, causality=causality, use_glu=use_glu)
        self.mid_1x1  = nn.Sequential(nn.Conv1d(e_ch,e_ch,1), nn.ReLU(),
                                       nn.Conv1d(e_ch,e_ch,1), nn.ReLU())
        self.last_1x1 = nn.Sequential(nn.Conv1d(e_ch,e_ch,1))

    def forward(self, x): # Input:bx(input_dim)*20
        x = self.front_1x1(x) # bx128*20
        x = self.h_block(x)   # bx128*20
        x = self.mid_1x1(x)  # bx128*20
        return self.last_1x1(x) # bx128*20
        

class SeqModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, e_ch=128, d_ch=128, use_glu=USE_GLU):
        super(SeqModel, self).__init__()
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.enc = SeqEncoder(input_ch=input_dim, e_ch=e_ch,
                                  h_k_szs=[2,2,2,3,1,1], #h_k_szs=[2,2,2,3,1,1],
                                  h_dils=[1,2,4,8,1,1], #h_dils=[1,2,4,8,1,1],
                                  use_glu=use_glu) # bx128*10
        
        self.feature = nn.Sequential(nn.Conv1d(d_ch,d_ch,1), nn.ReLU(),
                                        nn.Conv1d(d_ch,d_ch,1), nn.ReLU())
        self.classifier = nn.Conv1d(d_ch,1,1)
        
    def forward(self, x):
        x = self.enc(x) # bx128*10 
        x = self.feature(x)
        x = self.classifier(x).squeeze(1) # bx256*10 --> b*10
        return x# bx20
    
class SeqModel_Student(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, e_ch=128, d_ch=128, use_glu=USE_GLU):
        super(SeqModel_Student, self).__init__()
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.enc = SeqEncoder(input_ch=input_dim, e_ch=e_ch,
                                  h_k_szs=[2,2,2,3,1,1], #h_k_szs=[2,2,2,3,1,1],
                                  h_dils=[1,2,4,8,1,1], #h_dils=[1,2,4,8,1,1],
                                  use_glu=use_glu) # bx128*10
        
        self.feature = nn.Sequential(nn.Conv1d(d_ch,d_ch,1), nn.ReLU(),
                                        nn.Conv1d(d_ch,d_ch,1), nn.ReLU())
        self.classifier = nn.Conv1d(d_ch,1,1)
        
    def forward(self, x):
        enc_out = self.enc(x) # bx128*10 
        x = self.feature(enc_out)
        x = self.classifier(x).squeeze(1) # bx256*10 --> b*10
        return enc_out, x# bx20

#%%


def validate(mval_loader, SM, eval_mode, GPU):
    tqdm.write("Validation...")
    submit = []
    gt     = []
    total_vloss    = 0
    total_vcorrects = 0
    total_vquery    = 0
    val_sessions_iter = iter(mval_loader)
    
    for val_session in trange(len(val_sessions_iter), desc='val-sessions', position=2, ascii=True):
        SM.eval()        
        x, labels, y_mask, num_items, index = val_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS
        # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
        num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
        num_query   = num_items[:,1].detach().numpy().flatten()
        batch_sz    = num_items.shape[0]
        
        # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
        x = x.permute(0,2,1) # bx70*20
        x_feat = torch.zeros(batch_sz, 72, 20)
        x_feat[:,:70,:] = x.clone()
        x_feat[:, 70,:10] = 1  
        x_feat[:, 71,:10] = labels[:,:10].clone()
        x_feat = Variable(x_feat, requires_grad=False).cuda(GPU)
        
        # y
        y = labels.clone()
        
        # y_mask
        y_mask_que = y_mask.clone()
        y_mask_que[:,:10] = 0
        
        # Forward & update
        _, y_hat = SM(x_feat) # y_hat: b*20

#        if USE_PRED_LABEL is True:
#            # Predict
#            li = 70 if USE_SUPLOG is True else 29 # the label's dimension indice
#            _x = x[:,:,:11] # bx72*11
#            for q in range(11,20):
#                y_hat = SM(Variable(_x, requires_grad=False)) # will be bx11 at the first round 
#                # Append next features
#                _x = torch.cat((_x, x[:,:,q].unsqueeze(2)), 2) # now bx72*12
#                _x[:,li,q] = torch.sigmoid(y_hat[:,-1])
#            y_hat = SM(Variable(_x, requires_grad=False)) # y_hat(final): bx20
#            del _x
#        else:
#            y_hat = SM(x)
        
        # Calcultate BCE loss: 뒤에q만 봄
        loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))
        total_vloss += loss.item()
        
        # Decision
        y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
        y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
        y_numpy = labels[:,10:].numpy() # bx10
        # Acc
        total_vcorrects += np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
        total_vquery += np.sum(num_query)
        
        # Eval, Submission
        if eval_mode is not 0:
            for b in np.arange(batch_sz):
                submit.append(y_pred[b,:num_query[b]].flatten())
                gt.append(y_numpy[b,:num_query[b]].flatten())
                
        if (val_session+1)%400 == 0:
            sample_sup = labels[0,(10-num_support[0]):10].long().numpy().flatten() 
            sample_que = y_numpy[0,:num_query[0]].astype(int)
            sample_pred = y_pred[0,:num_query[0]]
            sample_prob = y_prob[0,10:10+num_query[0]]
            tqdm.write("S:" + np.array2string(sample_sup) +'\n'+
                       "Q:" + np.array2string(sample_que) + '\n' +
                       "P:" + np.array2string(sample_pred) + '\n' +
                       "prob:" + np.array2string(sample_prob))
            tqdm.write("val_session:{0:}  vloss:{1:.6f}  vacc:{2:.4f}".format(val_session,loss.item(), total_vcorrects/total_vquery))
        del loss, y_hat, x # Restore GPU memory
        
    # Avg.Acc
    if eval_mode==1:
        aacc = evaluate(submit, gt)
        tqdm.write("AACC={0:.6f}, FirstAcc={1:.6f}".format(aacc[0], aacc[1]))     
        
    hist_vloss.append(total_vloss/val_session)
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
                                      shuffle=True,
                                      seq_mode=True) # seq_mode implemented  
    
    mval_loader  = SpotifyDataloader(config_fpath=args.config,
                                      mtrain_mode=True, # True, because we use part of trainset as testset
                                      data_sel=(99965071, 104965071),#(99965071, 124950714), # 20%를 테스트
                                      batch_size=TS_BATCH_SZ,
                                      shuffle=False,
                                      seq_mode=True) 
    
    # Load Teacher net
    SMT = SeqModel().cuda(GPU) 
    checkpoint = torch.load(FPATH_T_NET_CHECKPOINT, map_location='cuda:{}'.format(GPU))
    tqdm.write("Loading saved teacher model from '{0:}'... loss: {1:.6f}".format(FPATH_T_NET_CHECKPOINT,checkpoint['loss']))
    SMT.load_state_dict(checkpoint['SM_state'])
    
    SMT_Enc  = nn.Sequential(*list(SMT.children())[:1]).cuda(GPU)
    #SMT_EncFeat = nn.Sequential(*list(SMT.children())[:2])
    
    
    # Init Student net --> copy classifier from the Teacher net
    SM = SeqModel_Student().cuda(GPU)
    SM.feature = deepcopy(SMT.feature)
    for p in list(SM.feature.parameters()):
        p.requires_grad = False
    SM.classifier = deepcopy(SMT.classifier)
    SM.classifier.weight.requires_grad = False
    SM.classifier.bias.requires_grad = False
    SM = SM.cuda(GPU)
    
    SM_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, SM.parameters()), lr=LEARNING_RATE)
    SM_scheduler = StepLR(SM_optim, step_size=1, gamma=0.9)  
    
    
    
    
    # Load checkpoint
    if args.load_continue_latest is None:
        START_EPOCH = 0        
    else:
        latest_fpath = max(glob.iglob(MODEL_SAVE_PATH + "check*.pth"),key=os.path.getctime)  
        checkpoint = torch.load(latest_fpath, map_location='cuda:{}'.format(GPU))
        tqdm.write("Loading saved model from '{0:}'... loss: {1:.6f}".format(latest_fpath,checkpoint['loss']))
        SM.load_state_dict(checkpoint['SM_state'])
        SM_optim.load_state_dict(checkpoint['SM_opt_state'])
        SM_scheduler.load_state_dict(checkpoint['SM_sch_state'])
        START_EPOCH = checkpoint['ep']
        
    # Train    
    for epoch in trange(START_EPOCH, EPOCHS, desc='epochs', position=0, ascii=True):
        tqdm.write('Train...')
        tr_sessions_iter = iter(mtrain_loader)
        total_corrects = 0
        total_query    = 0
        total_trloss   = 0
        for session in trange(len(tr_sessions_iter), desc='sessions', position=1, ascii=True):
            SMT.eval(); # Teacher-net
            SM.train(); # Student-net
            x, labels, y_mask, num_items, index = tr_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS 
            
            # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
            num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
            num_query   = num_items[:,1].detach().numpy().flatten()
            batch_sz    = num_items.shape[0]
            
            # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
            x = x.permute(0,2,1) # bx70*20
            
            # x_feat_T: Teacher-net input, x_feat_S: Student-net input(que-log is excluded)
            x_feat_T = torch.zeros(batch_sz, 72, 20)
            x_feat_T[:,:70,:] = x.clone()
            x_feat_T[:, 70,:10] = 1 # Sup/Que state indicator  
            x_feat_T[:, 71,:10] = labels[:,:10].clone()
                        
            x_feat_S = x_feat_T.clone()
            x_feat_S[:, :41, 10:] = 0 # remove que-log
            
            x_feat_T = x_feat_T.cuda(GPU)
            x_feat_S = Variable(x_feat_S).cuda(GPU)
            
            
            # Target: Prepare Teacher's intermediate output 
            enc_target = SMT_Enc(x_feat_T)
            #target = SMT_EncFeat(x_feat_T)
        
            
            # y_mask
            y_mask_que = y_mask.clone()
            y_mask_que[:,:10] = 0
            
            # Forward & update
            y_hat_enc, y_hat = SM(x_feat_S) # y_hat: b*20
            
            # Calcultate Distillation loss
            loss1 = F.binary_cross_entropy_with_logits(input=y_hat_enc, target=torch.sigmoid(enc_target.cuda(GPU)))
            loss2 = F.l1_loss(input=torch.sigmoid(y_hat_enc/10), target=torch.sigmoid(enc_target.cuda(GPU)/10))
            loss = loss1+loss2
            total_trloss += loss.item()
            SM.zero_grad()
            loss.backward()
            # Gradient Clipping
            #torch.nn.utils.clip_grad_norm_(SM.parameters(), 0.5)
            SM_optim.step()
            
            # Decision
            SM.eval();
            y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
            y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
            y_numpy = labels[:,10:].numpy() # bx10
            # Acc
            total_corrects += np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
            total_query += np.sum(num_query)
            
            # Restore GPU memory
            del loss, y_hat, y_hat_enc
    
            if (session+1)%500 == 0:
                hist_trloss.append(total_trloss/900)
                hist_tracc.append(total_corrects/total_query)
                # Prepare display
                sample_sup = labels[0,(10-num_support[0]):10].long().numpy().flatten() 
                sample_que = y_numpy[0,:num_query[0]].astype(int)
                sample_pred = y_pred[0,:num_query[0]]
                sample_prob = y_prob[0,10:10+num_query[0]]

                tqdm.write("S:" + np.array2string(sample_sup) +'\n'+
                           "Q:" + np.array2string(sample_que) + '\n' +
                           "P:" + np.array2string(sample_pred) + '\n' +
                           "prob:" + np.array2string(sample_prob))
                tqdm.write("tr_session:{0:}  tr_loss:{1:.6f}  tr_acc:{2:.4f}".format(session, hist_trloss[-1], hist_tracc[-1]))
                total_corrects = 0
                total_query    = 0
                total_trloss   = 0
                
            
            if (session+1)%25000 == 0:
                 # Validation
                 validate(mval_loader, SM, eval_mode=True, GPU=GPU)
                 # Save
                 torch.save({'ep': epoch, 'sess':session, 'SM_state': SM.state_dict(),'loss': hist_trloss[-1], 'hist_vacc': hist_vacc,
                             'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'SM_opt_state': SM_optim.state_dict(),
                             'SM_sch_state': SM_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        # Validation
        validate(mval_loader, SM, eval_mode=True, GPU=GPU)
        # Save
        torch.save({'ep': epoch, 'sess':session, 'SM_state': SM.state_dict(),'loss': hist_trloss[-1], 'hist_vacc': hist_vacc,
                    'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'SM_opt_state': SM_optim.state_dict(),
                    'SM_sch_state': SM_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        SM_scheduler.step()
    
if __name__ == '__main__':
    main()  
