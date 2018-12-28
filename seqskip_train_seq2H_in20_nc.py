#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:45:08 2018

seq2eH_in20: seqeunce learning model with separate encoder for support and query, 1stack each 
- non-autoregressive (not feeding predicted labels)
- instance Norm.
- G: GLU version
- H: Highway-net version
- applied more efficient dilated conv over seq1
- non-causal for sup
- using sup+query as input (20)
-

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
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type=str, default="./config_init_dataset.json")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_seq2H_in20_nc/")
parser.add_argument("-l","--load_continue_latest",type=str, default=None)
parser.add_argument("-spl","--use_suplog_as_feat", type=bool, default=True)
parser.add_argument("-qf","--use_quelog_as_feat", type=bool, default=True)
parser.add_argument("-pl","--use_predicted_label", type=bool, default=False)
parser.add_argument("-glu","--use_glu", type=bool, default=False)
parser.add_argument("-w","--class_num",type=int, default = 2)
parser.add_argument("-e","--epochs",type=int, default= 10)
parser.add_argument("-lr","--learning_rate", type=float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type=int, default = 2048)
parser.add_argument("-tsb","--test_batch_size", type=int, default = 1024)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
USE_SUPLOG = args.use_suplog_as_feat
USE_QUELOG = args.use_quelog_as_feat
USE_PRED_LABEL = args.use_predicted_label
USE_GLU    = args.use_glu
INPUT_DIM_S = 71 if USE_SUPLOG else 30 # default: 72
INPUT_DIM_Q = 72 if USE_QUELOG else 29 # default: 31

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
    def __init__(self, input_dim_s=INPUT_DIM_S, input_dim_q=INPUT_DIM_Q, e_ch=128, d_ch=256, use_glu=USE_GLU):
        super(SeqModel, self).__init__()
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.sup_enc = SeqEncoder(input_ch=input_dim_s, e_ch=d_ch, 
                                  h_k_szs=[3,3,3,1,1],
                                  h_dils=[1,3,9,1,1],
                                  causality=False,
                                  use_glu=use_glu) # bx256*10 
        self.que_enc = SeqEncoder(input_ch=input_dim_q, e_ch=e_ch,
                                  h_k_szs=[2,2,2,3,1,1], #h_k_szs=[2,2,2,3,1,1],
                                  h_dils=[1,2,4,8,1,1], #h_dils=[1,2,4,8,1,1],
                                  use_glu=use_glu) # bx128*10
        self.last_enc = SeqEncoder(input_ch=d_ch, e_ch=d_ch,
                                  h_k_szs=[2,2,3,1,1], #h_k_szs=[2,2,2,3,1,1],
                                  h_dils=[1,2,4,1,1], #h_dils=[1,2,4,8,1,1],
                                  use_glu=use_glu)
        self.classifier = nn.Sequential(nn.Conv1d(d_ch,e_ch,1), nn.ReLU(),
                                        nn.Conv1d(e_ch,1,1))
        
    def forward(self, x_sup, x_que):
        x_sup = self.sup_enc(x_sup) # bx256*10 
        x_que = self.que_enc(x_que) # bx128*10  
        
        # Attention: K,V from x_sup, Q from x_que
        x_sup = torch.split(x_sup, self.e_ch, dim=1) # K: x_sup[0], V: x_sup[1]
        att = F.softmax(torch.matmul(x_sup[0].transpose(1,2), x_que)/16, dim=1) # K'*Q: bx10*20
        x = torch.cat((torch.matmul(x_sup[1], att), x_que), 1) # {V*att, Q}: bx(128+128)*10     
        x = self.classifier(x).squeeze(1) # bx256*10 --> b*10
        return x, att # bx20, bx10x20

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
        x_sup = Variable(torch.cat((x[:,:,:10], labels[:,:10].unsqueeze(1)), 1)).cuda(GPU) # bx71(41+29+1)*10 
        
        x_que = torch.zeros(batch_sz, 72, 20)
        x_que[:,:41,:10] = x[:,:41,:10].clone() # fill with x_sup_log
        x_que[:,41:70,:] = x[:,41:,:].clone()   # fill with x_sup_feat and x_que_feat
        x_que[:, 70,:10] = 1                    # support marking
        x_que[:, 71,:10] = labels[:,:10] # labels marking
        x_que = Variable(x_que).cuda(GPU) # bx29*10

        # y 
        y = labels.clone() # bx20
        
        # y_mask
        y_mask_que = y_mask.clone()
        y_mask_que[:,:10] = 0
        
        # Forward & update
        y_hat, att = SM(x_sup, x_que) # y_hat: b*20, att: bx10*20

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
    
    # Init neural net
    SM = SeqModel().cuda(GPU)
    SM_optim = torch.optim.Adam(SM.parameters(), lr=LEARNING_RATE)
    SM_scheduler = StepLR(SM_optim, step_size=1, gamma=0.8)  
    
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
            SM.train();
            x, labels, y_mask, num_items, index = tr_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS 
            
            # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
            num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
            num_query   = num_items[:,1].detach().numpy().flatten()
            batch_sz    = num_items.shape[0]
            
            # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
            x = x.permute(0,2,1) # bx70*20
            x_sup = Variable(torch.cat((x[:,:,:10], labels[:,:10].unsqueeze(1)), 1)).cuda(GPU) # bx71(41+29+1)*10 
            x_que = torch.zeros(batch_sz, 72, 20)
            x_que[:,:41,:10] = x[:,:41,:10].clone() # fill with x_sup_log
            x_que[:,41:70,:] = x[:,41:,:].clone()   # fill with x_sup_feat and x_que_feat
            x_que[:, 70,:10] = 1                    # support marking
            x_que[:, 71,:10] = labels[:,:10] # labels marking
            x_que = Variable(x_que).cuda(GPU) # bx29*10
    
            # y 
            y = labels.clone() # bx20
            
            # y_mask
            y_mask_que = y_mask.clone()
            y_mask_que[:,:10] = 0
            
            # Forward & update
            y_hat, att = SM(x_sup, x_que) # y_hat: b*20, att: bx10*20
            
            # Calcultate BCE loss
            loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))
            total_trloss += loss.item()
            SM.zero_grad()
            loss.backward()
            # Gradient Clipping
            #torch.nn.utils.clip_grad_norm_(SM.parameters(), 0.5)
            SM_optim.step()
            
            # Decision
            y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
            y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
            y_numpy = labels[:,10:].numpy() # bx10
            # Acc
            total_corrects += np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
            total_query += np.sum(num_query)
            
            # Restore GPU memory
            del loss, y_hat 
    
            if (session+1)%500 == 0:
                hist_trloss.append(total_trloss/900)
                hist_tracc.append(total_corrects/total_query)
                # Prepare display
                sample_att = att[0,(10-num_support[0]):10, (10-num_support[0]):(10+num_query[0])].detach().cpu().numpy()
                
                sample_sup = labels[0,(10-num_support[0]):10].long().numpy().flatten() 
                sample_que = y_numpy[0,:num_query[0]].astype(int)
                sample_pred = y_pred[0,:num_query[0]]
                sample_prob = y_prob[0,10:10+num_query[0]]

                tqdm.write(np.array2string(sample_att, formatter={'float_kind':lambda sample_att: "%.2f" % sample_att}))
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
