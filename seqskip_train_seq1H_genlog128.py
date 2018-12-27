#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:45:08 2018

seq1H_genlog: seqeunce learning model 2stack conv enc
- q(x), l(x)
- non-autoregressive (not feeding predicted labels)
- instance Norm.
- G: GLU version
- H: Highway-net version

기존 버그 완전 수정!

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
parser.add_argument("-spl","--use_suplog_as_feat", type=bool, default=True)
parser.add_argument("-pl","--use_predicted_label", type=bool, default=False)
parser.add_argument("-glu","--use_glu", type=bool, default=False)
parser.add_argument("-w","--class_num",type=int, default = 2)
parser.add_argument("-e","--epochs",type=int, default= 15)
parser.add_argument("-lr","--learning_rate", type=float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type=int, default = 2048)
parser.add_argument("-tsb","--test_batch_size", type=int, default = 1024)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
USE_SUPLOG = args.use_suplog_as_feat
USE_PRED_LABEL = args.use_predicted_label
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

#%%


def validate(mval_loader, SM, eval_mode, GPU):
    tqdm.write("Validation...")
    submit = []
    gt     = []
    total_vloss    = 0
    total_vloss_qlog = 0
    total_vloss_skip = 0
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

        # x: bx70*20
        x = x.permute(0,2,1)
        
        # Prepare ground truth log and label, y
        y_qlog = x[:,:41,:].clone() # bx41*20
        y_skip = labels.clone() #bx20
        y_mask_qlog = y_mask.unsqueeze(1).repeat(1,41,1) #bx41*20
        y_mask_skip = y_mask #bx20

        # log shift: bx41*20
        log_shift = torch.zeros(batch_sz,41,20)
        log_shift[:,:,1:] = x[:,:41,:-1]
        log_shift[:,:,11:] = 0 # DELETE LOG QUE
        
        # labels_shift: bx1*20(model can only observe past labels)
        labels_shift = torch.zeros(batch_sz,1,20)
        labels_shift[:,0,1:] = labels[:,:-1].float()
        labels_shift[:,0,11:] = 0 #!!! NOLABEL for previous QUERY
        
        # support/query state labels: bx1*20
        sq_state = torch.zeros(batch_sz,1,20)
        sq_state[:,0,:11] = 1
        
        # Pack x: bx72*20 (or bx32*20 if not using sup_logs)
        x = torch.cat((log_shift, x[:,41:,:], labels_shift, sq_state), 1).cuda(GPU) # x: bx72*20
        
        if USE_PRED_LABEL is True:
            # Predict
            li = 70 # the label's dimension indice
            _x = x[:,:,:11].clone() # bx72*11
            for q in range(11,20):
                y_hat_qlog, y_hat_skip = SM(Variable(_x, requires_grad=False)) # will be bx11 at the first round 
                # Append next features
                _x = torch.cat((_x, x[:,:,q].unsqueeze(2)), 2) # now bx72*12
                _x[:,li,q] = torch.sigmoid(y_hat_skip[:,-1]) # replace with predicted label
                _x[:,:41,q] = torch.sigmoid(y_hat_qlog[:,-1])
            y_hat_qlog, y_hat_skip = SM(Variable(_x, requires_grad=False)) # y_hat(final): bx20
            del _x
        else:
            y_hat_qlog, y_hat_skip = SM(x) # y_hat_qlog: bx41*20, y_hat_skip: b*20
            
        # Calcultate BCE loss
        loss_qlog = F.binary_cross_entropy_with_logits(input=y_hat_qlog.cuda(GPU)*y_mask_qlog.cuda(GPU),
                                                       target=y_qlog.cuda(GPU)*y_mask_qlog.cuda(GPU))
        loss_skip = F.binary_cross_entropy_with_logits(input=y_hat_skip.cuda(GPU)*y_mask_skip.cuda(GPU),
                                                       target=y_skip.cuda(GPU)*y_mask_skip.cuda(GPU))
        loss      = loss_qlog + loss_skip
        total_vloss_qlog += loss_qlog.item()
        total_vloss_skip += loss_skip.item()
        total_vloss += loss.item()
        
        # Decision
        y_prob = torch.sigmoid(y_hat_skip.detach()*y_mask_skip.cuda(GPU)).cpu().numpy() # bx20               
        y_pred = (y_prob[:,10:]>=0.5).astype(np.int) # bx10
        y_numpy = y_skip[:,10:].numpy() # bx10
        # Acc
        total_vcorrects += np.sum((y_pred==y_numpy)*y_mask_skip[:,10:].numpy())
        total_vquery += np.sum(num_query)
        
        # Restore GPU memory
        del loss, loss_qlog, loss_skip, y_hat_qlog, y_hat_skip
            
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
            tqdm.write("val_session:{0:}  vloss(qlog|skip):{1:.6f}({2:.6f}|{3:.6f})  vacc:{4:.4f}".format(val_session,
                       total_vloss/total_vquery, total_vloss_qlog/total_vquery, 
                       total_vloss_skip/total_vquery, total_vcorrects/total_vquery))
        
    # Avg.Acc (skip labels only, log-generation acc is not implemented yet!)
    if eval_mode==1:
        aacc = evaluate(submit, gt)
        tqdm.write("AACC={0:.6f}, FirstAcc={1:.6f}".format(aacc[0], aacc[1]))    
        
    hist_vloss.append(total_vloss/total_vquery)
    hist_vloss_qlog.append(total_vloss_qlog/total_vquery)
    hist_vloss_skip.append(total_vloss_skip/total_vquery)
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
                                      data_sel=(99965071, 101065071),#104965071),#(99965071, 124950714), # 20%를 테스트
                                      batch_size=TS_BATCH_SZ,
                                      shuffle=False,
                                      seq_mode=True) 
    
    # Init neural net
    SM = SeqModel().cuda(GPU)
    SM_optim = torch.optim.Adam(SM.parameters(), lr=LEARNING_RATE)
    SM_scheduler = StepLR(SM_optim, step_size=1, gamma=0.7)  
    
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
        total_trloss_qlog = 0
        total_trloss_skip = 0
        total_trloss   = 0
        for session in trange(len(tr_sessions_iter), desc='sessions', position=1, ascii=True):
            SM.train();
            x, labels, y_mask, num_items, index = tr_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS
            
            # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
            num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
            num_query   = num_items[:,1].detach().numpy().flatten()
            batch_sz    = num_items.shape[0]
    
            # x: bx70*20
            x = x.permute(0,2,1)
            
            # Prepare ground truth log and label, y
            y_qlog = x[:,:41,:].clone() # bx41*20
            y_skip = labels.clone() #bx20
            y_mask_qlog = y_mask.unsqueeze(1).repeat(1,41,1) #bx41*20
            y_mask_skip = y_mask #bx20
    
            # log shift: bx41*20
            log_shift = torch.zeros(batch_sz,41,20)
            log_shift[:,:,1:] = x[:,:41,:-1]
            log_shift[:,:,11:] = 0 # DELETE LOG QUE
            
            # labels_shift: bx1*20(model can only observe past labels)
            labels_shift = torch.zeros(batch_sz,1,20)
            labels_shift[:,0,1:] = labels[:,:-1].float()
            labels_shift[:,0,11:] = 0 #!!! NOLABEL for previous QUERY
            
            # support/query state labels: bx1*20
            sq_state = torch.zeros(batch_sz,1,20)
            sq_state[:,0,:11] = 1
            
            # Pack x: bx72*20 (or bx32*20 if not using sup_logs)
            x = Variable(torch.cat((log_shift, x[:,41:,:], labels_shift, sq_state), 1)).cuda(GPU) # x: bx72*20
  
            # Forward & update
            y_hat_qlog, y_hat_skip = SM(x) # y_hat: b*20
            
            # Calcultate BCE loss
            loss_qlog = F.binary_cross_entropy_with_logits(input=y_hat_qlog.cuda(GPU)*y_mask_qlog.cuda(GPU),
                                                           target=y_qlog.cuda(GPU)*y_mask_qlog.cuda(GPU))
            loss_skip = F.binary_cross_entropy_with_logits(input=y_hat_skip.cuda(GPU)*y_mask_skip.cuda(GPU),
                                                           target=y_skip.cuda(GPU)*y_mask_skip.cuda(GPU))
            loss      = loss_qlog + loss_skip
            total_trloss_qlog += loss_qlog.item()
            total_trloss_skip += loss_skip.item()
            total_trloss += loss.item()
            SM.zero_grad()
            loss.backward()
            # Gradient Clipping
            #torch.nn.utils.clip_grad_norm_(SM.parameters(), 0.5)
            SM_optim.step()
            
            # Decision
            y_prob = torch.sigmoid(y_hat_skip.detach()*y_mask_skip.cuda(GPU)).cpu().numpy() # bx20               
            y_pred = (y_prob[:,10:]>=0.5).astype(np.int) # bx10
            y_numpy = y_skip[:,10:].numpy() # bx10
            
            # Label Acc*
            total_corrects += np.sum((y_pred==y_numpy)*y_mask_skip[:,10:].numpy())
            total_query += np.sum(num_query)
#            # Log generation Acc*
#            y_qlog_mask = y_mask[:,:41,10:]
            
            # Restore GPU memory
            del loss, loss_qlog, loss_skip, y_hat_qlog, y_hat_skip 
    
            if (session+1)%500 == 0:
                hist_trloss_qlog.append(total_trloss_qlog/500) #!
                hist_trloss_skip.append(total_trloss_skip/500) #!
                hist_trloss.append(total_trloss/500)
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
                tqdm.write("tr_session:{0:}  tr_loss(qlog|skip):{1:.6f}({2:.6f}|{3:.6f})  tr_acc:{4:.4f}".format(session,
                           hist_trloss[-1], hist_trloss_qlog[-1], hist_trloss_skip[-1], hist_tracc[-1]))
                total_corrects = 0
                total_query    = 0
                total_trloss   = 0
                total_trloss_qlog   = 0
                total_trloss_skip   = 0
            
            if (session+1)%8000 == 0:
                 # Validation
                 validate(mval_loader, SM, eval_mode=True, GPU=GPU)
                 # Save
                 torch.save({'ep': epoch, 'sess':session, 'SM_state': SM.state_dict(),'loss': hist_trloss[-1], 
                             'hist_trloss_qlog': hist_trloss_qlog, 'hist_trloss_skip': hist_trloss_skip,  'hist_vacc': hist_vacc,
                             'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'SM_opt_state': SM_optim.state_dict(),
                             'SM_sch_state': SM_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        # Validation
        validate(mval_loader, SM, eval_mode=True, GPU=GPU)
        # Save
        torch.save({'ep': epoch, 'sess':session, 'SM_state': SM.state_dict(),'loss': hist_trloss[-1],
                    'hist_trloss_qlog': hist_trloss_qlog, 'hist_trloss_skip': hist_trloss_skip,  'hist_vacc': hist_vacc,
                    'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'SM_opt_state': SM_optim.state_dict(),
                    'SM_sch_state': SM_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        SM_scheduler.step()
    
if __name__ == '__main__':
    main()  
