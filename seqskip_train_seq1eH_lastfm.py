#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:45:08 2018

seq1H: seqeunce learning model ONLY 1 Enc (256)
- q(x), l(x)
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
from blocks.highway_dil_conv import HighwayDCBlock
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type=str, default="./config_init_dataset.json")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_seq1eH_lastfm/")
parser.add_argument("-l","--load_continue_latest",type=str, default=None)
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
USE_PRED_LABEL = args.use_predicted_label
USE_GLU    = args.use_glu
INPUT_DIM = 70 + 2 + 50

CLASS_NUM = args.class_num
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
TR_BATCH_SZ = args.train_batch_size
TS_BATCH_SZ = args.test_batch_size
GPU = args.gpu

# Model-save directory
LFM_CHECKPOINT_PATH = "./LastFM/exp_lastfm_mlp_reg1/check_1200.pth"
MODEL_SAVE_PATH = args.save_path
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


hist_trloss = list()
hist_tracc  = list()
hist_vloss  = list()
hist_vacc   = list()
np.set_printoptions(precision=3)

        
class MLP_Regressor(nn.Module):
    def __init__(self, input_d=29, output_d=50,
                 emb_d=128, mid_d=256): # emb_dim means the l-1 layer's dimension
        super(MLP_Regressor, self).__init__()
        self.embedding_layer  = nn.Sequential(nn.Linear(input_d, mid_d),
                                              nn.ReLU(),
                                              nn.Linear(mid_d, emb_d))
        self.final_layer = nn.Linear(emb_d, output_d)
        
    def forward(self, x): # Input:
        emb = self.embedding_layer(x)
        x = self.final_layer(emb)
        return emb, x



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
                                        nn.Conv1d(e_ch,1,1))

    def forward(self, x): # Input:bx256*20
        x = self.front_1x1(x) # bx128*20
        x = self.h_block(x)   # bx128*20
        #x = self.last_1x1(x)  # bx64*20
        return self.classifier(x).squeeze(1) # bx20
        

class SeqModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, e_ch=256, d_ch=256, use_glu=USE_GLU):
        super(SeqModel, self).__init__()
        #self.enc = SeqFeatEnc(input_dim=input_dim, e_ch=e_ch, d_ch=d_ch, use_glu=use_glu)
        #self.clf = SeqClassifier(input_ch=d_ch, e_ch=e_ch, use_glu=use_glu)
        self.clf = SeqClassifier(input_ch=input_dim, e_ch=e_ch, use_glu=use_glu)
        
    def forward(self, x):
        #x = self.enc(x)
        return self.clf(x)

#%%


def validate(mval_loader, SM, LFM_model, eval_mode):
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

        x[:,10:,:41] = 0 # DELETE METALOG QUE
        labels_shift = torch.zeros(batch_sz,20,1)
        labels_shift[:,1:,0] = labels[:,:-1].float()
        labels_shift[:,11:,0] = 0 # REMOVE QUERY LABELS!
        sq_state = torch.zeros(batch_sz,20,1)
        sq_state[:,:11,0] = 1
        
        x_audio = x[:,:,41:].data.clone()
        x_audio = Variable(x_audio, requires_grad=False).cuda()
        x_emb_lastfm, x_lastfm = LFM_model(x_audio)
        x_lastfm = x_lastfm.cpu()
        del x_emb_lastfm
        # x: bx122*20
        
        x = torch.cat((x_lastfm, x, labels_shift, sq_state), dim=2).permute(0,2,1).cuda(GPU)

        y_hat = SM(x)
        # Calcultate BCE loss
        loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask.cuda(GPU), target=labels.cuda(GPU)*y_mask.cuda(GPU))
        total_vloss += loss.item()
        
        # Decision
        y_prob = torch.sigmoid(y_hat*y_mask.cuda(GPU)).detach().cpu().numpy() # bx20               
        y_pred = (y_prob[:,10:]>=0.5).astype(np.int) # bx10
        y_numpy = labels[:,10:].numpy() # bx10
        # Acc
        y_query_mask = y_mask[:,10:].numpy()
        total_vcorrects += np.sum((y_pred==y_numpy)*y_query_mask)
        total_vquery += np.sum(num_query)
        
        
        # Eval, Submission
        if eval_mode is not 0:
            for b in np.arange(batch_sz):
                submit.append(y_pred[b,:num_query[b]].flatten())
                gt.append(y_numpy[b,:num_query[b]].flatten())
                
        if (val_session+1)%400 == 0:
            sample_sup = labels[0,:num_support[0]].long().numpy().flatten() 
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
    SM_scheduler = StepLR(SM_optim, step_size=1, gamma=0.7)  
    
    LFM_model = MLP_Regressor().cuda(GPU)
    LFM_checkpoint = torch.load(LFM_CHECKPOINT_PATH, map_location='cuda:{}'.format(GPU))
    LFM_model.load_state_dict(LFM_checkpoint['model_state'])
    
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
            x[:,10:,:41] = 0 # DELETE METALOG QUE

            # labels_shift: (model can only observe past labels)
            labels_shift = torch.zeros(batch_sz,20,1)
            labels_shift[:,1:,0] = labels[:,:-1].float()
            #!!! NOLABEL for previous QUERY
            labels_shift[:,11:,0] = 0
            # support/query state labels
            sq_state = torch.zeros(batch_sz,20,1)
            sq_state[:,:11,0] = 1
            # compute lastfm_output
            x_audio = x[:,:,41:].data.clone()
            x_audio = Variable(x_audio, requires_grad=False).cuda()
            x_emb_lastfm, x_lastfm = LFM_model(x_audio)
            x_lastfm = x_lastfm.cpu()
            del x_emb_lastfm
        
            # Pack x: bx122*20
            x = Variable(torch.cat((x_lastfm, x, labels_shift, sq_state), dim=2).permute(0,2,1)).cuda(GPU)
             
  
            # Forward & update
            y_hat = SM(x) # y_hat: b*20
            # Calcultate BCE loss
            loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask.cuda(GPU), target=labels.cuda(GPU)*y_mask.cuda(GPU))
            total_trloss += loss.item()
            SM.zero_grad()
            loss.backward()
            # Gradient Clipping
            #torch.nn.utils.clip_grad_norm_(SM.parameters(), 0.5)
            SM_optim.step()
            
            # Decision
            y_prob = torch.sigmoid(y_hat*y_mask.cuda(GPU)).detach().cpu().numpy() # bx20               
            y_pred = (y_prob[:,10:]>=0.5).astype(np.int) # bx10
            y_numpy = labels[:,10:].numpy() # bx10
            # Acc
            y_query_mask = y_mask[:,10:].numpy()
            total_corrects += np.sum((y_pred==y_numpy)*y_query_mask)
            total_query += np.sum(num_query)
            # Restore GPU memory
            del loss, y_hat 
    
            if (session+1)%500 == 0:
                hist_trloss.append(total_trloss/900)
                hist_tracc.append(total_corrects/total_query)
                # Prepare display
                sample_sup = labels[0,:num_support[0]].long().numpy().flatten() 
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
                
            
            if (session+1)%20000 == 0:
                 # Validation
                 validate(mval_loader, SM, eval_mode=True)
                 # Save
                 torch.save({'ep': epoch, 'sess':session, 'SM_state': SM.state_dict(),'loss': hist_trloss[-1], 'hist_vacc': hist_vacc,
                             'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'SM_opt_state': SM_optim.state_dict(),
                             'SM_sch_state': SM_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        # Validation
        validate(mval_loader, SM, LFM_model, eval_mode=True)
        # Save
        torch.save({'ep': epoch, 'sess':session, 'SM_state': SM.state_dict(),'loss': hist_trloss[-1], 'hist_vacc': hist_vacc,
                    'hist_vloss': hist_vloss, 'hist_trloss': hist_trloss, 'SM_opt_state': SM_optim.state_dict(),
                    'SM_sch_state': SM_scheduler.state_dict()}, MODEL_SAVE_PATH + "check_{0:}_{1:}.pth".format(epoch, session))
        SM_scheduler.step()
    
if __name__ == '__main__':
    main()  
