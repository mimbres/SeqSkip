#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:44:49 2018

GenLog1H_Skip1eH: predict Skip labels by using que-logs generated with pre-trained seq1H_genlog model. 

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
import glob, os, time
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader import SpotifyDataloader
from utils.eval import evaluate
from blocks.highway_glu_dil_conv_v2 import HighwayDCBlock
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type=str, default="./config_init_dataset.json")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_GenLog1H_Skip1eH/")
parser.add_argument("-gf","--load_generator_fpath",type=str, default="./save/exp_seq1H_genlog128/check_14_48811.pth")
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
GENERATOR_LOAD_PATH = args.load_generator_fpath 
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

     
# Generator
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
###   
    
class SeqEncoder2(nn.Module):
    def __init__(self, input_ch, e_ch,
                 h_k_szs=[2,2,2,3,1,1], #h_k_szs=[2,2,5,1,1],
                 h_dils=[1,2,4,8,1,1],
                 causality=True,
                 use_glu=False):
        super(SeqEncoder2, self).__init__()
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

# Model for predicting skip labels
class SeqModel2(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, e_ch=128, d_ch=128, use_glu=USE_GLU):
        super(SeqModel2, self).__init__()
        self.e_ch = e_ch
        self.d_ch = d_ch
        self.enc = SeqEncoder2(input_ch=input_dim, e_ch=e_ch,
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
    
    


#%%


def validate(mval_loader, SM, SMG, eval_mode, GPU):
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

        # x: bx70*20
        x = x.permute(0,2,1)
        
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
        x_1 = Variable(torch.cat((log_shift, x[:,41:,:], labels_shift, sq_state), 1)).cuda(GPU) # x: bx72*20
  
        # Pre-trained Generator: forward & get qlog^ 
        y_hat_qlog, _ = SMG(x_1) # y_hat: b*20
        x_feat_T = torch.zeros(batch_sz, 72, 20)
        x_feat_T[:,:70,:] = x.clone()
        x_feat_T[:, 70,:10] = 1 # Sup/Que state indicator  
        x_feat_T[:, 71,:10] = labels[:,:10].clone()
                    
        x_feat_S = x_feat_T.clone()
        x_feat_S[:, :41, 10:] = y_hat_qlog[:,:,10:].clone() # remove que-log
        x_feat_S = Variable(x_feat_S).cuda(GPU)
        del y_hat_qlog, x_1
        # y 
        y = labels.clone() # bx20
        
        # y_mask
        y_mask_que = y_mask.clone()
        y_mask_que[:,:10] = 0
        
        y_hat = SM(x_feat_S)
        
        # Calcultate BCE loss
        loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))
        total_vloss += loss.item()
        
        # Decision
        y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
        y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
        y_numpy = labels[:,10:].numpy() # bx10
        # Acc
        total_vcorrects += np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
        total_vquery += np.sum(num_query)
        
        
        # Restore GPU memory
        del loss, y_hat
            
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
            tqdm.write("val_session:{0:}  vloss:{1:.6f}   vacc:{2:.4f}".format(val_session,
                       total_vloss/total_vquery, total_vcorrects/total_vquery))
        
    # Avg.Acc (skip labels only, log-generation acc is not implemented yet!)
    if eval_mode==1:
        aacc = evaluate(submit, gt)
        tqdm.write("AACC={0:.6f}, FirstAcc={1:.6f}".format(aacc[0], aacc[1]))    
        
    hist_vloss.append(total_vloss/total_vquery)
    hist_vacc.append(total_vcorrects/total_vquery)
    return submit

def save_submission(output, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path,"w") as f:
        for l in tqdm(output, ascii=True):
            line = ''.join(map(str,l))
            f.write(line + '\n')
    tqdm.write('submission saved to {}'.format(output_path))
    return


# Main
def main():  
    # Trainset stats: 2072002577 items from 124950714 sessions
    print('Initializing dataloader...')
#    mtrain_loader = SpotifyDataloader(config_fpath=args.config,
#                                      mtrain_mode=True,
#                                      data_sel=(0, 124000000), # 80% 트레인
#                                      batch_size=TR_BATCH_SZ,
#                                      shuffle=True,
#                                      seq_mode=True) # seq_mode implemented  
#    
#    mval_loader  = SpotifyDataloader(config_fpath=args.config,
#                                      mtrain_mode=True, # True, because we use part of trainset as testset
#                                      data_sel=(124000000, 124950714),#104965071),#(99965071, 124950714), # 20%를 테스트
#                                      batch_size=TS_BATCH_SZ,
#                                      shuffle=False,
#                                      seq_mode=True) 

    mtest_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=False, # False = testset for submission
                                  #data_sel=(0, 100),
                                  batch_size=2048,
                                  shuffle=False,
                                  seq_mode=True) 


    # Load Generator net
    SMG = SeqModel().cuda(GPU) 
    checkpoint = torch.load(GENERATOR_LOAD_PATH, map_location='cuda:{}'.format(GPU))
    tqdm.write("Loading saved teacher model from '{0:}'... loss: {1:.6f}".format(GENERATOR_LOAD_PATH,checkpoint['loss']))
    SMG.load_state_dict(checkpoint['SM_state'])
    SMG.cuda(GPU).eval();
    
    # Init neural net
    SM = SeqModel2().cuda(GPU)
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
        

    # Validation
    submission = validate(mtest_loader, SM, SMG, eval_mode=2, GPU=GPU)
    if len(submission)!=31251398: print("WARNING: submission size not matches.");

    # Save
    fpath = MODEL_SAVE_PATH+ "/submission_{}.txt".format(time.strftime('%Y%m%d_%Hh%Mm'))
    tqdm.write("Saving...")
    save_submission(submission, fpath)
    tqdm.write("Succesfully saved submission file: {}".format(fpath) )

    
if __name__ == '__main__':
    main()  
