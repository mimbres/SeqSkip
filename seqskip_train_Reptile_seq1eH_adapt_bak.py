#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 13:44:03 2018

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from torch.optim.lr_scheduler import StepLR
from torch.backends import cudnn
import numpy as np
import glob, os
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader_v2 import SpotifyDataloader
from utils.eval import evaluate
from blocks.highway_glu_dil_conv_v2 import HighwayDCBlock
#from blocks.multihead_attention import MultiHeadAttention
cudnn.benchmark = True


parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type=str, default="./config_init_dataset.json")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_Reptile_seeq1eH_adapt/")
parser.add_argument("-l","--load_continue_latest",type=str, default=None)
parser.add_argument("-glu","--use_glu", type=bool, default=False)
parser.add_argument("-w","--class_num",type=int, default = 2)
parser.add_argument("-e","--epochs",type = int, default= 10)
parser.add_argument("-i","--iterations",type=int, default= 5) # K inner-iteration
parser.add_argument("-lr","--learning_rate", type=float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type=int, default = 2048)
parser.add_argument("-tsb","--test_batch_size", type=int, default = 1024)
parser.add_argument("-disp","--validate_every", type=int, default=100)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
USE_GLU    = args.use_glu
INPUT_DIM = 74 

EPOCHS = args.epochs
CLASS_NUM = args.class_num
ITERATIONS = args.iterations
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
                 h_k_szs, h_dils,
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
        self.classifier = nn.Conv1d(e_ch,1,1)
        
    def forward(self, x):
        x = self.enc(x) # bx128*10 
        return self.classifier(x).squeeze(1) # bx20
        
    
    def clone(self):
        clone = SeqModel()
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda(GPU)
        return clone


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def get_loss(input, target):
    return F.binary_cross_entropy_with_logits(input=input, target=target)


def do_base_learning_adam(model, train_iter, lr_inner, n_inner, state=None):
    new_model = SeqModel()
    new_model.load_state_dict(model.state_dict())  # copy? looks okay
    inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr_inner)
    if state is not None:
        inner_optimizer.load_state_dict(state)

    # K steps of gradient descent
    for i in range(n_inner):

        x, labels, y_mask, num_items, index = train_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS 
            
        # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
        batch_sz    = num_items.shape[0]
        
        # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
        x = x.permute(0,2,1) # bx70*20
        
        x_feat = torch.zeros(batch_sz, 74, 20)
        x_feat[:,:70,:] = x.clone()
        x_feat[:,:41,10:] = 0
        x_feat[:, 70,:10] = 1  
        x_feat[:, 71:74,:10] = labels[:,:10,:].permute(0,2,1).clone()
        x_feat = Variable(x_feat).cuda(GPU)
        
        # y
        y = labels[:,:,1].clone()
        
        # y_mask
        y_mask_que = y_mask.clone()
        y_mask_que[:,:10] = 0
        
        # Forward & update
        y_hat = new_model(x_feat) # y_hat: b*20
        
        # Calcultate BCE loss
        loss = get_loss(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))

        # Backward pass - Update fast net
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        
    return new_model, inner_optimizer.state_dict()

def do_base_eval(new_model, val_iter):
    x, labels, y_mask, num_items, index = val_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS 
        
    # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
    batch_sz    = num_items.shape[0]
    num_query   = num_items[:,1].detach().numpy().flatten()
    
    # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
    x = x.permute(0,2,1) # bx70*20
    
    x_feat = torch.zeros(batch_sz, 74, 20)
    x_feat[:,:70,:] = x.clone()
    x_feat[:,:41,10:] = 0
    x_feat[:, 70,:10] = 1  
    x_feat[:, 71:74,:10] = labels[:,:10,:].permute(0,2,1).clone()
    x_feat = Variable(x_feat).cuda(GPU)
    
    # y
    y = labels[:,:,1].clone()
    
    # y_mask
    y_mask_que = y_mask.clone()
    y_mask_que[:,:10] = 0
    
    # Forward & update
    y_hat = new_model(x_feat) # y_hat: b*20
    
    # Calcultate BCE loss
    loss = get_loss(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))
    
    # Decision
    y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
    y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
    y_numpy = labels[:,10:,1].numpy() # bx10
    # Acc
    n_corrects = np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
    n_query = np.sum(num_query)
    accuracy = n_corrects/n_query
    tqdm.write("base_eval_acc:{0:.2f}".format(accuracy))    
    return loss.item()


def do_evaluation(net, test_iter, iterations):
    losses = []
    accuracies = []
    net.eval()
    for iteration in np.arange(iterations):
        x, labels, y_mask, num_items, index = test_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS 
            
        # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
        #num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
        num_query   = num_items[:,1].detach().numpy().flatten()
        batch_sz    = num_items.shape[0]
        
        # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
        x = x.permute(0,2,1) # bx70*20
        
        x_feat = torch.zeros(batch_sz, 74, 20)
        x_feat[:,:70,:] = x.clone()
        x_feat[:,:41,10:] = 0
        x_feat[:, 70,:10] = 1  
        x_feat[:, 71:74,:10] = labels[:,:10,:].permute(0,2,1).clone()
        x_feat = Variable(x_feat).cuda(GPU)
        
        # y
        y = labels[:,:,1].clone()
        
        # y_mask
        y_mask_que = y_mask.clone()
        y_mask_que[:,:10] = 0
        
        # Forward & update
        y_hat = net(x_feat) # y_hat: b*20
        
        # Calcultate BCE loss
        loss = get_loss(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))

        # Decision
        y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
        y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
        y_numpy = labels[:,10:,1].numpy() # bx10
        # Acc
        n_corrects = np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
        n_query = np.sum(num_query)
        accuracy = n_corrects/n_query
        losses.append(loss.item())
        accuracies.append(accuracy)

    return np.mean(losses), np.mean(accuracies)

#%%



    

# Main
def main():  
    # Trainset stats: 2072002577 items from 124950714 sessions
    print('Initializing dataloader...')
    mtrain_loader = SpotifyDataloader(config_fpath=args.config,
                                      mtrain_mode=True,
                                      data_sel=(0, 124050714), # 80% 트레인
                                      batch_size=TR_BATCH_SZ,
                                      shuffle=True,
                                      seq_mode=True) # seq_mode implemented  
    
    
    
    mval_loader  = SpotifyDataloader(config_fpath=args.config,
                                      mtrain_mode=True, # True, because we use part of trainset as testset
                                      data_sel=(124050714, 124950714),#(99965071, 124950714), # 20%를 테스트
                                      batch_size=TS_BATCH_SZ,
                                      shuffle=False,
                                      seq_mode=True) 
    
    # Build model, optimizer, and set states
    meta_net = SeqModel().cuda(GPU)
    meta_optimizer = torch.optim.Adan(meta_net.parameters(), lr=LEARNING_RATE)
    train_metalosses =[]
    test_metalosses = []
    
    inner_optimizer_state = None
    lr_inner = 0.001
    lr_outer = 0.001
    n_inner  = 5
    
    # Load checkpoint
    if args.load_continue_latest is None:
        START_EPOCH = 0        
    else:
        latest_fpath = max(glob.iglob(MODEL_SAVE_PATH + "check*.pth"),key=os.path.getctime)  
        checkpoint = torch.load(latest_fpath, map_location='cuda:{}'.format(GPU))
        tqdm.write("Loading saved model from '{0:}'... loss: {1:.6f}".format(latest_fpath,checkpoint['loss']))
        meta_net.load_state_dict(checkpoint['SM_state'])
        meta_optimizer.load_state_dict(checkpoint['SM_opt_state'])
        START_EPOCH = checkpoint['ep']
    
    
    
    # Main loop 
    for epoch in trange(START_EPOCH, EPOCHS, desc='epochs', position=0, ascii=True):
        tqdm.write('Train...')
        tr_sessions_iter = iter(mtrain_loader)
        val_sessions_iter = iter(mval_loader)
        total_corrects = 0
        total_query    = 0
        total_trloss   = 0

        
        for session in trange(len(tr_sessions_iter), desc='sessions', position=1, ascii=True):
            
            
            
            # Train Inner loop: Take k gradient steps
            new_model, inner_optimizer_state = do_base_learning_adam(meta_net, tr_sessions_iter, lr_inner, n_inner, inner_optimizer_state)
            
            # Meta-learn:
            train_metaloss = do_base_eval(new_model, tr_sessions_iter) 
            
            # Inject updates into each .grad
            for p, new_p in zip(meta_net.parameters(), new_model.parameters()):
                if p.grad is None:
                    p.grad = Variable(torch.zeros(p.size()).cuda(GPU))
                p.grad.data.add_(p.data - new_p.data)
            
            # Update meta-parameters
            meta_optimizer.step()
            meta_optimizer.zero_grad()
            
            ############# Validation
            new_model = do_base_learning(model, wave, lr_inner, n_inner)
            test_metaloss = do_base_eval(new_model, wave)
            
            
            
            
            
            x, labels, y_mask, num_items, index = tr_sessions_iter.next() # FIXED 13.Dec. SEPARATE LOGS. QUERY SHOULT NOT INCLUDE LOGS 
            
            # Sample data for 'support' and 'query': ex) 15 items = 7 sup, 8 queries...        
            num_support = num_items[:,0].detach().numpy().flatten() # If num_items was odd number, query has one more item. 
            num_query   = num_items[:,1].detach().numpy().flatten()
            batch_sz    = num_items.shape[0]
            
            # x: the first 10 items out of 20 are support items left-padded with zeros. The last 10 are queries right-padded.
            x = x.permute(0,2,1) # bx70*20
            
            x_feat = torch.zeros(batch_sz, 74, 20)
            x_feat[:,:70,:] = x.clone()
            x_feat[:,:41,10:] = 0
            x_feat[:, 70,:10] = 1  
            x_feat[:, 71:74,:10] = labels[:,:10,:].permute(0,2,1).clone()
            x_feat = Variable(x_feat).cuda(GPU)
            
            # y
            y = labels[:,:,1].clone()
            
            # y_mask
            y_mask_que = y_mask.clone()
            y_mask_que[:,:10] = 0
            
            # Forward & update
            y_hat = SM(x_feat) # y_hat: b*20
            
            # Calcultate BCE loss
            loss = F.binary_cross_entropy_with_logits(input=y_hat*y_mask_que.cuda(GPU), target=y.cuda(GPU)*y_mask_que.cuda(GPU))
            total_trloss += loss.item()
            SM.zero_grad()
            loss.backward()
            SM_optim.step()
            
            # Decision
            y_prob = torch.sigmoid(y_hat*y_mask_que.cuda(GPU)).detach().cpu().numpy() # bx20               
            y_pred = (y_prob[:,10:]>0.5).astype(np.int) # bx10
            y_numpy = labels[:,10:,1].numpy() # bx10
            # Acc
            total_corrects += np.sum((y_pred==y_numpy)*y_mask_que[:,10:].numpy())
            total_query += np.sum(num_query)
            
            # Restore GPU memory
            del loss, y_hat 
    
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
                
            

    
if __name__ == '__main__':
    main()  
