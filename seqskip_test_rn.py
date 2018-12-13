#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 04:24:24 2018

@author: mimbres
"""

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
import glob, os, time
import argparse
from tqdm import trange, tqdm 
from spotify_data_loader import SpotifyDataloader

parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type = str, default = "./config_init_dataset.json")
parser.add_argument("-s","--save_path",type = str, default = "./save/exp1/") # directory of saved checkpoint
parser.add_argument("-l","--load_file",type = str, default = None)
parser.add_argument("-z","--submission_out_path",type = str, default="./submissions/")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-t","--test_episode", type = int, default = 0)
parser.add_argument("-g","--gpu",type=int, default=0)
#parser.add_argument("-e","--embed_hidden_unit",type=int, default=2)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
TEST_EPISODE = args.test_episode
GPU = args.gpu

# Checkpoint directory
MODEL_SAVE_PATH = args.save_path

# Submission output
SUBMISSION_OUTPUT_PATH = args.submission_out_path

# TSSET stats: 518260758 items within 31251398 sessions
print('Initializing dataloader...')
mtest_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=False, # False = testset for submission
                                  #data_sel=(0, 100),
                                  batch_size=1,
                                  shuffle=False) 


#mtest_loader  = SpotifyDataloader(config_fpath=args.config,
#                                  mtrain_mode=True, # True, because we use part of trainset as testset
#                                  data_sel=(99965071, 110075071),#(99965071, 124950714), # 20%를 테스트
#                                  batch_size=1,
#                                  shuffle=True) 


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
                        nn.Linear(input_sz, 512), # 56x1x172 -> 56x1x512
                        nn.LayerNorm(512),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.LayerNorm(256),
                        nn.ReLU())
        self.fc1 = nn.Linear(256,10)
        self.fc2 = nn.Linear(10,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out
    

# Init neural net
FeatEnc = MLP(input_sz=29, hidden_sz=512, output_sz=64).apply(weights_init).cuda(GPU)
RN      = RelationNetwork(input_sz=172).apply(weights_init).cuda(GPU)


#%%
def test():
    tqdm.write("Test...")
    test_sessions_iter = iter(mtest_loader)
    submission = [] 
        
    for test_session in trange(len(test_sessions_iter), desc='test-sessions', position=0):
        FeatEnc.eval(); RN.eval();
        feats, logs, labels, num_items, index = test_sessions_iter.next()
        feats, logs, labels = Variable(feats).cuda(GPU), Variable(logs).cuda(GPU), Variable(labels).cuda(GPU)
        num_support = int(num_items/2) # If num_items was odd number, query has one more item. 
        num_query   = int(num_items) - num_support
        
        x_support = feats[:, :num_support, :].permute(1,0,2) # 7x1x29
        x_query   = feats[:, num_support:num_items, :].permute(1,0,2) # 8x1*29 (batch x ch x dim)
        x_feat_support = FeatEnc(x_support) # 7x1x64
        x_feat_query   = FeatEnc(x_query)   # 8x1x64
        # - concat support los(d=41) and labels(d=3) to feat_support: QUERY SHOULD NOT INCLUDE THESE...
        _extras = torch.cat((logs[:, :num_support, :],labels[:, :num_support, :]), 2)
        x_feat_support = torch.cat((x_feat_support, _extras.view(-1,1,44)), 2) #7x1x108
        x_feat_support_ext = x_feat_support.unsqueeze(0).repeat(num_query,1,1,1) # 8x7x1*108
        x_feat_query_ext   = x_feat_query.unsqueeze(0).repeat(num_support,1,1,1) # 7x8x1*64
        x_feat_query_ext = torch.transpose(x_feat_query_ext,0,1) # 8x7x1*64
        x_feat_relation_pairs = torch.cat((x_feat_support_ext, x_feat_query_ext),3) # 8x7x1*172
        x_feat_relation_pairs = x_feat_relation_pairs.view(num_support*num_query, 1, -1) # 56x1*172
        
        y_support_ext = labels[:, :num_support, 1].view(-1).repeat(num_query) # [56]
        y_query_ext   = labels[:, num_support:num_items, 1].repeat(num_support,1) 
        y_query_ext   = torch.transpose(y_query_ext,0,1).contiguous().view(-1) # [56]
        y_relation = (y_support_ext==y_query_ext).float().view(-1,1)  # 56x1
        y_hat_relation = RN(x_feat_relation_pairs) # 56x1
        
        sim_score = torch.FloatTensor(np.zeros((num_support*num_query,2))).detach().cpu()
        sim_score[:,0] = y_hat_relation.view(-1) * (y_support_ext == 0).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 1).float()
        sim_score[:,1] = y_hat_relation.view(-1) * (y_support_ext == 1).float() + (1 - y_hat_relation.view(-1)) * (y_support_ext == 0).float()
        sim_score = sim_score.view(num_query,-1,2) #8x7x2 (que x sup x class)
        
        # Generate Prediction from sim_score... 
        y_pred = torch.argmax(sim_score.sum(1),1).detach().long().cpu().numpy()

        # Append to submission
        submission.append(y_pred)
        
        if (test_session+1)%4000 == 0:
            tqdm.write(np.array2string(sim_score.detach().cpu().numpy(), suppress_small=True))
            tqdm.write("S:" + np.array2string(labels[:, :num_support, 1].detach().cpu().long().numpy().flatten()) + '\n'+
                       "P:" + np.array2string(y_pred) + "\t"+"test_session:{0:}".format(test_session))  
            
            ###########################
#            y_query = labels[:, num_support:num_items, 1].detach().cpu().long().numpy() # 8
#            tqdm.write("S:" + np.array2string(labels[:, :num_support, 1].detach().cpu().long().numpy().flatten()) +'\n'+
#                       "Q:" + np.array2string(y_query.flatten()) + '\n' +
#                       "P:" + np.array2string(y_pred.flatten()) + '\t index={}'.format(int(index.detach())))
#            tqdm.write(np.array2string(sim_score.sum(1).cpu().detach().numpy()))
            
    return submission


def save_submission(output, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path,"w") as f:
        for l in output:
            line = ''.join(map(str,l))
            f.write(line + '\n')
    tqdm.write('submission saved to {}'.format(output_path))


#%% Main
def main():
    if args.load_file is None:
        fpath = max(glob.iglob(MODEL_SAVE_PATH + "check*.pth"),key=os.path.getctime) # search latest fpath
        tqdm.write("Checkpoint is not specified. The latest checkpoint will be loaded...") 
    else:
        fpath = args.load_file
    checkpoint = torch.load(fpath)
    tqdm.write("Loading checkpoint from '{0:}'... loss: {1:.6f}".format(fpath, checkpoint['loss']))    
    FeatEnc.load_state_dict(checkpoint['FE_state'])
    RN.load_state_dict(checkpoint['RN_state'])
    
    # test
    submission = test()
    if len(submission)!=31251398: print("WARNING: submission size not matches.");
    print(len(submission))
    # save
    fpath = SUBMISSION_OUTPUT_PATH + "submission_{}.txt".format(time.strftime('%Y%m%d_%Hh%Mm'))
    save_submission(submission, fpath)
    
    return


if __name__ == '__main__':
    main()