#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 04:24:24 2018

@author: mimbres
"""
import torch
#import numpy as np
import os, time, sys
import argparse
from tqdm import tqdm 
from spotify_data_loader import SpotifyDataloader

parser = argparse.ArgumentParser(description="Sequence Skip Prediction")
parser.add_argument("-c","--config",type = str, default = "./config_init_dataset.json")
parser.add_argument("-m","--model_py", type = str, default="./seqskip_train_rnbc1_2048.py")
parser.add_argument("-s","--save_path",type = str, default="./save/exp_rnbc1_2048/check_6_39999.pth") # directory of saved checkpoint
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

sys.argv = [sys.argv[0]]

GPU = args.gpu
# Module (path of train code including validate() and model) 
MODEL_PATH = args.model_py
# Checkpoint directory
CHECKPOINT_PATH = args.save_path
# Submission output
SUBMISSION_OUTPUT_PATH = os.path.dirname(CHECKPOINT_PATH)

# TSSET stats: 518275860 items within 31251398 sessions
print('Initializing dataloader...')
mtest_loader  = SpotifyDataloader(config_fpath=args.config,
                                  mtrain_mode=False, # False = testset for submission
                                  #data_sel=(0, 100),
                                  batch_size=4096,
                                  shuffle=False) 

#mtest_loader  = SpotifyDataloader(config_fpath=args.config,
#                                  mtrain_mode=True, # True, because we use part of trainset as testset
#                                  data_sel=(99965071, 110075071),#(99965071, 124950714), # 20%를 테스트
#                                  batch_size=10,
#                                  shuffle=True) 

def save_submission(output, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path,"w") as f:
        for l in output:
            line = ''.join(map(str,l))
            f.write(line + '\n')
    tqdm.write('submission saved to {}'.format(output_path))
    return
#%% Main
def main():
    # Import module --> load model
    m = os.path.splitext(os.path.split(MODEL_PATH)[1])[0]
    
    MLP             = getattr(__import__(m, fromlist='MLP'), 'MLP')
    RelationNetwork = getattr(__import__(m, fromlist='RelationNetwork'), 'RelationNetwork')
    validate        = getattr(__import__(m, fromlist='validate'), 'validate')
    
    FeatEnc = MLP(input_sz=29, hidden_sz=512, output_sz=64).cuda(GPU)
    RN      = RelationNetwork().cuda(GPU)
    print(CHECKPOINT_PATH)
    print(MODEL_PATH)
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda:{}'.format(GPU))
    tqdm.write("Loading checkpoint from '{0:}'... epoch:{1:} vacc:{2:.6f}".format(CHECKPOINT_PATH, 
               checkpoint['ep'], checkpoint['hist_vacc'][-1]))
    FeatEnc.load_state_dict(checkpoint['FE_state'])
    RN.load_state_dict(checkpoint['RN_state'])

    # Test
    submission = validate(mtest_loader, FeatEnc, RN, True)
    if len(submission)!=31251398: print("WARNING: submission size not matches.");

    # Save
    fpath = SUBMISSION_OUTPUT_PATH + "/submission_{}.txt".format(time.strftime('%Y%m%d_%Hh%Mm'))
    save_submission(submission, fpath)
    tqdm.write("Succesfully saved submission file: {}".format(fpath) )
    return


if __name__ == '__main__':
    main()