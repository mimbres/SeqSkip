#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:01:10 2018

@author: mimbres

NOTE: In advance of using this code, plaease run "preparing_data.py" once.
"""
import numpy as np
import sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.save_load_config import load_config
from utils.matrix_math import indices_to_one_hot


# TRSET stats: 2072002577 items within 124950714 sessions

def SpotifyDataloader(config_fpath="./config_init_dataset.json",
                      mtrain_mode=True,
                      random_track_order=False,
                      data_sel=None,
                      batch_size=1,
                      shuffle=False,
                      num_workers=4,
                      pin_memory=True):
    
    dset = SpotifyDataset(config_fpath=config_fpath,
                                mtrain_mode=mtrain_mode,
                                random_track_order=random_track_order,
                                data_sel=data_sel)
    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=pin_memory)
    return dloader
     


class SpotifyDataset(Dataset):
    '''
    mtrain_mode : (bool) True for meta-train(default) / False for meta-test
    random_track_order !! NOT IMPLEMENTED YET !!: (bool) True for non-sequential / False for sequential(default) 
    data_sel: list(start_idx, end_idx) selection of sessions for using only partial data, None(default) uses all.
    
    NOTE: __getitem__() always keeps the output session length as 20. To do so, 
          it can add dummy rows filled with zeros. 
    '''
    def __init__(self,
                 config_fpath="./config_init_dataset.json",
                 mtrain_mode=True,
                 random_track_order=False, 
                 data_sel=None):
        
        config = load_config(config_fpath)
        TRACK_FEAT_NPY_PATH       = config.output_data_root + "track_feat.npy"
        TR_LOG_MEMMAP_DAT_PATH    = config.output_data_root + "tr_log_memmap.dat"
        TR_SESSION_SPLIT_IDX_PATH = config.output_data_root + "tr_session_split_idx.npy"
        TR_LOG_DATA_SHAPE = (2072002577, 23)
        #TS_LOG_MEMMAP_DAT_PATH    = config.output_data_root + "ts_log_memmap.dat"
        #TS_SESSION_SPLIT_IDX_PATH = config.output_data_root + "ts_session_split_idx.npy"
        #TS_LOG_SHAPE = (, )
        
        self.mtrain_mode           = mtrain_mode
        self.random_track_order    = random_track_order
        self.data_sel              = data_sel
        self.session_start_end_idx = []
        self.dt_mm                 = []
        self.track_feat            = []
        
        if self.mtrain_mode:
            fpath_log_mm_dat   = TR_LOG_MEMMAP_DAT_PATH
            fpath_sess_spl_idx = TR_SESSION_SPLIT_IDX_PATH
        else:
            fpath_log_mm_dat   = TS_LOG_MEMMAP_DAT_PATH
            fpath_sess_spl_idx = TS_SESSION_SPLIT_IDX_PATH
        
        
        # Import session log data: 'dt_mm' 
        self.dt_mm = np.memmap(TR_LOG_MEMMAP_DAT_PATH, dtype='uint8', mode='r', shape=TR_LOG_DATA_SHAPE)
        
        
        # Prepare 'session_start_end_idx' from 'session_split_indices'
        _sess_split = np.load(fpath_sess_spl_idx) # shape=(index,)   
        if self.data_sel is not None:
            _sess_split = _sess_split[range(data_sel[0], data_sel[1])]
        self.session_start_end_idx = np.empty(shape=(len(_sess_split), 2), dtype=np.uint32)
        self.session_start_end_idx[:,0]   = _sess_split        
        self.session_start_end_idx[:,1] = np.r_[_sess_split[1:], len(self.dt_mm)-1]
        
        # Prepare 'track_feat' 
        self.track_feat = np.load(TRACK_FEAT_NPY_PATH) 
        return None
    
    
    
    def __getitem__(self, index): 
        session_log = self.dt_mm[np.arange(self.session_start_end_idx[index,0],
                                           self.session_start_end_idx[index,1]), :]
        
        # 'num_items': the number of items(~=tracks) in one session   
        num_items = len(session_log)  
    
        # Unpack track_id and dates (packed as 4 uint8, each):
        track_ids = np.ascontiguousarray(session_log[:, :4]).view(np.uint32).flatten() # dim[0,1,2,3] for track_id
        #dates     = np.ascontiguousarray(session_log[:, 4:8]).view(np.uint32).flatten() # dim[4,5,6,7] for date.   
        # date는 일단 안씀.., 
        
        # NOTE: We always keep the session length of output feature as 20, and several last items are dummys filled with 0s.
        # DIMs: feats(dim=70) = [log_feat(dim=41), track_feat(dim=29)] 
        feats  = np.zeros(shape=(20, 70), dtype=np.float32)
        labels = np.zeros(shape=(20, 3), dtype=np.float32)
        # Fill out the feature dimensions as:
        # [0]       : minmax-scaled date in the range of -1 to 1
        # [1,...8] : n_seekfwd, n_seekback, skip_1,2,3, hist_sh, ct_swc, no_p, s_p, l_p, premium
        # [9,..40] : one-hot-encoded categorical labels of context_type, bh_start, bh_end 
        # [41,..69] : track features 
        feats[:num_items, 0]     = (session_log[:, 8] / 23) * 2 - 1
        feats[:num_items, 1:9]   = session_log[:, [9,10,14,15,16,17,18,19]]
        feats[:num_items, 9:15]  = indices_to_one_hot(data=session_log[:,20], nb_classes=6)
        feats[:num_items, 15:28] = indices_to_one_hot(data=session_log[:,21], nb_classes=13)
        feats[:num_items, 28:41] = indices_to_one_hot(data=session_log[:,22], nb_classes=13)        
        feats[:num_items, 41:]   = self.track_feat[track_ids, :]
        labels[:num_items, :]    = session_log[:, [11,12,13]] 
        
        return feats, labels, num_items, index


    def __len__(self):
        return len(self.session_start_end_idx) # return the number of sessions that we have
    
    
    def plot_bar(self, a, b, title):
        import matplotlib.pyplot as plt
        plt.bar(a,b)
        plt.ylabel('play count')
        plt.title(title)
        return