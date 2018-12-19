#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:27:50 2018

@author: mimbres

preprocessed trainset dtmm size = (2072002577, 23)
preprocessed trainset dtmm size = (518260758, 23)

NOTE: Don't use date from user-log data. We found some weird dates, log_6_20180918_000000000000.csv, index [1994795,...1994801] date=20320525 

"""
import pandas as pd
import numpy as np
import sys, glob, os, re
from tqdm import tqdm, trange
from utils.save_load_config import load_config
from sklearn.preprocessing import StandardScaler

if len(sys.argv) > 1:
    config_fpath = sys.argv[1]
else:
    config_fpath = "./config_init_dataset.json"  

# Pre-defined Dict
dict_context_type = {'editorial_playlist': 0, 'user_collection': 1,  'catalog': 2, 'radio': 3, 'charts': 4, 'personalized_playlist': 5}
dict_behavior = {'appload': 0, 'trackdone': 1,'clickrow': 2, 'fwdbtn': 3, 'backbtn': 4, 'remote': 5,
                 'trackerror': 6, 'playbtn': 7, 'endplay': 8,'popup': 9,'clickside': 10, 'uriopen': 11, 'logout':12} 


# Functions..
def init_trackfeatures(config_fpath=config_fpath):
    # Load config file & get data root paths...
    config = load_config(config_fpath)
    TRACK_FEATURES_ROOT = config.track_features_root
    OUTPUT_FEAT_NPY_PATH = config.output_data_root + "track_feat.npy"
    
    ## Track Features #########################################################
    # Load track_ids from Track-features, and merge them all...
    feat_all = pd.DataFrame()
    for fpath in tqdm(glob.glob(os.path.join(TRACK_FEATURES_ROOT, "*.csv"))):
        tqdm.write("TRACKFEAT: Collecting data from" + fpath)
        feat_all = feat_all.append(pd.read_csv(fpath, index_col=False, header=0), # usecols=[0] takes only track_ids
                                   ignore_index = True)        

    # Build dict to transform track_ids(string) into indices(int): keep 'dict_track_id' for log data processing...
    dict_track_id = pd.unique(feat_all.track_id)
    dict_track_id = pd.Series(np.arange(len(dict_track_id)), dict_track_id).to_dict()
    
    # Major/minor --> boolin
    _dict = {'major': True, 'minor': False}
    feat_all.loc[:,"mode"] = np.vectorize(_dict.get)(feat_all.loc[:,"mode"]) #---> Replace a whole column, very fast!
    
    # Convert to float32 --> Feature scaling 
    feat_all = feat_all.values[:,1:].astype(np.float32)
    _d15 = feat_all[:,15]
    _scaler = StandardScaler()
    feat_all = _scaler.fit_transform(feat_all)
    feat_all[:,15] = _d15
    
    # Save
    np.save(OUTPUT_FEAT_NPY_PATH, feat_all)
    feat_all = [] # return memory
    ###########################################################################
    return dict_track_id


def init_trdataset(dict_track_id, config_fpath=config_fpath):
    # Load config file & get data root paths...
    config = load_config(config_fpath)
    TR_LOG_DATA_ROOT = config.tr_log_data_root
    OUTPUT_TR_SESSION_SPLIT_IDX_PATH = config.output_data_root + "tr_session_split_idx.npy"
    OUTPUT_TR_LOG_MEMMAP_DAT_PATH = config.output_data_root + "tr_log_memmap.dat"
    os.makedirs(os.path.dirname(config.output_data_root), exist_ok=True)
    
    ## User Log Data ##########################################################
    
    # Training-log-data #######################################################
    session_split_idx = np.empty(shape=(0,), dtype=np.int32) # This will be appended in the loop...
    dt_mm = np.memmap(OUTPUT_TR_LOG_MEMMAP_DAT_PATH, mode='w+', dtype=np.uint8, shape=(1,1)) # This memmmap will record the resulting packed data(dt)...
    del dt_mm
    dtmm_old_shape = (0,0)
    index_start_from = 0 # keep the last number of items in user-log-data
    
    for file_count, fpath in enumerate(tqdm(glob.glob(os.path.join(TR_LOG_DATA_ROOT, "*.csv")))):
        tqdm.write("TRDATA: Collecting data from" + fpath)
        df = pd.read_csv(fpath, index_col=False, header=0)#, nrows=200)#usecols=[0,1])

        # Convert hash ids(string) into indices(int)
        # - Col 0, session_id
        _dict = pd.unique(df.session_id)
        _dict = pd.Series(np.arange(len(_dict)), _dict).to_dict()
        df.loc[:,"session_id"] = np.vectorize(_dict.get)(df.session_id) #---> Replace a whole column, very fast!
        
        # - Col 3, track_id_clean: reuse 'dict_track_id'
        df.loc[:, "track_id_clean"] = np.vectorize(dict_track_id.get)(df.track_id_clean)
        
        # - Col 16, date 2018-09-18 -> 20180918
        df.loc[:,"date"] = pd.to_numeric(df.date.str.replace('-',''), downcast='integer')  
            
        # - Col 18,19,20, context, bhs, bhe --> {6,10,11} classes
        df.loc[:,"context_type"] = np.vectorize(dict_context_type.get)(df.context_type)
        df.loc[:,"hist_user_behavior_reason_start"] = np.vectorize(dict_behavior.get)(df.hist_user_behavior_reason_start)
        df.loc[:,"hist_user_behavior_reason_end"] = np.vectorize(dict_behavior.get)(df.hist_user_behavior_reason_end)

        # Generate & append session split index
        session_split_idx = np.hstack((session_split_idx,
                np.insert(np.cumsum(np.unique(df.session_id.values, return_counts=True)[1])[:-1], 0, 0) + index_start_from))   
        index_start_from += len(df)
        
        # Pack data for new memmap array
        # [0,1,2,3]       | track_id                                        | int32 (packed as 4 uint8)
        # [4,5,6,7]       | date                                            | int32 (packed as 4 uint8) 
        # [8]             | hour                                            | uint8
        # [9,10]          | n_seekfwd, n_seekback                           | uint8
        # [11,...19]      | skip_1,2,3, hist_sh, ct_swc, no_p, s_p, l_p, pr | binary (as uint8)
        # [20,..22]       | context, bh_start, bh_end                       | uint8  
        dt = np.zeros(shape=(len(df), 23), dtype=np.uint8)
        dt[:,0:4] = df.track_id_clean.values.astype(np.int32).view(np.uint8).reshape(-1,4) # 3076898 --> [34,243,46,0] (4x uint8)
        dt[:,4:8] = df.date.values.astype(np.int32).view(np.uint8).reshape(-1,4)
        dt[:,8] = df.hour_of_day.values.astype(np.uint8)
        dt[:,9:11] = df.loc[:,["hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback"]].values.astype(np.uint8) 
        dt[:,11:20] = df.loc[:,["skip_1","skip_2","skip_3","hist_user_behavior_is_shuffle",
          "context_switch", "no_pause_before_play", "short_pause_before_play",
          "long_pause_before_play", "premium"]].values.astype(np.uint8)
        dt[:,[20,21,22]] = df.loc[:,["context_type","hist_user_behavior_reason_start",
          "hist_user_behavior_reason_end"]].values.astype(np.uint8)
        
        # Write to memmap file...
        dtmm_new_shape = (dtmm_old_shape[0] + dt.shape[0], dt.shape[1]) 
        dtmm = np.memmap(OUTPUT_TR_LOG_MEMMAP_DAT_PATH, mode='r+', dtype=np.uint8, shape=dtmm_new_shape) 
        dtmm[dtmm_old_shape[0]:, :] = dt
        dtmm.flush;  # Force writing to disk
        dtmm_old_shape = dtmm_new_shape
        tqdm.write('dtmm-writer {0:4d}: New memmap size is {1:.2f}Gb.'.format(file_count, dtmm_new_shape[0]*dtmm_new_shape[1]/2**30))        
        
    # Save the output session split index...
    np.save(OUTPUT_TR_SESSION_SPLIT_IDX_PATH, session_split_idx)
    ###########################################################################   
    return 0
    
    
    
def init_tsdataset(dict_track_id, config_fpath=config_fpath):       
    # Load config file & get data root paths...
    config = load_config(config_fpath)
    TS_LOG_DATA_ROOT = config.ts_log_data_root
    OUTPUT_TS_SESSION_SPLIT_IDX_PATH = config.output_data_root + "ts_session_split_idx.npy"
    OUTPUT_TS_LOG_MEMMAP_DAT_PATH = config.output_data_root + "ts_log_memmap.dat"
    os.makedirs(os.path.dirname(config.output_data_root), exist_ok=True)
    
    # Test-log-data #######################################################
    session_split_idx = np.empty(shape=(0,), dtype=np.int32) # This will be appended in the loop...
    dt_mm = np.memmap(OUTPUT_TS_LOG_MEMMAP_DAT_PATH, mode='w+', dtype=np.uint8, shape=(1,1)) # This memmmap will record the resulting packed data(dt)...
    del dt_mm
    dtmm_old_shape = (0,0)
    index_start_from = 0 # keep the last number of items in user-log-data
    
    for file_count, fpath_sup in enumerate(tqdm(sorted(glob.glob(os.path.join(TS_LOG_DATA_ROOT, "log_prehistory*.csv"))))):
        fpath_que = fpath_sup.replace("prehistory", "input")
        assert os.path.isfile(fpath_que), "Error! File '{}' does not exist.".format(fpath_que)
        tqdm.write("TSDATA: Collecting data from:\n" + fpath_sup + "\n" + fpath_que)
        
        # Load support and query from each files: df_s from fpath_sup, df_q from fpath_q.
        df_s = pd.read_csv(fpath_sup, index_col=False, header=0)#, nrows=200)#usecols=[0,1])
        df_q = pd.read_csv(fpath_que, index_col=False, header=0)#, nrows=200)#usecols=[0,1])
        # df = pd.merge_ordered(df_s, df_q, left_by='session_id') # This merges by group of sessions but too SLOW!
        # Instead, we work with numpy arrays here.
        s = df_s.values # converted as numpy ndarray
        q = df_q.values
        
        _s_sess_head_idx = np.unique(s[:,0], return_index=1)[1]
        _s_sess_tail_idx = np.append(_s_sess_head_idx[1:], len(s))
        _q_sess_head_idx = np.unique(q[:,0], return_index=1)[1]
        _q_sess_tail_idx = np.append(_q_sess_head_idx[1:], len(q))
        sess_lengths = s[_s_sess_head_idx,2]
        num_log_rows = len(s)+len(q)
        num_sess     = len(sess_lengths)
        _dt_sess_head_idx = np.insert(np.cumsum(sess_lengths), 0,0)[:-1] # the two dfs will be merged inplace using this idx.
        _dt_sess_tail_idx = np.cumsum(sess_lengths)
        
        # Pack data for new each session
        dt = np.zeros(shape=(num_log_rows, 23), dtype=np.uint8) 
        for i in trange(num_sess, desc='Pack data loop', position=1):
            # [0,1,2,3]       | track_id                                        | int32 (packed as 4 uint8)
            # [4,5,6,7]       | date                                            | int32 (packed as 4 uint8) 
            # [8]             | hour                                            | uint8
            # [9,10]          | n_seekfwd, n_seekback                           | uint8
            # [11,...19]      | skip_1,2,3, hist_sh, ct_swc, no_p, s_p, l_p, pr | binary (as uint8)
            # [20,..22]       | context, bh_start, bh_end                       | uint8  
            _s_sel = range(_s_sess_head_idx[i], _s_sess_tail_idx[i])
            _q_sel = range(_q_sess_head_idx[i], _q_sess_tail_idx[i])
            #_dt_sel = range(_dt_sess_head_idx[i], _dt_sess_tail_idx[i])
            _dt_s_sel = range(_dt_sess_head_idx[i], _dt_sess_head_idx[i] + len(_s_sel)) # head of session : df_splt : tail of session  
            _dt_q_sel = range(_dt_sess_head_idx[i] + len(_s_sel), _dt_sess_tail_idx[i])
            
            # [0,1,2,3]: track_id --> 64bit int --> 8-bit digits with length 4
            dt[_dt_s_sel, 0:4] = np.vectorize(dict_track_id.get)(s[_s_sel,3]).astype(np.int32).view(np.uint8).reshape(-1,4)   
            dt[_dt_q_sel, 0:4] = np.vectorize(dict_track_id.get)(q[_q_sel,1]).astype(np.int32).view(np.uint8).reshape(-1,4)
            
            # [4,5,6,7]: date
            dt[_dt_s_sel, 4:8] = np.asarray([int(re.sub('-', '',date)) for date in s[_s_sel,16]], dtype=np.int32).view(np.uint8).reshape(-1,4)   
            # [8]: hour
            dt[_dt_s_sel, 8] = s[_s_sel,15].astype(np.uint8)
            # [9,10]: n_seekfwd, n_seekback
            dt[_dt_s_sel, 9:11] = s[_s_sel,12:14].astype(np.uint8)
            # [11,...19]: skip_1,2,3, hist_sh, ct_swc, no_p, s_p, l_p, pr
            dt[_dt_s_sel, 11:20] = s[_s_sel,:][:,[4,5,6,14,8,9,10,11,17]].astype(np.uint8)
            # [20,...22]: context, bh_start, bh_end
            dt[_dt_s_sel, 20] = np.vectorize(dict_context_type.get)(s[_s_sel,18]).astype(np.uint8)
            dt[_dt_s_sel, 21] = np.vectorize(dict_behavior.get)(s[_s_sel,19]).astype(np.uint8)
            dt[_dt_s_sel, 22] = np.vectorize(dict_behavior.get)(s[_s_sel,20]).astype(np.uint8)

        # Generate & append session split index
        session_split_idx = np.hstack((session_split_idx, _dt_sess_head_idx + index_start_from))   
        index_start_from += num_log_rows

        
        # Write to memmap file...
        dtmm_new_shape = (dtmm_old_shape[0] + dt.shape[0], dt.shape[1]) 
        dtmm = np.memmap(OUTPUT_TS_LOG_MEMMAP_DAT_PATH, mode='r+', dtype=np.uint8, shape=dtmm_new_shape) 
        dtmm[dtmm_old_shape[0]:, :] = dt
        dtmm.flush;  # Force writing to disk
        dtmm_old_shape = dtmm_new_shape
        tqdm.write('dtmm-writer {0:4d}: New memmap size is {1:.2f}Gb.'.format(file_count, dtmm_new_shape[0]*dtmm_new_shape[1]/2**30))        
        
    # Save the output session split index...
    np.save(OUTPUT_TS_SESSION_SPLIT_IDX_PATH, session_split_idx)
    ###########################################################################   
    return 0


def main():
    dict_track_id = init_trackfeatures()
    #init_trdataset(dict_track_id)
    init_tsdataset(dict_track_id)

if __name__ == '__main__':
    main()