#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 06:08:38 2018

@author: mimbres
"""

q=np.ascontiguousarray(dt_mm[:, 4:8]).view(np.uint32).flatten()

total_cnt = 0
for file_count, fpath in enumerate(glob.glob(os.path.join(TR_LOG_DATA_ROOT, "*.csv"))):
    with open(fpath) as f:
        cnt_this = (sum(1 for line in f)) - 1
    
    total_cnt += cnt_this
    print(file_count)
    if total_cnt>=1715392750:
        nth_line = total_cnt - 1715392750
        print('FOUND!!! in file {} line {}'.format(fpath, nth_line))
        break
    
#FOUND!!! in file /mnt/ssd2/SeqSkip/training_set/log_6_20180918_000000000000.csv line 1195205

df = pd.read_csv('/mnt/ssd2/SeqSkip/training_set/log_6_20180918_000000000000.csv', index_col=False, header=0)

total_cnt = 1715392735 - 1 +1195221

#%% stats from test data
import pandas as pd
import glob
TS_LOG_DATA_ROOT = '/mnt/ssd3/test_set/'

for fpath in glob.glob(os.path.join(TS_LOG_DATA_ROOT, "*.csv")):
        #print("Collecting data from" + fpath)
        df = pd.read_csv(fpath, index_col=False, header=0) # usecols=[0] takes only track_ids
                                       
#하다맘. 총 19개라치면 9개가 support, 10개가 query인거 확인해서 더안봐도됨



#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.save_load_config import load_config

config_fpath = "./config_init_dataset.json"    
config = load_config(config_fpath)
TR_LOG_MEMMAP_DAT_PATH    = config.output_data_root + "tr_log_memmap.dat"
TR_LOG_DATA_SHAPE = (2072002577, 23)

dt_mm = np.memmap(TR_LOG_MEMMAP_DAT_PATH, dtype='uint8', mode='r', shape=TR_LOG_DATA_SHAPE)

#%% bar-graph of date
date_all = np.ascontiguousarray(dt_mm[:, 4:8]).view(np.uint32).flatten()
dates, play_counts = np.unique(date_all, return_counts=True)
del date_all

fig = plt.figure
plt.bar(np.arange(len(play_counts)), play_counts)
x_index = np.arange(len(play_counts), step=50)
plt.xticks(x_index, dates.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title('date') 


#%% Bar-graph: stats of skip/nskip
TR_SESSION_SPLIT_IDX_PATH = config.output_data_root + "tr_session_split_idx.npy"
_sess_split = np.load(TR_SESSION_SPLIT_IDX_PATH)
session_start_end_idx = np.empty(shape=(len(_sess_split), 2), dtype=np.uint32)
session_start_end_idx[:,0]   = _sess_split        
session_start_end_idx[:,1] = np.r_[_sess_split[1:], len(dt_mm)-1]

_total_sess_len=0
for i in range(len(session_start_end_idx)):
    _total_sess_len += (session_start_end_idx[i,1] - session_start_end_idx[i,0])
avg_sess_len = _total_sess_len / len(session_start_end_idx)


total_skip = 0; total_notskip = 0; total = 0
min_skip = 20; max_skip = 0; avg_skip = 0
min_sess_len = 20; max_sess_len = 0
 
for i, si in enumerate(tqdm(session_start_end_idx)):
    sum_skip_insess = sum(dt_mm[si[0]:si[1], 12])
    sub_total = len(dt_mm[si[0]:si[1], 12])
    total_skip += sum_skip_insess
    total_notskip += (sub_total - sum__skip_insess) 
    total += sub_total
    
    if sum_skip_insess < min_skip:
        min_skip = sum_skip_insess
    
    if sum_skip_insess > max_skip:
        max_skip = sum_skip_insess
    
    if sub_total < min_sess_len:
        min_sess_len = sub_total
        
    if sub_total > max_sess_len:
        max_sess_len = sub_total

avg_skip = total_skip/len(session_start_end_idx)
print('\n total_skip:{},\n total_notskip:{},\n total:{},\n min_skip:{},\n max_skip:{},\n avg_skip:{},\n min_sess_len:{},\n max_sess_len:{}'.format(
        total_skip, total_notskip, total, min_skip, max_skip, avg_skip, min_sess_len, max_sess_len))


# total skip, notskip
plt.figure()
labels = ['skip', 'notskip']
ax = plt.bar(labels,[total_skip, total_notskip])
plt.text(0-.25,200000, str(total_skip), fontsize=14,verticalalignment='bottom')
plt.text(1-.25,1, str(total_notskip), fontsize=14,verticalalignment='bottom')
plt.title('total = {}'.format(total))

# min max avg skip
plt.figure()
labels = ['min\nskip', 'avgerage\nskip', 'max\nskip', 'min\nsession\nlength','average\nsession\nlength','max\nsession\nlength']
plt.bar(labels, [min_skip, avg_skip, max_skip, min_sess_len, avg_sess_len, max_sess_len])
plt.yticks(np.arange(0,21,2))
plt.grid(linestyle='--', axis='y')
plt.title('number of skips in sessions')



#%% histogram of n_seekfwd, n_seekback
import matplotlib.pyplot as plt
import numpy as np

# n_seekfwd
n_seekfwd, cnt = np.unique(mtrain_loader.dataset.dt_mm[:,9], return_counts=True)
plt.figure(); plt.bar(np.arange(len(cnt)), cnt)
x_index = np.arange(len(cnt), step=50)
plt.xticks(x_index, n_seekfwd.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title("histogram of n_seekfwd(all)")

n_seekfwd = n_seekfwd[1:]; cnt = cnt[1:]
plt.figure(); plt.bar(np.arange(len(cnt)), cnt)
x_index = np.arange(len(cnt), step=50)
plt.xticks(x_index, n_seekfwd.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title("histogram of n_seekfwd(NOT 0)")

n_seekfwd = n_seekfwd[5:]; cnt = cnt[5:]
plt.figure(); plt.bar(np.arange(len(cnt)), cnt)
x_index = np.arange(len(cnt), step=50)
plt.xticks(x_index, n_seekfwd.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title("histogram of n_seekfwd(>5)")



# n_seekback
n_seekback, cnt = np.unique(mtrain_loader.dataset.dt_mm[:,10], return_counts=True)
plt.figure(); plt.bar(np.arange(len(cnt)), cnt)
x_index = np.arange(len(cnt), step=50)
plt.xticks(x_index, n_seekback.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title("histogram of n_seekback(all)")

n_seekback = n_seekback[1:]; cnt = cnt[1:]
plt.figure(); plt.bar(np.arange(len(cnt)), cnt)
x_index = np.arange(len(cnt), step=50)
plt.xticks(x_index, n_seekback.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title("histogram of n_seekback(NOT 0)")

n_seekback = n_seekback[5:]; cnt = cnt[5:]
plt.figure(); plt.bar(np.arange(len(cnt)), cnt)
x_index = np.arange(len(cnt), step=50)
plt.xticks(x_index, n_seekback.astype(str)[x_index], fontsize=12, rotation=45)
plt.ylabel('play count'); plt.title("histogram of n_seekback(>5)")


#%% Distillation plot
target = SMT_Enc(x_feat_T)
target2 = SMT_EncFeat(x_feat_T)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3,4,1)
plt.plot(target.detach().cpu().numpy()[1,:,8])
plt.title('dilated_conv_Encoder output: S8(skip=0)')

plt.subplot(3,4,2)
plt.plot(target2.detach().cpu().numpy()[1,:,8])
plt.title('(L-1) layer of classifier output S8(skip=0)')

plt.subplot(3,4,5)
plt.plot(target.detach().cpu().numpy()[1,:,7])
plt.title('dilated_conv_Encoder output: S7(skip=0)')

plt.subplot(3,4,6)
plt.plot(target2.detach().cpu().numpy()[1,:,7])
plt.title('(L-1) layer of classifier output S7(skip=0)')

plt.subplot(3,4,9)
plt.plot(target.detach().cpu().numpy()[1,:,9])
plt.title('dilated_conv_Encoder output: S9(skip=1)')

plt.subplot(3,4,10)
plt.plot(target2.detach().cpu().numpy()[1,:,9])
plt.title('(L-1) layer of classifier output S9(skip=1)')

plt.subplot(3,4,3)
plt.plot(target.detach().cpu().numpy()[1,:,12])
plt.title('dilated_conv_Encoder output: Q3(skip=0)')

plt.subplot(3,4,4)
plt.plot(target2.detach().cpu().numpy()[1,:,12])
plt.title('(L-1) layer of classifier output Q3(skip=0)')

plt.subplot(3,4,7)
plt.plot(target.detach().cpu().numpy()[1,:,11])
plt.title('dilated_conv_Encoder output: Q2(skip=1)')

plt.subplot(3,4,8)
plt.plot(target2.detach().cpu().numpy()[1,:,11])
plt.title('(L-1) layer of classifier output Q2(skip=1)')

plt.subplot(3,4,11)
plt.plot(target.detach().cpu().numpy()[1,:,10])
plt.title('dilated_conv_Encoder output: Q1(skip=1)')

plt.subplot(3,4,12)
plt.plot(target2.detach().cpu().numpy()[1,:,10])
plt.title('(L-1) layer of classifier output Q1(skip=1)')