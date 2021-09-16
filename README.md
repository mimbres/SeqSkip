# mimbres/seqskip:[Spotify Sequential Skip Prediction Challenge](https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge)
(Since 9th Dec 2018)
[Paper, WSDM 2019 Cup Result](https://arxiv.org/abs/1901.08203)

# Summary
Our best submission result (aacc=0.637) with sequence learning was based on seqskip_train_seq1HL.py
- For test, seqskip_test_seq1.py
- This model consists of highway layers(or GLUs) and dilated convolution layers.
- Very similar model can be found in their encoder part of [DCTTS](https://arxiv.org/abs/1710.08969) (ICASSP 2018, H Tachibana et al.).
- Other variants of this model use various attention modules.

Other approaches (in progress):
- [Multi-task learning](https://arxiv.org/pdf/1707.08114.pdf) with 1 stack generative model (aacc=0.635) that generates userlog in advance of skip-prediction is in seqskip_train_GenLog1H_Skip1eH.py
- 2 stack large generative model (not submitted) is very slow, seqskip_train_GenLog1H_Skip1eH256full.py
- [Learn to compare](https://arxiv.org/abs/1711.06025) (CVPR 2018, F Sung et al.), a relation-net based meta-leraning approach is in seqskip_train_rnbc1*.py
- [Learn to compare](https://arxiv.org/abs/1711.06025) with Some improvement is in seqskip_train_rnbc2*.py
- A naive [Transformer](https://arxiv.org/abs/1706.03762) with multi-head attention model is in seqskip_train_MH*.py
- seqskip_train_MH_seq1H_v3_dt2.py can be thought as a similar approach to [SNAIL](https://arxiv.org/abs/1707.03141) (CVPR 2018, N Mishra et al.) without their data shuffling method. 
- [Distillation](https://arxiv.org/abs/1503.02531) (NIPS 2014, G Hinton et al.) approaches are in seqskip_train_T* and seqskip_train_S* for the teacher (Surprisingly, aacc>0.8 in validation, by using metadata for queries!!) and student nets, respectively. We beilieve that this approach can be an intersting issue at the workshop!
- etc.

# Note that we did not use any external data nor pre-trained model.
# Data split:
- 80% for training
- 20% for validation
# System requirements:
- PyTorch 0.4 or 1.0
- pandas, numpy, tqdm, sklearn
- tested with Titan V or 1080ti GPU
# Preprocessing:
- Please modify config_init_dataset.json to set path of original data files. 
- Please run preparing_data.py(Because the data is huge, we compress it as 8-bit uint formatted np.memmap)
- Thanks to np.memmap, we can have 50Gb+ virtual memory for meta data.
- Acoustic features are loaded into physical memory (11Gb).
- spotify_data_loader.py or spotify_data_loader_v2.py is the data loader class used for training.
- Normalization:
  - Many categorical user behavior logs are decoded into one-hot vectors.
  - Number of click fwd/backwd was minmax normalized after taking logarithm.
  - We didn't make use of dates.
  - Acoustic echonest features were standardized to mean=0 and std=1.

# Download pre-processed data (speed-up training with memmap)
https://storage.googleapis.com/skchang_data/seqskip/data/tr_log_memmap.dat
https://storage.googleapis.com/skchang_data/seqskip/data/tr_session_split_idx.npy
https://storage.googleapis.com/skchang_data/seqskip/data/track_feat.npy
https://storage.googleapis.com/skchang_data/seqskip/data/ts_log_memmap.dat
https://storage.googleapis.com/skchang_data/seqskip/data/ts_session_split_idx.npy
(Updated, Apr 2020)

# Download original dataset without login
- [Training_Set_Split_Download.txt](https://crowdai-prd.s3.eu-central-1.amazonaws.com/dataset_files/challenge_50/2cb466df-239c-4fbb-9883-7ff784a2b5e9_Training_Set_Split_Download.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAILFF3ZEGG7Y4HXEQ%2F20210916%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20210916T141007Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=f7dc0bbda1625647acdef122c029cdc9a1d638408aecfa6137dc6b952e3f6263)
- [Sample_Submissions.tar.gz](https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181113_submissions.tar.gz)
- [Training_Set.tar.gz](https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181113_training_set.tar.gz)
- [Dataset Description](https://crowdai-prd.s3.eu-central-1.amazonaws.com/dataset_files/challenge_50/7dcfad42-65c6-4481-abe8-5a44339fa305_Dataset%20Description.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAILFF3ZEGG7Y4HXEQ%2F20210916%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20210916T141008Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=79e00ec4fc5a8d083aa77a3da93694d86525b8c9904e9850f9cb1b36b8a9c212)
- [Terms and Conditions](https://crowdai-prd.s3.eu-central-1.amazonaws.com/dataset_files/challenge_50/7826027a-c745-4faa-94ec-bae6783fdc41_Terms%20and%20Conditions.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAILFF3ZEGG7Y4HXEQ%2F20210916%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20210916T141008Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=2de282e848c52d86c4827b6465d33956d148dd6cafbe8079e51de8f2961048b7)
- [Track_Features.tar.gz](https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181120_track_features.tar.gz)
- [Test_Set.tar.gz](https://os.zhdk.cloud.switch.ch/swift/v1/crowdai-public/spotify-sequential-skip-prediction-challenge/20181113_test_set.tar.gz)
- [Training_Set_And_Track_Features_Mini](https://crowdai-prd.s3.eu-central-1.amazonaws.com/dataset_files/challenge_50/16772e7f-7871-4d42-a44f-5f399f40fd94_training_set_track_features_mini.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAILFF3ZEGG7Y4HXEQ%2F20210916%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20210916T141008Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=cb96dfffbe00d19bcbbf06332944c4ec6dc56aaf0778e895ba17fd7c3f42450d)
(Updated, Sep 2021)

# Plots:
- plot_dataset.py can display some useful stats of Spotify dataset. You can see them in /images folder.

# This repository needs clean-up!

