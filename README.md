# SeqSkip, team name: mimbres


Our latest/best submission (aacc=0.637) with sequence learning was based on seqskip_train_seq1HL.py
- For test, seqskip_test_seq1.py

Other approaches (in progress):
- 1 stack generative model (aacc=0.635) is in seqskip_train_GenLog1H_Skip1eH.py
- 2 stack large generative model (not submitted) is very slow, seqskip_train_GenLog1H_Skip1eH256full.py
- "Learn to compare", a relation-net based meta-leraning approach is in seqskip_train_rnbc1*.py
- "Learn to compare" with Some improvement is in seqskip_train_rnbc2*.py
- Distillation approaches are in seqskip_train_T* and seqskip_train_S* for the teacher and student nets, respectively.
- etc.

# Note that we did not use any external data nor pre-trained model.
# System requirements:
- PyTorch 0.4 or 1.0
- pandas, numpy
- tested with Titan V or 1080ti GPU
# Preprocessing:
- Please modify config_init_dataset.json to set path of original data files. 
- Please run preparing_data.py(Because the data is huge, we compress it as 8-bit uint formatted np.memmap)
- Thanks to np.memmap, we can have 50Gb+ virtual memory for meta data.
- Acoustic features are loaded into physical memory(11Gb).
- spotify_data_loader.py or spotify_data_loader_v2.py is the data loader class used for training.
# Plots:
- plot_dataset.py can display some stats of dataset

# This repository needs clean-up!


