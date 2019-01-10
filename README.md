# SeqSkip

Our latest/best submission (aacc=0.637) with sequence learning was based on seqskip_train_seq1HL.py
- For test, seqskip_test_seq1.py

Other approaches (in progress):
- 1 stack generative model (aacc=0.635) is in seqskip_train_GenLog1H_Skip1eH.py
- 2 stack large generative model (not submitted) is very slow, seqskip_train_GenLog1H_Skip1eH256full.py
- "Learn to compare", an relation-net based model is in seqskip_train_rnbc1*.py
- "Learn to compare" with Some improvement is in seqskip_train_rnbc2*.py
- Distillation approaches are in seqskip_train_T* and seqskip_train_S*. 
- and so on.

# Note that we did not use any external data nor pre-trained model.
# This repository needs clean-up!
