#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:05:00 2018

@author: mimbres
"""
import numpy as np

def indices_to_one_hot(data, nb_classes, target_dtype=np.float32):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets].astype(target_dtype)