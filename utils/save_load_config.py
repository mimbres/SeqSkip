#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:20:51 2018

@author: sungkyun
"""

import argparse
import json

def save_config(args, save_file_path):
    
    dict_args = vars(args)
    with open(save_file_path, 'w') as fp:
        json.dump(dict_args, fp, indent=0)
    
    
def load_config(save_file_path):
    # Over-ride configs to existing arg parser
    with open(save_file_path, 'r') as fp:
        dict_args = json.load(fp)
        
    return argparse.Namespace(**dict_args)

    
        
    
    