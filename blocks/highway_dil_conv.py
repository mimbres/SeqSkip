#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 02:00:29 2018

Highway Blocks (or GLU) with Dilated Conv.
@author: mimbres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Highway Layer & Block:  
class HighwayLayer(nn.Module):
    def __init__(self, in_ch=int, out_ch=int, k_sz=2, dil=int, causality=False, use_glu=False):        
        super(HighwayLayer, self).__init__()
        
        self.out_ch    = out_ch
        self.k_sz      = k_sz        
        self.dil       = dil
        self.causality = causality   
        self.use_glu   = use_glu
        
        self.L = nn.Conv1d(in_ch, out_ch*2, kernel_size=k_sz, dilation=dil)
        self.inorm = nn.InstanceNorm1d(out_ch*2)
        if self.use_glu is True:
            self.glu = nn.GLU(dim=1)
            
        return None
    
    def forward(self, x):
        if self.k_sz is not 1:
            if self.causality is True:
                pad = (self.dil, 0) # left-padding
            else:
                pad = (self.dil, self.dil) # padding to both sides
        else:            
            pad = (0, 0) # in this case, just 1x1 conv..
        h = self.L(F.pad(x, pad))
        h = self.inorm(h)
        if self.use_glu is True:
            return self.glu(h)    
        else:
            h1, h2 = h[:, :self.out_ch,:], h[:, self.out_ch:, :]
            return F.sigmoid(h1) * h2 + (1-F.sigmoid(h1)) * x
    
    
    
class HighwayDCBlock(nn.Module):
    def __init__(self, io_chs=list, k_szs=list, dils=list, causality=False, use_glu=False):
        super(HighwayDCBlock, self).__init__()
        
        #assert(len(io_chs)==len(k_szs) & len(io_chs)==len(dils)) 
        self.causality = causality
        self.use_glu   = use_glu
        self.hlayers   = nn.Sequential()
        self.construct_hlayers(io_chs, k_szs, dils)
        return None

    def construct_hlayers(self, io_chs, k_szs, dils):
        total_layers = len(io_chs) # = len(k_szs)
        for l in range(total_layers):
            self.hlayers.add_module(str(l),
                                    HighwayLayer(in_ch=io_chs[l],
                                                 out_ch=io_chs[l],
                                                 k_sz=k_szs[l],
                                                 dil=dils[l],
                                                 causality=self.causality,
                                                 use_glu=self.use_glu))
        return
    
    def forward(self, x):
        return self.hlayers(x)