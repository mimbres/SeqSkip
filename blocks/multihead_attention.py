#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:06:01 2018

@author: mimbres

based on https://github.com/akurniawan/pytorch-transformer/blob/master/modules/attention.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = nn.Parameter(torch.tensor(
            data=[key_dim], requires_grad=False, dtype=torch.float32))
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention_map = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention_map = attention_map / torch.sqrt(self._key_dim)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention_map[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention_map.size())
            mask = torch.ones(diag_mat.size()) * (-2**32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation and could perform our calculation
            # faster
            attention_map = (attention_map * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        attention_map = F.softmax(attention_map, dim=-1)
        # apply dropout
        attention_map = F.dropout(attention_map, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention_map, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        attention = self.bn(attention.transpose(1, 2).contiguous()).transpose(1, 2)

        return attention, attention_map