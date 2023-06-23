#!/usr/bin/env python

"""
    model.py
"""

import torch
from torch import nn
from torch.nn import functional as F

class NormalizedEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._emb = nn.Embedding(*args, **kwargs)
    
    def forward(self, x):
        w = F.normalize(self._emb.weight, dim=-1)
        return w[x]


class EmbeddingPPNP(nn.Module):
    def __init__(self, ppr, n_nodes, hidden_dim):
        
        super().__init__()
        
        self.encoder = NormalizedEmbedding(n_nodes, hidden_dim)
        self.ppr     = ppr
        self.X       = torch.arange(n_nodes)
        self.n_nodes = n_nodes
    
    def forward(self, idx):
        node_enc = self.encoder(idx)
        hood_enc = self.ppr(self.X, idx, self.encoder)
        return node_enc, hood_enc


class SupervisedEmbeddingPPNP(nn.Module):
    def __init__(self, ppr, n_nodes, hidden_dim, n_classes):
        
        super().__init__()
        
        self.encoder = NormalizedEmbedding(n_nodes, hidden_dim)
        self.output  = nn.Linear(hidden_dim, n_classes)
        
        self.ppr     = ppr
        self.X       = torch.arange(n_nodes)
        self.n_nodes = n_nodes
    
    def forward(self, idx):
        # node_enc = self.encoder(idx)
        hood_enc = self.ppr(self.X, idx, self.encoder)
        return self.output(hood_enc)
