#!/usr/bin/env python

"""
    training.py
"""

import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch.nn import functional as F

def train_unsupervised(model, lr, epochs, batch_size):
    assert model.training
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_hist = []
    gen = trange(epochs)
    for epoch in gen:
        
        node_enc, hood_enc = model(idx=np.random.choice(n_nodes, batch_size))
        
        loss = ((node_enc - hood_enc) ** 2).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss_hist.append(float(loss))
        gen.set_postfix(loss=float(loss))
    
    return loss_hist