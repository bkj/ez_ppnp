#!/usr/bin/env python

"""
    ppr.py
"""

import sys
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import cg
from joblib import Parallel, delayed

import torch
from torch import nn
from torch.nn import functional as F

# --
# Exact PPR

def calc_A_hat(adj, mode):
    A = adj + sp.eye(adj.shape[0])
    D = np.sum(A, axis=1).A1
    if mode == 'sym':
        D_inv = sp.diags(1 / np.sqrt(D))
        return D_inv @ A @ D_inv
    elif mode == 'rw':
        D_inv = sp.diags(1 / D)
        return D_inv @ A


def exact_ppr(adj, alpha, mode='sym'):
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = sp.eye(adj.shape[0]) - (1 - alpha) * A_hat
    return alpha * np.linalg.inv(A_inner.toarray())


def _cg(A_inner, signals):
    tmp = [cg(A_inner, signal, maxiter=10000, tol=1e-8)[0] for signal in signals]
    return np.row_stack(tmp)

def exact_ppr_joblib(adj, alpha, mode='sym', n_jobs=60):
    assert mode == 'sym'
    
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = sp.eye(adj.shape[0]) - (1 - alpha) * A_hat
    
    signals = np.eye(adj.shape[0])
    jobs    = [delayed(_cg)(A_inner, chunk) for chunk in np.array_split(signals, 4 * n_jobs)]
    res     = Parallel(backend='loky', n_jobs=n_jobs, verbose=10)(jobs)
    return np.row_stack(res)

# --

class PPR(nn.Module):
    def __init__(self, ppr):
        self.register_buffer('ppr', torch.FloatTensor(ppr))
        
        self.sparse = False
        self.batch  = False
        
    def forward(self, X, idx, encoder):
        
        # Straightforward
        if not self.sparse and not self.batch:
            return self.ppr[idx] @ encoder(X.cuda())
        
        # Don't pass unecessary points through encoder
        if not self.sparse and self.batch:
            tmp = self.ppr[idx]
            sel = (tmp > 0).any(dim=0)
            tmp = tmp[:,sel]
            
            return tmp @ encoder(X[sel].cuda())
        
        if self.sparse:
            indices  = self.indices[idx]
            values   = self.values[idx]
            
            
            # !! What's the mode efficient way to do this?
            # <<
            u_fwd, u_bwd = indices.unique(return_inverse=True)
            return (encoder(X[u_fwd])[u_bwd] * values.unsqueeze(-1)).sum(axis=1)
            # --
            # Alternatives
            # return (encoder(X[indices]) * values.unsqueeze(-1)).sum(axis=1)
            # return (encoder(X)[indices] * values.unsqueeze(-1)).sum(axis=1)
            # <<
        
        raise Exception()
