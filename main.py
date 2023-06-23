#!/usr/env/python

"""
  main.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.preprocessing import normalize

from sklearn.svm import LinearSVC
from cas_classifier import CASClassifier


import torch
from torch import nn
from torch.nn import functional as F

_DATASETS = [
  # "cifar10",
  "Multimodal-Fatima/OxfordPets_train",
  "Multimodal-Fatima/StanfordCars_train",
  "jonathan-roberts1/NWPU-RESISC45",
  "nelorth/oxford-flowers",
  "fashion_mnist",
  "food101",
]

# --
# IO

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='jonathan-roberts1/NWPU-RESISC45')
  parser.add_argument('--model',   type=str, default='openai/clip-vit-large-patch14')
  parser.add_argument('--sample',  type=int, default=10_000)
  parser.add_argument('--seed',    type=int, default=234)
  args = parser.parse_args()
  
  assert args.dataset in _DATASETS
  
  return args


args = parse_args()
np.random.seed(args.seed)

X = np.load(os.path.join('/home/ubuntu/data/img_feats', args.dataset, 'train', args.model, 'X.npy'))
X = normalize(X)

y = np.load(os.path.join('/home/ubuntu/data/img_feats', args.dataset, 'train', args.model, 'y.npy'))

# shuffle
p    = np.random.permutation(X.shape[0])
X, y = X[p], y[p]

# subset
X, y = X[:args.sample], y[:args.sample]

X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)

# train/test split
n_class = len(set(y))

n_samples_per_class = 2

# --
# Run

idxs      = np.arange(X.shape[0])
idx_train = pd.Series(idxs).groupby(y).apply(lambda x: x.sample(n_samples_per_class)).values
idx_test  = np.setdiff1d(idxs, idx_train)

svc       = LinearSVC().fit(X[idx_train], y[idx_train])
svc_preds = svc.predict(X[idx_test])
svc_acc   = (y[idx_test] == svc_preds).mean()

y_ss           = y.copy()
y_ss[idx_test] = -1

cas_preds = CASClassifier().fit_predict(X, y_ss)
cas_acc   = (y[idx_test] == cas_preds[idx_test].argmax(axis=-1)).mean() # !! careful about imbalance

print(f'svc_acc={svc_acc} | cas_acc={cas_acc}')

# --


class PrecomputedPPR(nn.Module):
    def __init__(self, ppr):
        super().__init__()
        self.register_buffer('ppr', torch.FloatTensor(ppr))
        
    def forward(self, X, idx):
        return self.ppr[idx] @ X


def calc_A_hat(adj, mode):
    A = adj + torch.eye(adj.shape[0])
    D = A.sum(axis=1)
    D_inv = torch.diag(1 / D.sqrt())
    return D_inv @ A @ D_inv

def exact_ppr(adj, alpha, mode='sym'):
    ralpha  = 1 - alpha
    A_hat   = calc_A_hat(adj, mode=mode)
    A_inner = torch.eye(adj.shape[0]) - (1 - ralpha) * A_hat
    return ralpha * torch.linalg.inv(A_inner)


class PPNP(nn.Module):
    def __init__(self, X, n_class):
        super().__init__()
        self.X       = X
        
        self.A       = nn.Parameter(torch.eye(X.shape[1]))
        self.encoder = nn.Sequential()
        self.output  = nn.Sequential(
            nn.Linear(X.shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, n_class),
        )
    
    def forward(self, idx, ppr, diffuse=True):
        if ppr is None:
            Xp       = X @ self.A
            sim      = Xp @ Xp.T
            thresh   = torch.topk(sim, 10, axis=-1).values[:,-1]
            adj      = sim > thresh
            adj      = (adj | adj.T).float()
            self.ppr = exact_ppr(adj, alpha=0.85)
        else:
            self.ppr = ppr
        
        out = self.encoder(self.X)
        
        if diffuse:
            out = self.ppr[idx] @ out
        else:
            out = out[idx]
        
        out = self.output(out)
        return out





X = torch.FloatTensor(X)
y = torch.LongTensor(y)

n_obs    = X.shape[0]
n_class  = y.max() + 1

sim      = X @ X.T
thresh   = torch.topk(sim, 10, axis=-1).values[:,-1]
adj      = (sim > thresh).float()
adj      = ((adj + adj.T) > 0).float()

alpha = 0.85
ppr   = torch.FloatTensor(exact_ppr(adj, alpha=alpha))
model = PPNP(X=X, n_class=n_class)

lr  = 1e-3
opt = torch.optim.Adam(model.parameters(), lr=lr)

epochs     = 10000
batch_size = 32

loss_hist = []
gen = trange(epochs)
for epoch in gen:
    
    _ppr = ppr if epoch < 2500 else None
    
    out  = model(idx=idx_train, ppr=_ppr, diffuse=epoch > 1000)
    loss = F.cross_entropy(out, y[idx_train])
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    acc = (y[idx_test] == model(idx=idx_test, ppr=model.ppr, diffuse=epoch > 1000).argmax(axis=-1)).float().mean()
    loss_hist.append(float(loss))
    gen.set_postfix(loss=f'{float(loss):0.5f}', acc=f'{float(acc):0.5f}')
    
    

breakpoint()