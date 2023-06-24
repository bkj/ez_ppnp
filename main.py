#!/usr/env/python

"""
  main.py
  
  !! Need to test sklearn methods ... last I checked they don't really work, but who knows?
  
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
  parser.add_argument('--dataset', type=str, default='food101')
  parser.add_argument('--model',   type=str, default='openai/clip-vit-large-patch14')
  parser.add_argument('--sample',  type=int, default=5_000)
  parser.add_argument('--seed',    type=int, default=234)
  args = parser.parse_args()
  
  assert args.dataset in _DATASETS
  
  return args


args = parse_args()
np.random.seed(args.seed)

X = np.load(os.path.join('/home/ubuntu/data/img_feats', args.dataset, 'train', args.model, 'X.npy'))
X = normalize(X)

y = np.load(os.path.join('/home/ubuntu/data/img_feats', args.dataset, 'train', args.model, 'y.npy'))

# train/test split
n_class = len(set(y))

# subset
#
# X, y = X[:args.sample], y[:args.sample]
# --
idx_keep = pd.Series(np.arange(X.shape[0])).groupby(y).apply(lambda x: x.sample(args.sample // n_class)).values
X, y = X[idx_keep], y[idx_keep]

X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)

n_samples_per_class = 10

# --
# Run

idxs      = np.arange(X.shape[0])
idx_train = pd.Series(idxs).groupby(y).apply(lambda x: x.sample(n_samples_per_class)).values
idx_valid  = np.setdiff1d(idxs, idx_train)

svc       = LinearSVC().fit(X[idx_train], y[idx_train])
svc_preds = svc.predict(X[idx_valid])
svc_acc   = (y[idx_valid] == svc_preds).mean()

y_ss           = y.copy()
y_ss[idx_valid] = -1

cas_preds = CASClassifier().fit_predict(X, y_ss)
cas_acc   = (y[idx_valid] == cas_preds[idx_valid].argmax(axis=-1)).mean() # !! careful about imbalance

print(f'svc_acc={svc_acc} | cas_acc={cas_acc}')

# breakpoint()

# # <<
# # Why are these so bad?

# from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# lp       = LabelPropagation(kernel='knn', n_neighbors=10).fit(X, y_ss)
# lp_preds = lp.predict(X)
# (y[idx_valid] == lp_preds[idx_valid]).mean()

# ls       = LabelSpreading(kernel='knn', n_neighbors=10).fit(X, y_ss)
# ls_preds = ls.predict(X)
# (y[idx_valid] == ls_preds[idx_valid]).mean()

# # >>

# --

def calc_A_hat(adj):
    A     = adj + torch.eye(adj.shape[0], device=adj.device)
    D_inv = torch.diag(1 / A.sum(axis=1).sqrt())
    return D_inv @ A @ D_inv

def exact_ppr(adj, alpha):
    ralpha  = 1 - alpha
    A_hat   = calc_A_hat(adj)
    A_inner = torch.eye(adj.shape[0], device=adj.device) - (1 - ralpha) * A_hat
    return ralpha * torch.linalg.inv(A_inner)

def partial_ppr(adj, alpha, idx):
    n_nodes = adj.shape[0]
    
    ralpha  = 1 - alpha
    A_hat   = calc_A_hat(adj)
    A_inner = torch.eye(n_nodes, device=adj.device) - (1 - ralpha) * A_hat
        
    signals = torch.zeros((n_nodes, len(idx)), device=adj.device)
    signals[(idx, torch.arange(len(idx)))] = 1
    
    return ralpha * torch.linalg.solve(A_inner, signals).T

class PPNP(nn.Module):
    def __init__(self, X, ppr0, n_class):
        super().__init__()
        self.X       = X
        self.ppr0    = ppr0
        self.A       = nn.Parameter(torch.eye(X.shape[1]))
        self.encoder = nn.Sequential()
        self.output  = nn.Linear(X.shape[1], n_class)
        # self.output  = nn.Sequential(
        #     nn.Linear(X.shape[1], 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_class),
        # )
        # self.output = nn.Parameter(torch.randn(X.shape[1], n_class))
    
    def forward(self, idx, beta=None):
        # if beta > 0.5:
        #     Xp     = self.X @ self.A
        #     sim    = Xp @ Xp.T
        #     thresh = torch.topk(sim, 10, axis=-1).values[:,-1]
        #     adj    = sim > thresh
        #     adj    = (adj | adj.T).float()
        #     # ppr    = exact_ppr(adj, alpha=0.85)[idx]
        #     ppr    = partial_ppr(adj, alpha=0.85, idx=idx)
        # else:
        ppr = self.ppr0[idx]
        
        out = self.encoder(self.X)
        out = beta * ppr @ out + ((1 - beta) * out[idx])
        out = self.output(out)
        return out



X = torch.FloatTensor(X).cuda()
y = torch.LongTensor(y).cuda()

n_obs    = X.shape[0]
n_class  = y.max() + 1

sim      = X @ X.T
thresh   = torch.topk(sim, 10, axis=-1).values[:,-1]
adj      = sim > thresh
adj      = (adj | adj.T).float()

ppr0  = exact_ppr(adj, alpha=0.85)
model = PPNP(X=X, ppr0=ppr0, n_class=n_class).cuda()

lr  = 1e-2
opt = torch.optim.Adam(model.parameters(), lr=lr)

epochs     = 10000
batch_size = 32

loss_hist = []
gen = trange(epochs)
for epoch in gen:
    beta = epoch / 2500 if epoch < 2500 else 1
    if epoch == 2500: print(f'2500 = {train_acc:0.5f} | {valid_acc:0.5f}')
    
    out  = model(idx=idx_train, beta=beta)
    loss = F.cross_entropy(out, y[idx_train])
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if epoch % 25 == 0:
        train_acc = (y[idx_train] == model(idx=idx_train, beta=beta).argmax(axis=-1)).float().mean()
        valid_acc = (y[idx_valid] == model(idx=idx_valid, beta=beta).argmax(axis=-1)).float().mean()
    
    loss_hist.append(float(loss))
    gen.set_postfix(
        loss=f'{float(loss):0.5f}', 
        train_acc=f'{float(valid_acc):0.5f}', 
        valid_acc=f'{float(valid_acc):0.5f}', 
        beta=f'{float(beta):0.5f}'
    )