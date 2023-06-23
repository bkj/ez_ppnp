# import faiss
import numpy as np
import scipy.sparse as sp
from tqdm import trange
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# from time import time

def label_propagation(A, Y, alpha, niter=10, clip=(0, 1)):
  Z = Y.copy()
  for _ in range(niter):
    Z = alpha * (A @ Z)
    Z = Z + (1 - alpha) * Y
    Z = Z.clip(*clip)
  
  return Z


class CASClassifier:
  def __init__(self, k=10, alpha=0.85, use_faiss=False):
    self.k         = k
    self.alpha     = alpha
    self.use_faiss = use_faiss
    
    self.DAD = None
    self.AD  = None

  def _make_adj(self, X):
    # Compute KNN graph (quick and dirty; binary; symmetrized -- can do other things here)
    
    n_obs = X.shape[0]
    sim   = X @ X.T
    
    # Adjacency matrix (knn)
    thresh = np.partition(sim, -self.k, axis=-1)[:,-self.k]
    adj    = (sim > thresh).astype(np.float32)
    
    # Symmetric sparse matrix
    row, col = adj.nonzero()
    val      = adj[(row, col)]
    
    adj = sp.csr_matrix((val, (row, col)), shape=(n_obs, n_obs))  
    adj = (adj + adj.T).astype(np.float32)
    
    # Compute normalization matrix
    d    = np.asarray(adj.sum(axis=0)).squeeze()
    assert d.min() > 0
    Dinv = sp.diags(1 / d)

    # Compute normalized adjacency matrices
    DAD = np.sqrt(Dinv) @ adj @ np.sqrt(Dinv)
    AD  = adj @ Dinv

    return DAD, AD
  
  def fit_predict(self, X, y, cache_adj=False):
    X = X.astype(np.float32)
      
    DAD, AD = self._make_adj(X)
  
    n_obs   = X.shape[0]
    n_class = y.max() + 1
    assert n_class == np.unique(y[y != -1]).shape[0]
    
    idx_train = np.where(y != -1)[0]
    idx_test  = np.where(y == -1)[0]
    
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    
    # Fit shallow model
    clf    = LinearSVC().fit(X_train, y_train)
    Z_orig = clf.decision_function(X)
    if n_class == 2:
      Z_orig = np.column_stack([-Z_orig, Z_orig])
      
    Z_orig  = np.exp(Z_orig) / np.exp(Z_orig).sum(axis=-1, keepdims=True)
    
    # Residual targets
    Y_resid = np.zeros((n_obs, n_class))
    Y_resid[(idx_train, y_train)] = 1
    Y_resid[idx_train] -= Z_orig[idx_train]
    
    # Spread residuals
    resid = label_propagation(AD, Y_resid, alpha=self.alpha, clip=(-1, 1)) # !! Why AD?
    
    # Corrected predictions
    num   = np.abs(Y_resid[idx_train]).sum() / idx_train.shape[0]
    denom = np.abs(resid).sum(axis=-1, keepdims=True)
    scale = (num / denom)
    scale[denom == 0]   = 1
    scale[scale > 1000] = 1
    print(scale.shape)
    
    Z_corrected = Z_orig + scale * resid
    
    # Corrected targets
    Y_corrected = Z_corrected.copy()
    Y_corrected[idx_train] = 0
    Y_corrected[(idx_train, y_train)] = 1
    
    # Spread predictions
    Z_smoothed = label_propagation(DAD, Y_corrected, alpha=self.alpha, clip=(0, 1)) # !! Why DAD?
    return Z_smoothed

