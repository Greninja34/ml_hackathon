"""Small MLP classifier trained on the GPU with PyTorch, exposed with a scikit-learn style API
(fit / predict_proba) so it drops into the same experiment loop as the other models.

Standardisation statistics are computed from the training rows passed to fit() only.
"""
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(64, 32), lr=1e-3, batch_size=4096, max_epochs=40, patience=5,
                 weight_decay=0.0, dropout=0.0, val_frac=0.1, seed=42):
        self.hidden, self.lr, self.batch_size, self.max_epochs = hidden, lr, batch_size, max_epochs
        self.patience, self.weight_decay, self.dropout, self.val_frac, self.seed = patience, weight_decay, dropout, val_frac, seed

    def _net(self, d):
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32, device=dev)
        yt = torch.tensor(y, dtype=torch.float32, device=dev)
        perm = torch.tensor(rng.permutation(len(X)), device=dev)
        nv = int(len(X) * self.val_frac)
        vi, ti = perm[:nv], perm[nv:]
        self.net_ = self._net(X.shape[1]).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        self.loss_curve_, self.val_loss_curve_ = [], []
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.max_epochs):
            self.net_.train()
            sh = ti[torch.randperm(len(ti), device=dev)]
            tot = 0.0
            for s in range(0, len(sh), self.batch_size):
                b = sh[s:s + self.batch_size]
                opt.zero_grad()
                loss = lossf(self.net_(Xt[b]).squeeze(1), yt[b])
                loss.backward(); opt.step()
                tot += loss.item() * len(b)
            self.loss_curve_.append(tot / len(ti))
            self.net_.eval()
            with torch.no_grad():
                vl = lossf(self.net_(Xt[vi]).squeeze(1), yt[vi]).item()
            self.val_loss_curve_.append(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1]); self.device_ = dev
        return self

    def predict_proba(self, X):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 200000):
                xt = torch.tensor((X[s:s + 200000] - self.mu_) / self.sd_, dtype=torch.float32, device=self.device_)
                out.append(torch.sigmoid(self.net_(xt).squeeze(1)).cpu().numpy())
        p = np.concatenate(out)
        return np.column_stack([1 - p, p])
