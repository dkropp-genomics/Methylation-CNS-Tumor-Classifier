"""A small fully connected network that accepts incomplete input (PyTorch, CPU).

Input per probe: two numbers. The standardized beta value (0 when the probe is
not observed) and a flag, 1 if observed, 0 if not. The flag lets the network
tell "missing" from "average", which filling with the mean cannot.

Masked training: every time a training sample is shown, a random share of its
probes is hidden. The share is drawn per sample between `min_fraction` and 1 on
a log scale, so the network practises on 0.1%, 1%, 10% and 100% equally often.
"""
from __future__ import annotations

import math

import numpy as np


class Standardizer:
    """Per-probe mean and SD, learned from the training fold only."""

    def fit(self, Z):
        self.mean_ = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.sd_ = np.maximum(Z.std(axis=0, dtype=np.float64), 1e-3).astype(np.float32)
        return self

    def transform(self, Z):
        return ((Z - self.mean_) / self.sd_).astype(np.float32)


def class_weights(y_idx, n_classes):
    """'Balanced' weights: every class contributes equally to the loss."""
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise SystemExit("ERROR: nn: a class has no training samples")
    return (len(y_idx) / (n_classes * counts)).astype(np.float32)


def build_mlp(n_features, n_classes, hidden=(512, 256), dropout=0.2):
    import torch
    layers, width = [], 2 * n_features            # value + observed flag per probe
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        width = h
    layers.append(torch.nn.Linear(width, n_classes))
    return torch.nn.Sequential(*layers)


def make_input(values, obs):
    """values: standardized betas (n, F); obs: bool (n, F). Returns (n, 2F)."""
    import torch
    flag = obs.to(values.dtype)
    return torch.cat([values * flag, flag], dim=1)


def predict_proba(net, Zs, obs, batch_size=512):
    """Class probabilities for standardized betas `Zs` with observed-mask `obs`."""
    import torch
    was_training = net.training
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, Zs.shape[0], batch_size):
            x = make_input(torch.from_numpy(Zs[i:i + batch_size]),
                           torch.from_numpy(obs[i:i + batch_size]))
            out.append(torch.softmax(net(x), dim=1).numpy())
    net.train(was_training)
    return np.vstack(out)


def train_mlp(Zs, y_idx, n_classes, *, masked, hidden=(512, 256), dropout=0.2, lr=1e-3,
              weight_decay=0.01, batch_size=128, max_epochs=60, eval_every=5,
              min_fraction=0.001, seed=0, n_threads=6, on_checkpoint=None):
    """Train on standardized training-fold betas `Zs` (n, F) and labels 0..K-1.

    `on_checkpoint(epoch, net)` is called every `eval_every` epochs and after
    the last one; that is how held-out scores are recorded along the way.
    Returns the trained network.
    """
    import torch
    torch.set_num_threads(int(n_threads))
    torch.manual_seed(int(seed))
    g = torch.Generator().manual_seed(int(seed))
    n, F = Zs.shape
    X = torch.from_numpy(np.ascontiguousarray(Zs, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_idx, dtype=np.int64))
    w = torch.from_numpy(class_weights(np.asarray(y_idx), n_classes))
    net = build_mlp(F, n_classes, hidden, dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    log_min = math.log(float(min_fraction))
    net.train()
    for epoch in range(1, int(max_epochs) + 1):
        order = torch.randperm(n, generator=g)
        for start in range(0, n, int(batch_size)):
            idx = order[start:start + int(batch_size)]
            xb = X[idx]
            if masked:      # per sample: a coverage share, then a random subset of probes
                share = torch.exp(torch.rand(len(idx), 1, generator=g) * log_min)
                obs = torch.rand(len(idx), F, generator=g) < share
            else:
                obs = torch.ones(len(idx), F, dtype=torch.bool)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(make_input(xb, obs)), y[idx], weight=w)
            loss.backward()
            opt.step()
        if on_checkpoint is not None and (epoch % int(eval_every) == 0 or epoch == max_epochs):
            on_checkpoint(epoch, net)
    return net
