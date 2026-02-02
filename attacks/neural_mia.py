# attacks/neural_mia.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def _as_torch(x, dtype=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unpack(outputs):
    if isinstance(outputs, dict):
        logits = outputs.get("logits", None)
        labels = outputs.get("labels", None)
        embeds = outputs.get("embeds", None)
    else:
        logits, labels, embeds = outputs, None, None

    if logits is None:
        raise KeyError("neural_mia: outputs must contain 'logits' or be logits directly.")

    logits = _as_torch(logits, dtype=torch.float32)
    labels = _as_torch(labels, dtype=torch.long) if labels is not None else None
    embeds = _as_torch(embeds, dtype=torch.float32) if embeds is not None else None
    return logits, labels, embeds


@torch.no_grad()
def _build_features(outputs):
    logits, _, embeds = _unpack(outputs)

    probs = torch.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    top2, _ = probs.topk(2, dim=1)
    margin = (top2[:, 0] - top2[:, 1]).abs()
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

    logit_mean = logits.mean(dim=1)
    logit_std = logits.std(dim=1)

    feats = [conf, margin, entropy, logit_mean, logit_std]

    if embeds is not None and embeds.numel() > 0:
        emb_norm = embeds.norm(p=2, dim=1)
        feats.append(emb_norm)

    X = torch.stack(feats, dim=1)
    return X.cpu().numpy()


class _MIANet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def neural_mia_auc(outputs, member_mask, epochs=25, lr=1e-3, seed=0, attacker_best: bool = True):
    X = _build_features(outputs)
    y = _as_numpy(member_mask).astype(bool).astype(int).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"neural_mia: X rows {X.shape[0]} != y {y.shape[0]}")

    if len(np.unique(y)) < 2:
        return 0.5

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    torch.manual_seed(int(seed))

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    model = _MIANet(X_t.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        s = model(X_t)
        loss = bce(s, y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_t)).cpu().numpy()

    auc = float(roc_auc_score(y, scores))
    return float(max(auc, 1.0 - auc)) if attacker_best else auc
