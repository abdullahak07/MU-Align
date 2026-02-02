# attacks/classic_attacks.py

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Helpers: robust conversion
# ---------------------------------------------------------------------
def _as_torch(x, dtype=None):
    """
    Convert x (torch.Tensor | np.ndarray | list) -> torch.Tensor on CPU.
    Attacks run on CPU. No gradients.
    """
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
    """Convert x (torch.Tensor | np.ndarray | list) -> np.ndarray."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------
# Helper: unpack outputs robustly
# ---------------------------------------------------------------------
def _unpack_outputs(outputs):
    """
    Supports:
      1) Dict: {"logits": [N,C], "labels": [N] (optional), "embeds": [N,D] (optional)}
      2) Tensor/ndarray: logits only [N,C]
    """
    if isinstance(outputs, dict):
        logits = outputs.get("logits", None)
        labels = outputs.get("labels", None)
        embeds = outputs.get("embeds", None)
    else:
        logits, labels, embeds = outputs, None, None

    if logits is None:
        raise KeyError("outputs must contain 'logits' (or be logits tensor/ndarray).")

    logits_t = _as_torch(logits, dtype=torch.float32)
    labels_t = _as_torch(labels, dtype=torch.long) if labels is not None else None
    embeds_t = _as_torch(embeds, dtype=torch.float32) if embeds is not None else None
    return logits_t, labels_t, embeds_t


# ---------------------------------------------------------------------
# Core scoring helper
# ---------------------------------------------------------------------
def _loss_conf_margin_entropy(outputs):
    """
    Returns:
        nll      : [N]
        max_conf : [N]
        margin   : [N]
        entropy  : [N]
        probs    : [N, C]
        logits   : [N, C]
    """
    logits, labels, _ = _unpack_outputs(outputs)

    if labels is None:
        labels = logits.argmax(dim=1)

    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        max_conf, _ = probs.max(dim=1)

        one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
        true_prob = (probs * one_hot).sum(dim=1)
        nll = -torch.log(true_prob + 1e-12)

        top2, _ = probs.topk(2, dim=1)
        margin = (top2[:, 0] - top2[:, 1]).abs()

        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

    return (
        nll.cpu().numpy(),
        max_conf.cpu().numpy(),
        margin.cpu().numpy(),
        entropy.cpu().numpy(),
        probs.cpu().numpy(),
        logits.cpu().numpy(),
    )


# ---------------------------------------------------------------------
# Generic AUC helper
# ---------------------------------------------------------------------
def mia_auc_from_scores(scores, member_mask, attacker_best: bool = True):
    """
    scores      : [N], higher => more likely member
    member_mask : [N] bool/int
    attacker_best: if True, return max(AUC, 1-AUC) so 0.5 is ideal.
    """
    scores = _as_numpy(scores).reshape(-1)
    mm = _as_numpy(member_mask).astype(bool).reshape(-1)
    if scores.shape[0] != mm.shape[0]:
        raise ValueError(f"Scores length {scores.shape[0]} != member_mask length {mm.shape[0]}")
    y = mm.astype(int)
    auc = float(roc_auc_score(y, scores))
    return float(max(auc, 1.0 - auc)) if attacker_best else auc


# ---------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------
def attack1_loss_threshold(outputs, member_mask):
    nll, _, _, _, _, _ = _loss_conf_margin_entropy(outputs)
    return mia_auc_from_scores(-nll, member_mask)


def attack2_conf_threshold(outputs, member_mask):
    _, conf, _, _, _, _ = _loss_conf_margin_entropy(outputs)
    return mia_auc_from_scores(conf, member_mask)


def attack3_entropy_threshold(outputs, member_mask):
    _, _, _, entropy, _, _ = _loss_conf_margin_entropy(outputs)
    return mia_auc_from_scores(-entropy, member_mask)


def attack4_logit_lr(outputs, member_mask, seed: int = 0):
    _, _, _, _, probs, logits = _loss_conf_margin_entropy(outputs)

    logits = np.asarray(logits)
    probs = np.asarray(probs)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    feats = np.concatenate([logits, probs], axis=1)
    y = _as_numpy(member_mask).astype(bool).astype(int).reshape(-1)

    clf = LogisticRegression(max_iter=4000, solver="liblinear", random_state=int(seed))
    clf.fit(feats, y)
    scores = clf.predict_proba(feats)[:, 1]
    return mia_auc_from_scores(scores, y)


def attack5_rep_knn(outputs, member_mask, k=5):
    logits, _, embeds = _unpack_outputs(outputs)

    emb = embeds.detach().cpu().numpy() if embeds is not None else logits.detach().cpu().numpy()
    y = _as_numpy(member_mask).astype(bool).astype(int).reshape(-1)

    knn = KNeighborsClassifier(n_neighbors=int(k))
    knn.fit(emb, y)
    scores = knn.predict_proba(emb)[:, 1]
    return mia_auc_from_scores(scores, y)


def attack7_margin_label_only(outputs, member_mask):
    _, _, margin, _, _, _ = _loss_conf_margin_entropy(outputs)
    return mia_auc_from_scores(margin, member_mask)


def attack8_shadow_feature(outputs, member_mask, seed: int = 0):
    """
    Shadow-style feature attack on [loss, conf, margin, entropy].
    """
    nll, conf, margin, entropy, _, _ = _loss_conf_margin_entropy(outputs)
    feats = np.stack([nll, conf, margin, entropy], axis=1)
    y = _as_numpy(member_mask).astype(bool).astype(int).reshape(-1)

    clf = LogisticRegression(max_iter=4000, solver="liblinear", random_state=int(seed))
    clf.fit(feats, y)
    scores = clf.predict_proba(feats)[:, 1]
    return mia_auc_from_scores(scores, y)


def run_all_classic_attacks(outputs, member_mask):
    a1 = attack1_loss_threshold(outputs, member_mask)
    a2 = attack2_conf_threshold(outputs, member_mask)
    a3 = attack3_entropy_threshold(outputs, member_mask)
    a4 = attack4_logit_lr(outputs, member_mask)
    a5 = attack5_rep_knn(outputs, member_mask)
    a7 = attack7_margin_label_only(outputs, member_mask)
    a8 = attack8_shadow_feature(outputs, member_mask)

    return {
        "A1_loss_threshold": float(a1),
        "A2_conf_threshold": float(a2),
        "A3_entropy_threshold": float(a3),
        "A4_logit_lr": float(a4),
        "A5_rep_knn": float(a5),
        "A7_margin_label": float(a7),
        "A8_shadow_feature": float(a8),
    }
