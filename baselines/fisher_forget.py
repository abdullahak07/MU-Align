import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.vqa_model import VQAModel
from models.utils import (
    collect_outputs,
    standard_forget_retain_acc,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# 1. Fisher Information estimation on forget set
# -------------------------------------------------------

@torch.no_grad()
def _zero_like_params(model):
    """Helper: create a dict of zero tensors with same shape as params."""
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = torch.zeros_like(p, device=p.device)
    return out


def estimate_fisher(model, forget_loader):
    """
    Estimate diagonal Fisher on the FORGET set.

    F_i ≈ E[(∂ log p(y|x,θ) / ∂θ_i)^2]

    Implementation:
      - model in TRAIN mode (important for GRU/cuDNN backward)
      - for each batch:
          * compute CE loss
          * backward
          * accumulate grad^2 into Fisher buffer
    """
    model.train()  # <-- CRITICAL: RNN backward requires train mode
    ce = nn.CrossEntropyLoss()

    fisher = _zero_like_params(model)
    num_batches = 0

    for batch in forget_loader:
        img = batch["image"].to(DEVICE)
        q   = batch["question"].to(DEVICE)
        y   = batch["label"].to(DEVICE)

        model.zero_grad(set_to_none=True)

        logits, _ = model(img, q)
        loss = ce(logits, y)
        loss.backward()

        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                fisher[name] += p.grad.detach() ** 2

        num_batches += 1

    if num_batches == 0:
        return fisher

    # average over batches
    with torch.no_grad():
        for name in fisher:
            fisher[name] /= float(num_batches)

    return fisher


# -------------------------------------------------------
# 2. EWC-style Fisher forgetting
# -------------------------------------------------------

def ewc_penalty(model, fisher, theta_star, lam):
    """
    EWC penalty: sum_i F_i * (θ_i - θ*_i)^2
    """
    reg = torch.tensor(0.0, device=DEVICE)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name not in fisher:
            continue
        reg = reg + (fisher[name] * (p - theta_star[name]) ** 2).sum()
    return lam * reg


def snapshot_params(model):
    """
    Take a parameter snapshot θ* as a plain dict of tensors.
    """
    theta = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            theta[name] = p.detach().clone()
    return theta


# -------------------------------------------------------
# 3. Top-level Fisher Forget baseline
# -------------------------------------------------------

def run_fisher_forget(
    train_loader_full,
    train_loader_forget,
    val_loader_full,
    vocab_size,
    num_answers,
    epochs_orig: int = 3,
    epochs_unlearn: int = 3,
    lam_ewc: float = 50.0,
):
    """
    Fisher Forget baseline:

      1) Train ORIG_FF on full data D_r ∪ D_f
      2) Estimate Fisher on D_f
      3) EWC-style forgetting phase:
           minimize L_ce(D_r) + λ * Σ F_i (θ_i - θ*_i)^2
         where θ* are params after step 1.

    Returns:
      model_ff : final model after forgetting
      out_ff   : outputs dict on val_full (for metrics)
    """
    # ---------- 1) Train original model on full data ----------
    print("[FISHER-ORIG] Training base model on D_r ∪ D_f")
    model = VQAModel(vocab_size=vocab_size, num_answers=num_answers).to(DEVICE)
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs_orig + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for batch in train_loader_full:
            img = batch["image"].to(DEVICE)
            q   = batch["question"].to(DEVICE)
            y   = batch["label"].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits, _ = model(img, q)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total += img.size(0)

        avg_loss = total_loss / max(1, total)
        avg_acc  = total_correct / max(1, total)
        print(f"[FISHER-ORIG] Epoch {ep}/{epochs_orig} loss={avg_loss:.4f} acc={avg_acc:.4f}")

    # Snapshot θ* for EWC
    theta_star = snapshot_params(model)

    # ---------- 2) Estimate Fisher on FORGET set ----------
    print("[FISHER] Estimating Fisher on forget set")
    fisher = estimate_fisher(model, train_loader_forget)

    # ---------- 3) EWC-style forgetting ----------
    print("[FISHER-UNL] Running EWC forgetting phase")
    opt = optim.Adam(model.parameters(), lr=5e-4)

    for ep in range(1, epochs_unlearn + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for batch in train_loader_full:
            img = batch["image"].to(DEVICE)
            q   = batch["question"].to(DEVICE)
            y   = batch["label"].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits, _ = model(img, q)
            loss_ce = ce(logits, y)
            loss_ewc = ewc_penalty(model, fisher, theta_star, lam_ewc)
            loss = loss_ce + loss_ewc

            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            total += img.size(0)

        avg_loss = total_loss / max(1, total)
        print(f"[FISHER-UNL] Epoch {ep}/{epochs_unlearn} loss={avg_loss:.4f}")

    # ---------- 4) Evaluate on VAL ----------
    out_ff = collect_outputs(model, val_loader_full)
    fa, ra = standard_forget_retain_acc(out_ff)
    print(f"FISHER VAL ForgetAcc={fa:.4f}, RetainAcc={ra:.4f}")

    return model, out_ff
