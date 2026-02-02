# experiments/methods.py
import time
from typing import Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.vqa_model import VQAModel

# IMPORTANT:
# We will wrap mu_align_loss to guarantee a stable return signature.
from models.losses import mu_align_loss as _mu_align_loss_raw

from models.utils import (
    train_epoch,
    collect_outputs,
    standard_forget_retain_acc,
)

from baselines.amnesiac import run_amnesiac
from baselines.fisher_forget import run_fisher_forget
from baselines.scrub import run_scrub
from baselines.multidelete import run_multidelete

from .config import (
    DEVICE,
    LR,
    EPOCHS_ORIG,
    EPOCHS_RETR,
    EPOCHS_MUALIGN,

    # MU_ALIGN: main objective
    MUALIGN_LAMBDA_UNIF,
    MUALIGN_LAMBDA_CSSA,
    MUALIGN_LAMBDA_CMFC,
    MUALIGN_LAMBDA_CORAL,
    MUALIGN_LABEL_SMOOTHING,

    # forget-only privacy in fused space (train-time)
    MUALIGN_FORGET_NOISE_STD,
    MUALIGN_FORGET_CLIP_NORM,
    MUALIGN_FORGET_SCRUB_ALPHA,

    # low-FPR hardening (forget-only): logit confidence tails
    MUALIGN_FORGET_CONF_CAP,
    MUALIGN_FORGET_MARGIN_CAP,
    MUALIGN_LAMBDA_FCONF,
    MUALIGN_LAMBDA_FMARG,

    # low-FPR hardening (forget-only): embedding/logit norm tails
    MUALIGN_FORGET_EMBNORM_CAP,
    MUALIGN_LAMBDA_FEMBNORM,
    MUALIGN_FORGET_LOGITNORM_CAP,
    MUALIGN_LAMBDA_FLOGITNORM,

    # KD-to-teacher (retain stabilization)
    MUALIGN_LAMBDA_KD,
    MUALIGN_KD_TEMP,
)


# -----------------------------
# Utilities
# -----------------------------
def _ts():
    return time.strftime("%H:%M:%S")


def make_model(vocab_size: int, num_answers: int) -> VQAModel:
    model = VQAModel(vocab_size, num_answers)
    return model.to(DEVICE)


def _attach_forget_retain_metrics(out: dict) -> dict:
    fa, ra = standard_forget_retain_acc(out)
    out["forget_acc"] = float(fa)
    out["retain_acc"] = float(ra)
    return out


def _require_mu_align_loss_signature(ret: Any) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Normalize mu_align_loss return to EXACTLY 10 tensors:
      (total, ce, unif, fconf, fmarg, femb, flog, cssa, cmfc, coral)
    """
    if not isinstance(ret, (tuple, list)):
        raise ValueError(
            "mu_align_loss returned a non-tuple object. "
            "You likely imported the wrong mu_align_loss implementation."
        )

    if len(ret) == 10:
        return tuple(ret)  # type: ignore

    raise ValueError(
        f"mu_align_loss returned {len(ret)} values, expected 10.\n"
        "Expected: (total, ce, unif, fconf, fmarg, femb, flog, cssa, cmfc, coral)\n"
        "Action: ensure models/losses.py exports the NEW mu_align_loss with 10 outputs, "
        "and that no other module shadows it."
    )


# ---- helpers (mirrors models/losses.py logic, but returns differentiable tensors) ----
def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1)


def _coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.zeros((), device=x.device)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cx = (x.t() @ x) / max((x.size(0) - 1), 1)
    cy = (y.t() @ y) / max((y.size(0) - 1), 1)
    return ((cx - cy) ** 2).mean()


def _max_prob_and_margin(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    max_prob, _ = probs.max(dim=1)
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return max_prob, margin


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x, ord=2, dim=1)


def _safe_params(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten_dot(a_list, b_list) -> torch.Tensor:
    # Sum over params; skip None
    device = None
    out = None
    for ga, gb in zip(a_list, b_list):
        if ga is None or gb is None:
            continue
        device = ga.device
        val = (ga * gb).sum()
        out = val if out is None else out + val
    if out is None:
        return torch.zeros((), device=device or "cpu")
    return out


def _flatten_norm2(g_list) -> torch.Tensor:
    device = None
    out = None
    for g in g_list:
        if g is None:
            continue
        device = g.device
        val = (g * g).sum()
        out = val if out is None else out + val
    if out is None:
        return torch.zeros((), device=device or "cpu")
    return out


# -----------------------------
# ORIG / RETR
# -----------------------------
def run_orig(train_loader_full, val_loader_full, vocab_size: int, num_answers: int):
    model = make_model(vocab_size, num_answers)
    ce_loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(LR))

    for ep in range(1, int(EPOCHS_ORIG) + 1):
        loss, acc = train_epoch(model, train_loader_full, opt, ce_loss)
        print(f"[{_ts()}][ORIG] Epoch {ep}/{EPOCHS_ORIG} loss={loss:.4f} acc={acc:.4f}")

    out_val = collect_outputs(model, val_loader_full)
    out_val = _attach_forget_retain_metrics(out_val)
    print(f"[{_ts()}][ORIG] VAL ForgetAcc={out_val['forget_acc']:.4f}, RetainAcc={out_val['retain_acc']:.4f}")
    return model, out_val


def run_retr(train_loader_retain, val_loader_full, vocab_size: int, num_answers: int):
    model = make_model(vocab_size, num_answers)
    ce_loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(LR))

    for ep in range(1, int(EPOCHS_RETR) + 1):
        loss, acc = train_epoch(model, train_loader_retain, opt, ce_loss)
        print(f"[{_ts()}][RETR] Epoch {ep}/{EPOCHS_RETR} loss={loss:.4f} acc={acc:.4f}")

    out_val = collect_outputs(model, val_loader_full)
    out_val = _attach_forget_retain_metrics(out_val)
    print(f"[{_ts()}][RETR] VAL ForgetAcc={out_val['forget_acc']:.4f}, RetainAcc={out_val['retain_acc']:.4f}")
    return model, out_val


# -----------------------------
# MU_ALIGN
# -----------------------------
def run_mu_align(
    train_loader_full,
    train_loader_forget,   # kept for compatibility; not required by this trainer
    val_loader_full,
    vocab_size: int,
    num_answers: int,
    lambda_unif: Optional[float] = None,
    lambda_cssa: Optional[float] = None,
    lambda_cmfc: Optional[float] = None,
    lambda_coral: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    init_state_dict=None,
    device: Optional[torch.device] = None,
    retrained_teacher=None,   # RETR model as teacher
):
    if device is None:
        device = torch.device(DEVICE) if isinstance(DEVICE, str) else DEVICE

    # Main knobs (allow override, else take config)
    lambda_unif = float(MUALIGN_LAMBDA_UNIF if lambda_unif is None else lambda_unif)
    lambda_cssa = float(MUALIGN_LAMBDA_CSSA if lambda_cssa is None else lambda_cssa)
    lambda_cmfc = float(MUALIGN_LAMBDA_CMFC if lambda_cmfc is None else lambda_cmfc)
    lambda_coral = float(MUALIGN_LAMBDA_CORAL if lambda_coral is None else lambda_coral)
    label_smoothing = float(MUALIGN_LABEL_SMOOTHING if label_smoothing is None else label_smoothing)

    # low-FPR hardening: logit confidence tails
    lambda_fconf = float(MUALIGN_LAMBDA_FCONF)
    lambda_fmarg = float(MUALIGN_LAMBDA_FMARG)
    conf_cap = float(MUALIGN_FORGET_CONF_CAP)
    margin_cap = float(MUALIGN_FORGET_MARGIN_CAP)

    # low-FPR hardening: embedding/logit norm tails
    lambda_fembnorm = float(MUALIGN_LAMBDA_FEMBNORM)
    embnorm_cap = float(MUALIGN_FORGET_EMBNORM_CAP)
    lambda_flogitnorm = float(MUALIGN_LAMBDA_FLOGITNORM)
    logitnorm_cap = float(MUALIGN_FORGET_LOGITNORM_CAP)

    # KD-to-teacher
    lambda_kd = float(MUALIGN_LAMBDA_KD)
    kd_temp = float(MUALIGN_KD_TEMP)

    # Forget-only privacy in fused space (train-time)
    noise_std = float(MUALIGN_FORGET_NOISE_STD)
    clip_norm = float(MUALIGN_FORGET_CLIP_NORM)
    scrub_alpha = float(MUALIGN_FORGET_SCRUB_ALPHA)

    # Grad-ortho toggle (always ON for your current goal)
    GRAD_ORTHO = True
    ORTHO_EPS = 1e-12

    model = make_model(vocab_size, num_answers).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=float(LR) * 0.5)
    num_epochs = int(EPOCHS_MUALIGN)

    teacher_on = (retrained_teacher is not None) and (lambda_kd > 0.0)
    if teacher_on:
        retrained_teacher = retrained_teacher.to(device)
        retrained_teacher.eval()
        for p in retrained_teacher.parameters():
            p.requires_grad = False

    print(f"\n[{_ts()}]=== Training MU-Align (low-FPR hardened + KD-to-teacher + grad-ortho) ===")
    print(
        f"[{_ts()}][MU_ALIGN] lambdas: "
        f"unif={lambda_unif} cssa={lambda_cssa} cmfc={lambda_cmfc} coral={lambda_coral} "
        f"fconf={lambda_fconf} fmarg={lambda_fmarg} fembnorm={lambda_fembnorm} flogitnorm={lambda_flogitnorm} "
        f"kd={lambda_kd} (T={kd_temp}) label_smoothing={label_smoothing}"
    )
    print(
        f"[{_ts()}][MU_ALIGN] caps: "
        f"conf_cap={conf_cap} margin_cap={margin_cap} embnorm_cap={embnorm_cap} logitnorm_cap={logitnorm_cap}"
    )
    print(f"[{_ts()}][MU_ALIGN] forget-privacy: noise_std={noise_std} clip_norm={clip_norm} scrub_alpha={scrub_alpha}")
    print(f"[{_ts()}][MU_ALIGN] teacher={'ON' if teacher_on else 'OFF'}")
    print(f"[{_ts()}][MU_ALIGN] gradient_orthogonalization={'ON' if GRAD_ORTHO else 'OFF'} (forget ⟂ retain)")

    params = _safe_params(model)

    for epoch in range(1, num_epochs + 1):
        model.train()
        t_epoch = time.time()

        total_loss = total_ce = total_unif = 0.0
        total_fconf = total_fmarg = total_femb = total_flog = 0.0
        total_cssa = total_cmfc = total_coral = total_kd = 0.0
        total = 0

        for batch in train_loader_full:
            images = batch["image"].to(device, non_blocking=True)
            questions = batch["question"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            forget_flags = batch["forget"].to(device, non_blocking=True)

            forget_mask = (forget_flags == 1)
            retain_mask = ~forget_mask

            optimizer.zero_grad(set_to_none=True)

            # Forward decomposed
            logits_raw, img_emb, txt_emb, fused_emb = model.forward_decomposed(images, questions)

            # Forget-only privacy perturbation in fused space
            fused_priv = model.apply_forget_privacy(
                fused_emb,
                forget_mask=forget_mask,
                noise_std=noise_std,
                clip_norm=clip_norm,
            )

            # SCRUB-like suppression for forget samples (optional)
            if scrub_alpha > 0.0 and bool(forget_mask.any().item()):
                a = scrub_alpha
                fused_priv = fused_priv.clone()
                fused_priv[forget_mask] = (1.0 - a) * fused_priv[forget_mask]

            # Classify from perturbed fused embedding
            logits = model.classify_from_fused(fused_priv)

            # (A) Compute standard MU_ALIGN loss for logging (safe signature)
            ret = _mu_align_loss_raw(
                logits=logits,
                img_emb=img_emb,
                txt_emb=txt_emb,
                fused_emb=fused_priv,
                y=labels,
                forget_mask=forget_mask,
                retain_mask=retain_mask,
                lambda_unif=lambda_unif,
                lambda_cssa=lambda_cssa,
                lambda_cmfc=lambda_cmfc,
                lambda_coral=lambda_coral,
                label_smoothing=label_smoothing,
                lambda_fconf=lambda_fconf,
                conf_cap=conf_cap,
                lambda_fmarg=lambda_fmarg,
                margin_cap=margin_cap,
                lambda_fembnorm=lambda_fembnorm,
                embnorm_cap=embnorm_cap,
                lambda_flogitnorm=lambda_flogitnorm,
                logitnorm_cap=logitnorm_cap,
            )
            (
                base_loss,
                ce_val,
                unif_val,
                fconf_val,
                fmarg_val,
                femb_val,
                flog_val,
                cssa_val,
                cmfc_val,
                coral_val,
            ) = _require_mu_align_loss_signature(ret)

            # KD term (retain only)
            kd = torch.zeros((), device=device)
            if teacher_on and bool(retain_mask.any().item()):
                with torch.no_grad():
                    t_logits, _ = retrained_teacher(images, questions)
                s = logits[retain_mask] / float(kd_temp)
                t = t_logits[retain_mask] / float(kd_temp)
                kd = F.kl_div(
                    F.log_softmax(s, dim=1),
                    F.softmax(t, dim=1),
                    reduction="batchmean"
                ) * (float(kd_temp) ** 2)

            # (B) Grad-orthogonalized update
            if GRAD_ORTHO:
                # Build differentiable retain objective
                retain_obj = torch.zeros((), device=device)
                if bool(retain_mask.any().item()):
                    retain_obj = retain_obj + F.cross_entropy(
                        logits[retain_mask],
                        labels[retain_mask],
                        label_smoothing=float(label_smoothing),
                    )
                    if lambda_cssa > 0.0:
                        sim = _cos_sim(img_emb[retain_mask], txt_emb[retain_mask])
                        retain_obj = retain_obj + (lambda_cssa * (1.0 - sim).mean())
                    if lambda_cmfc > 0.0:
                        mi = img_emb[retain_mask].mean(dim=0)
                        mt = txt_emb[retain_mask].mean(dim=0)
                        retain_obj = retain_obj + (lambda_cmfc * F.mse_loss(mi, mt))
                    if lambda_coral > 0.0:
                        retain_obj = retain_obj + (lambda_coral * _coral_loss(img_emb[retain_mask], txt_emb[retain_mask]))

                    if teacher_on and lambda_kd > 0.0:
                        retain_obj = retain_obj + (lambda_kd * kd)

                # Build differentiable forget objective
                forget_obj = torch.zeros((), device=device)
                if bool(forget_mask.any().item()):
                    lf = logits[forget_mask]
                    pf = F.softmax(lf, dim=1)
                    k = pf.size(1)
                    u = torch.full_like(pf, 1.0 / max(k, 1))
                    if lambda_unif > 0.0:
                        forget_obj = forget_obj + (lambda_unif * F.kl_div(torch.log(pf + 1e-12), u, reduction="batchmean"))

                    if lambda_fconf > 0.0 and conf_cap > 0.0:
                        max_prob_f, margin_f = _max_prob_and_margin(lf)
                        forget_obj = forget_obj + (lambda_fconf * F.relu(max_prob_f - conf_cap).pow(2).mean())
                        if lambda_fmarg > 0.0 and margin_cap > 0.0:
                            forget_obj = forget_obj + (lambda_fmarg * F.relu(margin_f - margin_cap).pow(2).mean())

                    if lambda_fembnorm > 0.0 and embnorm_cap > 0.0:
                        n = _l2_norm(fused_priv[forget_mask])
                        forget_obj = forget_obj + (lambda_fembnorm * F.relu(n - embnorm_cap).pow(2).mean())

                    if lambda_flogitnorm > 0.0 and logitnorm_cap > 0.0:
                        ln = _l2_norm(lf)
                        forget_obj = forget_obj + (lambda_flogitnorm * F.relu(ln - logitnorm_cap).pow(2).mean())

                # Compute gradients
                if retain_obj.requires_grad:
                    g_r = torch.autograd.grad(retain_obj, params, retain_graph=True, allow_unused=True)
                else:
                    g_r = [None for _ in params]

                if forget_obj.requires_grad:
                    g_f = torch.autograd.grad(forget_obj, params, retain_graph=False, allow_unused=True)
                else:
                    g_f = [None for _ in params]

                dot = _flatten_dot(g_f, g_r)
                n2 = _flatten_norm2(g_r)
                coef = dot / (n2 + ORTHO_EPS)

                # Set parameter grads: g = g_r + (g_f - coef*g_r)
                for p, gr, gf in zip(params, g_r, g_f):
                    if gr is None and gf is None:
                        p.grad = None
                        continue
                    if gr is None:
                        p.grad = gf
                        continue
                    if gf is None:
                        p.grad = gr
                        continue
                    p.grad = gr + (gf - coef * gr)

                optimizer.step()

                # For logging: use combined scalar (not used for backward)
                loss = base_loss + (lambda_kd * kd)

            else:
                # Fallback standard training
                loss = base_loss + (lambda_kd * kd)
                loss.backward()
                optimizer.step()

            # Logging accumulation
            bsz = labels.size(0)
            total_loss += float(loss.item()) * bsz
            total_ce += float(ce_val.item()) * bsz
            total_unif += float(unif_val.item()) * bsz
            total_fconf += float(fconf_val.item()) * bsz
            total_fmarg += float(fmarg_val.item()) * bsz
            total_femb += float(femb_val.item()) * bsz
            total_flog += float(flog_val.item()) * bsz
            total_cssa += float(cssa_val.item()) * bsz
            total_cmfc += float(cmfc_val.item()) * bsz
            total_coral += float(coral_val.item()) * bsz
            total_kd += float(kd.item()) * bsz
            total += bsz

        print(
            f"[{_ts()}][MU_ALIGN] Epoch {epoch}/{num_epochs} DONE in {time.time()-t_epoch:.1f}s | "
            f"loss={total_loss/max(total,1):.4f} CE={total_ce/max(total,1):.4f} UNIF={total_unif/max(total,1):.4f} "
            f"FCONF={total_fconf/max(total,1):.4f} FMARG={total_fmarg/max(total,1):.4f} "
            f"FEMB={total_femb/max(total,1):.4f} FLOG={total_flog/max(total,1):.4f} "
            f"CSSA={total_cssa/max(total,1):.4f} CMFC={total_cmfc/max(total,1):.4f} CORAL={total_coral/max(total,1):.4f} "
            f"KD={total_kd/max(total,1):.4f}"
        )

    out_val = collect_outputs(model, val_loader_full)
    out_val = _attach_forget_retain_metrics(out_val)
    print(f"[{_ts()}][MU_ALIGN] VAL ForgetAcc={out_val['forget_acc']:.4f}, RetainAcc={out_val['retain_acc']:.4f}")
    return model, out_val


# -----------------------------
# Baselines (pass-through)
# -----------------------------
def run_amnesiac_baseline(train_loader_full, train_loader_forget, val_loader_full, vocab_size: int, num_answers: int):
    model, out_val = run_amnesiac(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    out_val = _attach_forget_retain_metrics(out_val) if isinstance(out_val, dict) else out_val
    return model, out_val


def run_fisher_baseline(train_loader_full, train_loader_forget, val_loader_full, vocab_size: int, num_answers: int):
    model, out_val = run_fisher_forget(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    out_val = _attach_forget_retain_metrics(out_val) if isinstance(out_val, dict) else out_val
    return model, out_val


def run_scrub_baseline(train_loader_full, train_loader_forget, val_loader_full, vocab_size: int, num_answers: int):
    model, out_val = run_scrub(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    out_val = _attach_forget_retain_metrics(out_val) if isinstance(out_val, dict) else out_val
    return model, out_val


def run_multidelete_baseline(train_loader_full, train_loader_forget, val_loader_full, vocab_size: int, num_answers: int):
    model, out_val = run_multidelete(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    out_val = _attach_forget_retain_metrics(out_val) if isinstance(out_val, dict) else out_val
    return model, out_val
