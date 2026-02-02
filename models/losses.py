# models/losses.py
import torch
import torch.nn.functional as F


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1)


def _coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
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


def mu_align_loss(
    logits: torch.Tensor,
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    fused_emb: torch.Tensor,
    y: torch.Tensor,
    forget_mask: torch.Tensor,
    retain_mask: torch.Tensor,
    lambda_unif: float = 1.0,
    lambda_cssa: float = 0.1,
    lambda_cmfc: float = 0.1,
    lambda_coral: float = 0.02,
    label_smoothing: float = 0.05,
    # --- low-FPR hardening (forget-only) ---
    lambda_fconf: float = 0.0,
    conf_cap: float = 0.0,
    lambda_fmarg: float = 0.0,
    margin_cap: float = 0.0,
    # --- embedding/logit tail suppression (forget-only) ---
    lambda_fembnorm: float = 0.0,
    embnorm_cap: float = 0.0,
    lambda_flogitnorm: float = 0.0,
    logitnorm_cap: float = 0.0,
):
    """
    IMPORTANT:
    - Returns *differentiable* tensors (no .detach()) so callers can backprop.
    - Output count is EXACTLY 10:
        (total, ce, unif, fconf, fmarg, femb, flog, cssa, cmfc, coral)
    """

    device = logits.device

    ce = torch.tensor(0.0, device=device)
    unif = torch.tensor(0.0, device=device)
    cssa = torch.tensor(0.0, device=device)
    cmfc = torch.tensor(0.0, device=device)
    coral = torch.tensor(0.0, device=device)

    fconf = torch.tensor(0.0, device=device)
    fmarg = torch.tensor(0.0, device=device)
    femb = torch.tensor(0.0, device=device)
    flog = torch.tensor(0.0, device=device)

    # Retain utility: CE on retain only
    if retain_mask is not None and retain_mask.sum() > 0:
        ce = F.cross_entropy(
            logits[retain_mask],
            y[retain_mask],
            label_smoothing=float(label_smoothing),
        )

    # Forget: uniform predictions
    if forget_mask is not None and forget_mask.sum() > 0 and float(lambda_unif) > 0.0:
        lf = logits[forget_mask]
        pf = F.softmax(lf, dim=1)
        k = pf.size(1)
        u = torch.full_like(pf, 1.0 / max(k, 1))
        unif = F.kl_div(torch.log(pf + 1e-12), u, reduction="batchmean")

    # Retain alignment regularizers
    if retain_mask is not None and retain_mask.sum() > 0 and float(lambda_cssa) > 0.0:
        sim = _cos_sim(img_emb[retain_mask], txt_emb[retain_mask])
        cssa = (1.0 - sim).mean()

    if retain_mask is not None and retain_mask.sum() > 0 and float(lambda_cmfc) > 0.0:
        mi = img_emb[retain_mask].mean(dim=0)
        mt = txt_emb[retain_mask].mean(dim=0)
        cmfc = F.mse_loss(mi, mt)

    if retain_mask is not None and retain_mask.sum() > 0 and float(lambda_coral) > 0.0:
        coral = _coral_loss(img_emb[retain_mask], txt_emb[retain_mask])

    # Forget-only tail suppression
    if forget_mask is not None and forget_mask.sum() > 0:
        if float(lambda_fconf) > 0.0 and float(conf_cap) > 0.0:
            max_prob_f, margin_f = _max_prob_and_margin(logits[forget_mask])
            fconf = F.relu(max_prob_f - float(conf_cap)).pow(2).mean()
            if float(lambda_fmarg) > 0.0 and float(margin_cap) > 0.0:
                fmarg = F.relu(margin_f - float(margin_cap)).pow(2).mean()

        if float(lambda_fembnorm) > 0.0 and float(embnorm_cap) > 0.0:
            n = _l2_norm(fused_emb[forget_mask])
            femb = F.relu(n - float(embnorm_cap)).pow(2).mean()

        if float(lambda_flogitnorm) > 0.0 and float(logitnorm_cap) > 0.0:
            ln = _l2_norm(logits[forget_mask])
            flog = F.relu(ln - float(logitnorm_cap)).pow(2).mean()

    total = (
        ce
        + (float(lambda_unif) * unif)
        + (float(lambda_cssa) * cssa)
        + (float(lambda_cmfc) * cmfc)
        + (float(lambda_coral) * coral)
        + (float(lambda_fconf) * fconf)
        + (float(lambda_fmarg) * fmarg)
        + (float(lambda_fembnorm) * femb)
        + (float(lambda_flogitnorm) * flog)
    )

    return (
        total,
        ce,
        unif,
        fconf,
        fmarg,
        femb,
        flog,
        cssa,
        cmfc,
        coral,
    )
