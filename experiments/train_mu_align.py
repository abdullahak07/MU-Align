# experiments/train_mu_align.py
import torch
import torch.nn.functional as F
from models.losses import mu_align_loss


@torch.no_grad()
def evaluate_forget_retain(model, val_full_loader, device):
    model.eval()
    total_forget_correct = total_forget = 0
    total_retain_correct = total_retain = 0

    for batch in val_full_loader:
        img = batch["image"].to(device, non_blocking=True)
        q = batch["question"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        forget = batch["forget"].to(device, non_blocking=True).bool()
        retain = ~forget

        logits, _ = model(img, q)
        pred = logits.argmax(1)

        if bool(forget.any().item()):
            total_forget_correct += (pred[forget] == y[forget]).sum().item()
            total_forget += int(forget.sum().item())
        if bool(retain.any().item()):
            total_retain_correct += (pred[retain] == y[retain]).sum().item()
            total_retain += int(retain.sum().item())

    forget_acc = total_forget_correct / max(total_forget, 1)
    retain_acc = total_retain_correct / max(total_retain, 1)
    return forget_acc, retain_acc


def train_mu_align(
    model_mu,
    model_teacher_orig,   # frozen ORIG teacher OR a retrained teacher (depending on your call-site)
    train_full_loader,
    val_full_loader,
    device,
    epochs: int = 8,
    lr: float = 5e-4,

    # loss weights (defaults tuned toward low-FPR)
    lambda_unif: float = 6.0,
    lambda_cssa: float = 0.01,
    lambda_cmfc: float = 0.01,
    lambda_coral: float = 0.02,
    label_smoothing: float = 0.05,

    # KD on retain
    lambda_kd: float = 0.6,
    kd_temp: float = 2.0,

    # forget privacy perturbation in fused space (light)
    forget_emb_noise: float = 0.10,
    forget_emb_clip: float = 1.5,

    # caps (SCRUB-like)
    lambda_fconf: float = 1.0,
    conf_cap: float = 0.35,
    lambda_fmarg: float = 0.5,
    margin_cap: float = 1.0,
    lambda_fembnorm: float = 0.4,
    embnorm_cap: float = 6.0,
    lambda_flogitnorm: float = 0.4,
    logitnorm_cap: float = 8.0,

    # novelty: tail-risk
    lambda_tail: float = 1.5,
    tail_alpha: float = 0.10,
):
    """
    MU_ALIGN++ trainer (low-FPR hardened + tail-risk objective).
    Returns:
      model_mu, {"forget_acc":..., "retain_acc":...}
    """
    opt = torch.optim.Adam(model_mu.parameters(), lr=float(lr))

    # freeze teacher
    model_teacher_orig.eval()
    for p in model_teacher_orig.parameters():
        p.requires_grad = False

    print("\n=== Training MU_ALIGN++ (low-FPR hardened + CVaR-tail) ===")
    print(
        f"[MU_ALIGN++] lambdas: unif={lambda_unif} cssa={lambda_cssa} cmfc={lambda_cmfc} coral={lambda_coral} "
        f"fconf={lambda_fconf} fmarg={lambda_fmarg} femb={lambda_fembnorm} flog={lambda_flogitnorm} "
        f"tail={lambda_tail}(alpha={tail_alpha}) kd={lambda_kd}(T={kd_temp}) ls={label_smoothing}"
    )
    print(
        f"[MU_ALIGN++] caps: conf_cap={conf_cap} margin_cap={margin_cap} embnorm_cap={embnorm_cap} logitnorm_cap={logitnorm_cap} "
        f"forget_noise={forget_emb_noise} forget_clip={forget_emb_clip}"
    )

    for ep in range(1, int(epochs) + 1):
        model_mu.train()

        seen = 0
        sum_loss = 0.0
        sums = {k: 0.0 for k in ["CE","UNIF","CSSA","CMFC","CORAL","FCONF","FMARG","FEMBN","FLOGN","TAIL"]}

        for batch in train_full_loader:
            img = batch["image"].to(device, non_blocking=True)
            q = batch["question"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            forget = batch["forget"].to(device, non_blocking=True).bool()
            retain = ~forget

            # forward: request embeddings
            out = model_mu(img, q, return_dict=True)
            logits = out["logits"]
            img_emb = out["img_emb"]
            txt_emb = out["txt_emb"]
            fused = out["fused_emb"]

            # forget-only privacy perturbation in fused space
            fused = model_mu.apply_forget_privacy(
                fused,
                forget_mask=forget,
                noise_std=float(forget_emb_noise),
                clip_norm=float(forget_emb_clip),
            )
            logits = model_mu.classify_from_fused(fused)

            # loss
            base_loss, metrics = mu_align_loss(
                logits=logits,
                img_emb=img_emb,
                txt_emb=txt_emb,
                fused_emb=fused,
                y=y,
                forget_mask=forget,
                retain_mask=retain,
                lambda_unif=float(lambda_unif),
                lambda_cssa=float(lambda_cssa),
                lambda_cmfc=float(lambda_cmfc),
                lambda_coral=float(lambda_coral),
                label_smoothing=float(label_smoothing),
                lambda_fconf=float(lambda_fconf),
                conf_cap=float(conf_cap),
                lambda_fmarg=float(lambda_fmarg),
                margin_cap=float(margin_cap),
                lambda_fembnorm=float(lambda_fembnorm),
                embnorm_cap=float(embnorm_cap),
                lambda_flogitnorm=float(lambda_flogitnorm),
                logitnorm_cap=float(logitnorm_cap),
                lambda_tail=float(lambda_tail),
                tail_alpha=float(tail_alpha),
            )

            # KD on retain only
            kd = torch.tensor(0.0, device=device)
            if float(lambda_kd) > 0.0 and bool(retain.any().item()):
                with torch.no_grad():
                    t_logits, _ = model_teacher_orig(img, q)
                s = logits[retain] / float(kd_temp)
                t = t_logits[retain] / float(kd_temp)
                kd = F.kl_div(
                    F.log_softmax(s, dim=1),
                    F.softmax(t, dim=1),
                    reduction="batchmean"
                ) * (float(kd_temp) ** 2)

            loss = base_loss + float(lambda_kd) * kd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(img.size(0))
            seen += bs
            sum_loss += float(loss.item()) * bs

            # aggregate metrics (detach already)
            for k in sums.keys():
                sums[k] += float(metrics[k].item()) * bs

        # epoch log
        denom = max(seen, 1)
        msg = (
            f"[MU_ALIGN++] Epoch {ep}/{epochs} "
            f"loss={sum_loss/denom:.4f} "
            f"CE={sums['CE']/denom:.4f} "
            f"UNIF={sums['UNIF']/denom:.4f} "
            f"FCONF={sums['FCONF']/denom:.4f} "
            f"FMARG={sums['FMARG']/denom:.4f} "
            f"FEMBN={sums['FEMBN']/denom:.4f} "
            f"FLOGN={sums['FLOGN']/denom:.4f} "
            f"TAIL={sums['TAIL']/denom:.4f} "
            f"CSSA={sums['CSSA']/denom:.4f} "
            f"CMFC={sums['CMFC']/denom:.4f} "
            f"CORAL={sums['CORAL']/denom:.4f} "
            f"KD={float(kd.item()):.4f}"
        )
        print(msg)

    fa, ra = evaluate_forget_retain(model_mu, val_full_loader, device)
    print(f"[MU_ALIGN++] VAL ForgetAcc={fa:.4f}, RetainAcc={ra:.4f}")
    return model_mu, {"forget_acc": float(fa), "retain_acc": float(ra)}
