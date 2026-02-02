# attacks/query_adapt_mia.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def _as_torch(x, dtype=torch.float32):
    """Convert numpy/list/torch -> torch tensor on CPU (no grad)."""
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)
    return t.to(dtype)


def query_adapt_mia_auc(
    outputs,
    member_mask,
    num_noise: int = 5,
    sigma: float = 0.05,
    seed: int | None = 0,
):
    """
    Query-adapted Membership Inference Attack (logit-space variant).

    Idea:
      Add small random noise to logits multiple times and measure instability.
      score_i = std_j( max_softmax(logits_i + noise_j) )

    Supports:
      - outputs as dict with key "logits" (torch or numpy)
      - outputs as logits tensor/ndarray directly

    Returns:
      ROC-AUC (0.5 is ideal / random-guess attacker).
    """
    # --- unpack logits robustly ---
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise KeyError("query_adapt_mia_auc: outputs dict must contain key 'logits'.")
        logits = outputs["logits"]
    else:
        logits = outputs

    logits = _as_torch(logits, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError(f"query_adapt_mia_auc: expected logits shape [N, C], got {tuple(logits.shape)}")

    N, C = logits.shape
    if N == 0 or C == 0:
        return 0.5

    # --- member mask ---
    y = np.asarray(member_mask).astype(bool).reshape(-1)
    M = min(N, y.shape[0])
    logits = logits[:M]
    y = y[:M].astype(int)

    # If labels are degenerate, AUC is undefined; return chance.
    if len(np.unique(y)) < 2:
        return 0.5

    # deterministic noise if desired
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

    # --- compute max-softmax under noisy queries ---
    max_confs = []
    for _ in range(int(num_noise)):
        if gen is None:
            noise = torch.randn_like(logits) * float(sigma)
        else:
            noise = torch.randn(logits.shape, generator=gen, dtype=logits.dtype) * float(sigma)

        perturbed = logits + noise
        probs = F.softmax(perturbed, dim=1)
        max_conf = probs.max(dim=1).values  # [M]
        max_confs.append(max_conf.unsqueeze(0))

    max_confs = torch.cat(max_confs, dim=0)      # [num_noise, M]
    scores = max_confs.std(dim=0).cpu().numpy()  # [M]

    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        return 0.5

    return float(auc)
