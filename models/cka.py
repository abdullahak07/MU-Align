# models/cka.py

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    one_n = K.new_ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def _gram_linear(X: torch.Tensor) -> torch.Tensor:
    return X @ X.t()


def cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float()
    Y = Y.float()

    K = _gram_linear(X)
    L = _gram_linear(Y)

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    hsic = (Kc * Lc).sum()
    norm_x = torch.sqrt((Kc * Kc).sum() + 1e-12)
    norm_y = torch.sqrt((Lc * Lc).sum() + 1e-12)

    return float((hsic / (norm_x * norm_y)).item())


def _extract_from_batch(batch):
    """
    Supports:
    - dict batch: expects keys image/question
    - tuple/list batch: tries standard positions
    """
    if isinstance(batch, dict):
        img = batch.get("image", None)
        q = batch.get("question", batch.get("q", None))
        return img, q

    if isinstance(batch, (tuple, list)):
        # (img, q, y, extras) or similar
        if len(batch) >= 2:
            return batch[0], batch[1]

    raise RuntimeError(f"CKA: Unsupported batch type {type(batch)}")


def _collect_embeddings(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50,   # limit so CKA doesn't become huge on full VQA
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    img_list, txt_list, fused_list = [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break

            imgs, questions = _extract_from_batch(batch)
            if imgs is None or questions is None:
                raise RuntimeError("CKA: Could not extract image/question from batch.")

            imgs = imgs.to(device, non_blocking=True)
            questions = questions.to(device, non_blocking=True)

            # Your VQAModel supports return_dict=True
            out = model(imgs, questions, return_dict=True)

            img_emb = out["img_emb"].detach().cpu()
            txt_emb = out["txt_emb"].detach().cpu()
            fused_emb = out["fused_emb"].detach().cpu()

            img_list.append(img_emb)
            txt_list.append(txt_emb)
            fused_list.append(fused_emb)

    if not img_list:
        raise RuntimeError("CKA: No embeddings collected (empty loader?)")

    return torch.cat(img_list, dim=0), torch.cat(txt_list, dim=0), torch.cat(fused_list, dim=0)


def compute_and_plot_cka_pairwise(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    save_dir: str,
    tag: str = "",
) -> Dict[str, float]:
    os.makedirs(save_dir, exist_ok=True)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    try:
        img_emb, txt_emb, fused_emb = _collect_embeddings(model, val_loader, device)
    except Exception as e:
        print(f"[CKA] Warning: could not collect embeddings for tag '{tag}' due to: {e}. Skipping CKA.")
        return {}

    if img_emb.numel() == 0 or txt_emb.numel() == 0 or fused_emb.numel() == 0:
        print(f"[CKA] Warning: empty embeddings for tag '{tag}'. Skipping CKA.")
        return {}

    cka_img_img = cka_linear(img_emb, img_emb)
    cka_txt_txt = cka_linear(txt_emb, txt_emb)
    cka_fus_fus = cka_linear(fused_emb, fused_emb)

    cka_img_txt = cka_linear(img_emb, txt_emb)
    cka_fus_img = cka_linear(fused_emb, img_emb)
    cka_fus_txt = cka_linear(fused_emb, txt_emb)

    stats = {
        "Image-Image": cka_img_img,
        "Text-Text": cka_txt_txt,
        "Fused-Fused": cka_fus_fus,
        "Image-Text": cka_img_txt,
        "Fused-Image": cka_fus_img,
        "Fused-Text": cka_fus_txt,
    }

    labels = ["Image", "Text", "Fused"]
    mat = np.zeros((3, 3), dtype=float)
    mat[0, 0] = cka_img_img
    mat[1, 1] = cka_txt_txt
    mat[2, 2] = cka_fus_fus
    mat[0, 1] = mat[1, 0] = cka_img_txt
    mat[0, 2] = mat[2, 0] = cka_fus_img
    mat[1, 2] = mat[2, 1] = cka_fus_txt

    plt.figure(figsize=(4, 3))
    plt.imshow(mat, vmin=0.0, vmax=1.0)
    plt.colorbar(label="Linear CKA")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.title(f"CKA ({tag})" if tag else "CKA")

    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fname = f"cka_{tag}.png" if tag else "cka.png"
    out_path = os.path.join(save_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[CKA] Saved heatmap to {out_path}")
    return stats
