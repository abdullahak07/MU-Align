# mu_align/baselines/sisa.py

import math
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Subset

from models.utils import DEVICE, train_epoch_ce, collect_outputs
from models.vqa_model import VQAModel
from models.utils import standard_forget_retain_acc


def run_sisa(train_ds_full, val_loader_full, vocab_size, num_answers, num_shards=4, epochs=3, lr=1e-3):
    """
    SISA-like sharded unlearning:
      - split training data into shards,
      - train separate model per shard,
      - average logits at test time.
    """
    ce_loss = nn.CrossEntropyLoss()
    n = len(train_ds_full)
    shard_size = math.ceil(n / num_shards)
    models = []

    for s in range(num_shards):
        start = s * shard_size
        end = min((s + 1) * shard_size, n)
        if start >= end:
            continue
        shard_ds = Subset(train_ds_full, list(range(start, end)))
        shard_loader = torch.utils.data.DataLoader(
            shard_ds, batch_size=64, shuffle=True
        )
        m = VQAModel(vocab_size, num_answers).to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=lr)
        print(f"[SISA] Training shard {s+1}/{num_shards} ({end-start} samples)")
        for ep in range(epochs):
            loss, acc = train_epoch_ce(m, shard_loader, opt, ce_loss)
            print(f"  Shard {s+1} Epoch {ep+1}/{epochs} loss={loss:.4f} acc={acc:.4f}")
        models.append(m)

    # Ensemble evaluation: average logits
    @torch.no_grad()
    def collect_ensemble(models, loader):
        all_logits, all_embeds, all_labels, all_forget, all_qtype = [], [], [], [], []
        for batch in loader:
            img = batch["image"].to(DEVICE)
            q = batch["question"].to(DEVICE)
            logits_sum = None
            emb_sum = None
            for m in models:
                m.eval()
                logits, emb = m(img, q)
                if logits_sum is None:
                    logits_sum = logits
                    emb_sum = emb
                else:
                    logits_sum += logits
                    emb_sum += emb
            logits_avg = logits_sum / len(models)
            emb_avg = emb_sum / len(models)
            all_logits.append(logits_avg.cpu())
            all_embeds.append(emb_avg.cpu())
            all_labels.append(batch["label"])
            all_forget.append(batch["forget"])
            all_qtype.append(batch["qtype"])
        return {
            "logits": torch.cat(all_logits),
            "embeds": torch.cat(all_embeds),
            "labels": torch.cat(all_labels),
            "forget": torch.cat(all_forget),
            "qtype": torch.cat(all_qtype),
        }

    out_val = collect_ensemble(models, val_loader_full)
    fa, ra = standard_forget_retain_acc(out_val)
    print(f"SISA VAL ForgetAcc={fa:.4f}, RetainAcc={ra:.4f}")
    return models, out_val
