# mu_align/baselines/multidelete.py

import torch
import torch.optim as optim
import torch.nn as nn

from experiments.config import DEVICE, LR, EPOCHS_RETR, EPOCHS_MUALIGN, EPOCHS_ORIG  # pick what the baseline uses
from models.utils import train_epoch, collect_outputs, standard_forget_retain_acc

from models.vqa_model import VQAModel
from models.utils import standard_forget_retain_acc


def run_multidelete(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers, epochs=3, lr=1e-3):
    """
    MultiDelete-style (highly simplified):
      - train base model,
      - simulate multiple deletion requests by repeatedly reweighting loss
        on forget samples.
    """
    ce_loss = nn.CrossEntropyLoss()
    model = VQAModel(vocab_size, num_answers).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for batch in train_loader_full:
            img = batch["image"].to(DEVICE)
            q = batch["question"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            forget = batch["forget"].to(DEVICE)

            logits, _ = model(img, q)
            loss = ce_loss(logits, y)

            # amplify loss for forget samples to push them away
            if forget.any():
                loss_f = ce_loss(logits[forget == 1], y[forget == 1])
                loss = loss + 2.0 * loss_f

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total += img.size(0)
        print(f"[MultiDelete] Epoch {ep+1}/{epochs} loss={total_loss/total:.4f} acc={total_correct/total:.4f}")

    out_val = collect_outputs(model, val_loader_full)
    fa, ra = standard_forget_retain_acc(out_val)
    print(f"MultiDelete VAL ForgetAcc={fa:.4f}, RetainAcc={ra:.4f}")
    return model, out_val
