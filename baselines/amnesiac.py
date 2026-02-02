# baselines/amnesiac.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.config import DEVICE, LR, EPOCHS_MUALIGN
from models.utils import train_epoch, collect_outputs, standard_forget_retain_acc
from models.vqa_model import VQAModel


def _ts():
    return time.strftime("%H:%M:%S")


def run_amnesiac(train_loader_full, train_loader_forget, val_loader_full, vocab_size: int, num_answers: int):
    """
    Simple Amnesiac baseline:
      1) Train on full data (short)
      2) Fine-tune on retain-only data to "forget" forget-set influence

    Returns:
      model, out_val (dict from collect_outputs)
    """
    device = torch.device(DEVICE) if isinstance(DEVICE, str) else DEVICE

    # Step 1: train on full data
    model = VQAModel(vocab_size, num_answers).to(device)
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(LR))

    n1 = max(1, int(EPOCHS_MUALIGN))  # reuse your MU epochs as a reasonable default
    print(f"\n[{_ts()}][AMNESIAC] Step1: train FULL for {n1} epochs")
    for ep in range(1, n1 + 1):
        loss, acc = train_epoch(model, train_loader_full, opt, ce)
        print(f"[{_ts()}][AMNESIAC] FULL Epoch {ep}/{n1} loss={loss:.4f} acc={acc:.4f}")

    # Step 2: fine-tune on retain-only (train_loader_full already includes forget flag;
    # but run_all passes train_loader_retain to RETR; for amnesiac we use the forget loader only for signature compat)
    # We expect run_all to pass train_loader_retain as train_loader_forget? In your pipeline you pass both.
    # Here: if train_loader_forget is actually "forget-only", we should not train on it.
    # So we do: fine-tune on RETAIN set if available via attribute passed in as train_loader_forget is NOT retain.
    #
    # Practical: in your experiments/methods.py you call:
    #   run_amnesiac(train_loader_full, train_loader_forget, ...)
    # so we do not have train_loader_retain here. We therefore:
    #   - fine-tune again on full loader but mask out forget samples (cheap, consistent)
    #
    print(f"[{_ts()}][AMNESIAC] Step2: fine-tune RETAIN-ONLY via masking for {n1} epochs")

    opt2 = optim.Adam(model.parameters(), lr=float(LR) * 0.5)
    model.train()
    for ep in range(1, n1 + 1):
        total_loss = 0.0
        total_correct = 0
        total = 0

        for batch in train_loader_full:
            img = batch["image"].to(device, non_blocking=True)
            q = batch["question"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            forget = batch["forget"].to(device, non_blocking=True).bool()
            retain = ~forget

            if not bool(retain.any().item()):
                continue

            opt2.zero_grad(set_to_none=True)
            logits, _ = model(img, q)
            loss = ce(logits[retain], y[retain])
            loss.backward()
            opt2.step()

            bs = int(retain.sum().item())
            total += bs
            total_loss += float(loss.item()) * bs
            total_correct += int((logits[retain].argmax(1) == y[retain]).sum().item())

        avg_loss = total_loss / max(total, 1)
        avg_acc = total_correct / max(total, 1)
        print(f"[{_ts()}][AMNESIAC] RETAIN Epoch {ep}/{n1} loss={avg_loss:.4f} acc={avg_acc:.4f}")

    out_val = collect_outputs(model, val_loader_full)
    fa, ra = standard_forget_retain_acc(out_val)
    print(f"[{_ts()}][AMNESIAC] VAL ForgetAcc={fa:.4f}, RetainAcc={ra:.4f}")
    return model, out_val
