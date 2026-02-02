import torch
import torch.optim as optim
import torch.nn.functional as F

from models.vqa_model import VQAModel
from models.utils import build_vqav2_loaders
from .train_orig import eval_forget_retain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_RETR = 5
LR = 1e-3


def train_retr(vqav2_root):
    (
        train_loader_full,
        train_loader_retain,
        train_loader_forget,
        val_loader_full,
        val_loader_forget,
        meta,
    ) = build_vqav2_loaders(vqav2_root)

    vocab_size = len(meta["word2idx"])
    num_answers = len(meta["ans2idx"])

    model = VQAModel(vocab_size, num_answers).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    print("\n=== Training RETR (D_r only) ===")
    for ep in range(1, EPOCHS_RETR + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for batch in train_loader_retain:
            img = batch["image"].to(DEVICE)
            q = batch["question"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            out = model(img, q)
            logits = out["logits_fused"]

            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * img.size(0)
            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total += img.size(0)

        print(
            f"[RETR] Epoch {ep}/{EPOCHS_RETR} "
            f"loss={total_loss/total:.4f} acc={total_correct/total:.4f}"
        )

    model.eval()
    with torch.no_grad():
        fa, ra = eval_forget_retain(model, val_loader_full, meta["val_forget_mask"])
    print(f"RETR VAL ForgetAcc={fa:.4f}, RetainAcc={ra:.4f}")

    return {
        "model": model,
        "meta": meta,
        "loaders": {
            "train_full": train_loader_full,
            "train_retain": train_loader_retain,
            "train_forget": train_loader_forget,
            "val_full": val_loader_full,
            "val_forget": val_loader_forget,
        },
    }
