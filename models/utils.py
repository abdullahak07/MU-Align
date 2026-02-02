# models/utils.py

import os
import json
import random
import time
from collections import Counter

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

# ---------------------------------------------------------------------
# Config import (keep this compatible with your codebase)
# ---------------------------------------------------------------------
try:
    from experiments.config import (
        FORGET_ANSWER,
        TOP_K_ANSWERS,
        MAX_QUESTION_LEN,
        BATCH_SIZE,
        DEVICE,
        NUM_WORKERS,
        PIN_MEMORY,
        PERSISTENT_WORKERS,
        PREFETCH_FACTOR,
        PRIVACY_ON_TRAIN,
        PRIVACY_ON_EVAL,
        PRIVACY_EMB_CLIP,
        PRIVACY_EMB_NOISE_STD,
    )
except Exception:
    FORGET_ANSWER = "yes"
    TOP_K_ANSWERS = 1000
    MAX_QUESTION_LEN = 16
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = 2
    PRIVACY_ON_TRAIN = False
    PRIVACY_ON_EVAL = False
    PRIVACY_EMB_CLIP = 0.0
    PRIVACY_EMB_NOISE_STD = 0.0

# Project policy: no global eval-time privacy.
_EVAL_PRIVACY_DISABLED = True


def _ts():
    return time.strftime("%H:%M:%S")


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Privacy helper (kept for backward compatibility)
# ============================================================
def privatize_embedding(z: torch.Tensor, clip: float = 5.0, noise_std: float = 0.05) -> torch.Tensor:
    if z is None:
        return None
    if clip is not None and clip > 0:
        norm = z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = (clip / norm).clamp(max=1.0)
        z = z * scale
    if noise_std is not None and noise_std > 0:
        z = z + noise_std * torch.randn_like(z)
    return z


# ============================================================
# VQAv2 helpers
# ============================================================
def coco_image_path(split: str, image_id: int) -> str:
    return f"COCO_{split}_{image_id:012d}.jpg"


def tokenize(text: str):
    return text.lower().strip().split()


def encode_question(text, vocab, max_len=MAX_QUESTION_LEN):
    toks = tokenize(text)
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


def question_type(ans: str) -> int:
    if ans in {"yes", "no"}:
        return 0
    if any(ch.isdigit() for ch in ans):
        return 1
    return 2


def load_vqav2_split(q_path, a_path):
    """
    Returns list of dicts: {image_id, question_id, question, answer}
    Uses majority vote for answers where available.
    """
    with open(q_path, "r", encoding="utf-8") as f:
        q_raw = json.load(f)
    with open(a_path, "r", encoding="utf-8") as f:
        a_raw = json.load(f)

    q_list = q_raw["questions"] if isinstance(q_raw, dict) and "questions" in q_raw else q_raw
    a_list = a_raw["annotations"] if isinstance(a_raw, dict) and "annotations" in a_raw else a_raw

    q_by_id = {q["question_id"]: q for q in q_list}

    examples = []
    for ann in a_list:
        qid = ann["question_id"]
        img_id = ann["image_id"]
        q_txt = q_by_id[qid]["question"]

        if "answers" in ann and ann["answers"]:
            answers = [a["answer"] for a in ann["answers"]]
            majority = Counter(answers).most_common(1)[0][0]
        else:
            majority = ann.get("answer", "")

        examples.append({"image_id": img_id, "question_id": qid, "question": q_txt, "answer": majority})
    return examples


def build_answer_vocab(examples, top_k=TOP_K_ANSWERS):
    cnt = Counter(e["answer"] for e in examples)
    most_common = [a for a, _ in cnt.most_common(int(top_k))]
    ans2idx = {a: i for i, a in enumerate(most_common)}
    idx2ans = {i: a for a, i in ans2idx.items()}
    return ans2idx, idx2ans


def build_question_vocab(examples, min_freq=2):
    cnt = Counter()
    for e in examples:
        cnt.update(tokenize(e["question"]))

    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in cnt.items():
        if c >= int(min_freq) and w not in vocab:
            vocab[w] = len(vocab)

    inv = {i: w for w, i in vocab.items()}
    return vocab, inv


# ============================================================
# Dataset
# ============================================================
class VQADataset(Dataset):
    def __init__(
        self,
        examples,
        img_dir,
        ans2idx,
        word2idx,
        split_name="train2014",
        transform=None,
        forget_answer=FORGET_ANSWER,
        max_len=MAX_QUESTION_LEN,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self._forget_flags = []

        kept = 0
        missing_img = 0
        skipped_oov = 0

        for e in examples:
            if e["answer"] not in ans2idx:
                skipped_oov += 1
                continue

            img_file = coco_image_path(split_name, e["image_id"])
            img_path = os.path.join(img_dir, img_file)
            if not os.path.exists(img_path):
                missing_img += 1
                continue

            label = ans2idx[e["answer"]]
            q_enc = encode_question(e["question"], word2idx, max_len=max_len)
            forget_flag = 1 if e["answer"] == forget_answer else 0
            qtype = question_type(e["answer"])

            self.data.append(
                {
                    "img_path": img_path,
                    "q_tokens": q_enc,
                    "label": label,
                    "forget": forget_flag,
                    "qtype": qtype,
                }
            )
            self._forget_flags.append(forget_flag)
            kept += 1

        self._forget_flags = np.asarray(self._forget_flags, dtype=np.int64)
        print(
            f"[{_ts()}][VQADataset] built split={split_name} kept={kept} "
            f"(skipped_oov={skipped_oov}, missing_img={missing_img}) | "
            f"forget={int(self._forget_flags.sum())} retain={kept-int(self._forget_flags.sum())}"
        )

    def __len__(self):
        return len(self.data)

    def get_forget_mask_numpy(self) -> np.ndarray:
        return self._forget_flags.astype(bool)

    def __getitem__(self, idx):
        d = self.data[idx]
        img = Image.open(d["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        q = torch.tensor(d["q_tokens"], dtype=torch.long)
        y = torch.tensor(d["label"], dtype=torch.long)
        forget = torch.tensor(d["forget"], dtype=torch.long)
        qtype = torch.tensor(d["qtype"], dtype=torch.long)
        return {"image": img, "question": q, "label": y, "forget": forget, "qtype": qtype}


# ============================================================
# Training (exports expected by baselines)
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        q = batch["question"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits, _ = model(img, q)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += img.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_epoch_ce(model, loader, opt, ce_loss=None):
    if ce_loss is None:
        ce_loss = torch.nn.CrossEntropyLoss()
    return train_epoch(model, loader, opt, ce_loss)


# ============================================================
# Output collection (NO global eval-time privacy)
# ============================================================
@torch.no_grad()
def collect_outputs(model, loader):
    model.eval()
    device = next(model.parameters()).device

    all_logits, all_embeds, all_labels, all_forget, all_qtype = [], [], [], [], []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        q = batch["question"].to(device, non_blocking=True)

        logits, emb = model(img, q)

        if (not _EVAL_PRIVACY_DISABLED) and PRIVACY_ON_EVAL:
            emb = privatize_embedding(emb, float(PRIVACY_EMB_CLIP), float(PRIVACY_EMB_NOISE_STD))
            logits = model.classify_from_fused(emb)

        all_logits.append(logits.detach().cpu())
        all_embeds.append(emb.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
        all_forget.append(batch["forget"].detach().cpu())
        all_qtype.append(batch["qtype"].detach().cpu())

    return {
        "logits": torch.cat(all_logits) if all_logits else torch.empty(0),
        "embeds": torch.cat(all_embeds) if all_embeds else torch.empty(0),
        "labels": torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long),
        "forget": torch.cat(all_forget) if all_forget else torch.empty(0, dtype=torch.long),
        "qtype": torch.cat(all_qtype) if all_qtype else torch.empty(0, dtype=torch.long),
    }


def standard_forget_retain_acc(outputs):
    y = outputs["labels"]
    preds = outputs["logits"].argmax(1)
    forget_mask = outputs["forget"] == 1
    retain_mask = ~forget_mask

    forget_acc = (preds[forget_mask] == y[forget_mask]).float().mean().item() if forget_mask.any() else 0.0
    retain_acc = (preds[retain_mask] == y[retain_mask]).float().mean().item() if retain_mask.any() else 0.0
    return forget_acc, retain_acc


# ============================================================
# Privacy-cache helpers expected by run_all / attacks
# MEMBERS   = train_forget
# NONMEMBER = val_forget ONLY
# ============================================================
def build_member_nonmember_masks(train_forget_loader, val_forget_loader):
    n_train = len(train_forget_loader.dataset)
    n_val = len(val_forget_loader.dataset)
    member = np.zeros(n_train + n_val, dtype=bool)
    member[:n_train] = True
    nonmember = ~member
    return member, nonmember


def _apply_mu_forget_rules_if_present(model, embeds: torch.Tensor, forget_mask: torch.Tensor) -> torch.Tensor:
    """
    If MU_ALIGN stored its forget-privacy knobs on the model, apply them here so
    cached outputs reflect the same behavior.
    """
    if embeds is None:
        return embeds
    if forget_mask is None or (not torch.is_tensor(forget_mask)):
        return embeds

    noise_std = getattr(model, "_mu_forget_noise_std", None)
    clip_norm = getattr(model, "_mu_forget_clip_norm", None)
    scrub_alpha = getattr(model, "_mu_forget_scrub_alpha", None)

    if noise_std is None and clip_norm is None and scrub_alpha is None:
        return embeds

    out = embeds

    # clip (forget only)
    if clip_norm is not None and float(clip_norm) > 0 and forget_mask.any():
        z = out[forget_mask]
        norm = z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = (float(clip_norm) / norm).clamp(max=1.0)
        z = z * scale
        out = out.clone()
        out[forget_mask] = z

    # noise (forget only)
    if noise_std is not None and float(noise_std) > 0 and forget_mask.any():
        out = out.clone()
        out[forget_mask] = out[forget_mask] + float(noise_std) * torch.randn_like(out[forget_mask])

    # deterministic scrub (forget only)
    if scrub_alpha is not None and float(scrub_alpha) > 0 and forget_mask.any():
        a = float(scrub_alpha)
        out = out.clone()
        out[forget_mask] = (1.0 - a) * out[forget_mask] + a * torch.zeros_like(out[forget_mask])

    return out


@torch.no_grad()
def collect_forget_outputs_concat(model, train_forget_loader, val_forget_loader, device=None):
    """
    Correct for MIA:
      - members   = train_forget
      - nonmember = val_forget
    Returns dict with keys: logits, labels, embeds
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_logits, all_embeds, all_labels = [], [], []

    for loader in (train_forget_loader, val_forget_loader):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            questions = batch["question"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits, embeds = model(images, questions)

            # apply MU_ALIGN cache-time forget rules if present
            # Here, all samples in these loaders are "forget" by construction.
            forget_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=embeds.device)
            embeds = _apply_mu_forget_rules_if_present(model, embeds, forget_mask)

            # recompute logits if your model supports classify_from_fused
            if hasattr(model, "classify_from_fused"):
                logits = model.classify_from_fused(embeds)

            if (not _EVAL_PRIVACY_DISABLED) and PRIVACY_ON_EVAL:
                embeds = privatize_embedding(embeds, float(PRIVACY_EMB_CLIP), float(PRIVACY_EMB_NOISE_STD))
                logits = model.classify_from_fused(embeds)

            all_logits.append(logits.detach().cpu())
            all_embeds.append(embeds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if len(all_logits) == 0:
        return {
            "logits": torch.empty(0, 0),
            "embeds": torch.empty(0, 0),
            "labels": torch.empty(0, dtype=torch.long),
        }

    return {
        "logits": torch.cat(all_logits, 0),
        "embeds": torch.cat(all_embeds, 0),
        "labels": torch.cat(all_labels, 0),
    }
