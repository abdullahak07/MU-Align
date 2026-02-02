# experiments/datasets.py

import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

from models.utils import (
    load_vqav2_split,
    build_answer_vocab,
    build_question_vocab,
    VQADataset,
)

from .config import (
    VQAV2_ROOT,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
    FORGET_ANSWER,
    BATCH_SIZE,
    TOP_K_ANSWERS,
    MAX_QUESTION_LEN,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
)


def _ts():
    return time.strftime("%H:%M:%S")


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vqav2_paths(root):
    train_img = os.path.join(root, "COCO Images", "train2014")
    val_img = os.path.join(root, "COCO Images", "val2014")

    train_q = os.path.join(
        root, "Train questions", "v2_Questions_Train_mscoco",
        "v2_OpenEnded_mscoco_train2014_questions.json"
    )
    val_q = os.path.join(
        root, "Train questions", "v2_Questions_Val_mscoco",
        "v2_OpenEnded_mscoco_val2014_questions.json"
    )

    train_a = os.path.join(
        root, "Train annotations", "v2_Annotations_Train_mscoco",
        "v2_mscoco_train2014_annotations.json"
    )
    val_a = os.path.join(
        root, "Train annotations", "v2_Annotations_Val_mscoco",
        "v2_mscoco_val2014_annotations.json"
    )

    return train_img, val_img, train_q, val_q, train_a, val_a


def _default_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _maybe_subsample(examples, max_n):
    if max_n is None:
        return examples
    return examples[: int(max_n)]


def _make_loader(ds, shuffle: bool):
    """
    Windows-safe DataLoader kwargs:
    - Only set persistent_workers/prefetch_factor when NUM_WORKERS > 0
    - pin_memory only makes sense when CUDA is available
    """
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=int(NUM_WORKERS),
        pin_memory=bool(PIN_MEMORY and torch.cuda.is_available()),
    )

    if int(NUM_WORKERS) > 0:
        kwargs["persistent_workers"] = bool(PERSISTENT_WORKERS)
        kwargs["prefetch_factor"] = int(PREFETCH_FACTOR)

    return DataLoader(ds, **kwargs)


def _build_vqav2(seed, mode_name, max_train, max_val):
    _seed_all(seed)

    print(f"[{_ts()}][DATASET] {mode_name} | Seed={seed}")
    print(f"[{_ts()}][DATASET] VQAV2_ROOT={VQAV2_ROOT}")

    train_img, val_img, train_q, val_q, train_a, val_a = _vqav2_paths(VQAV2_ROOT)

    print(f"[{_ts()}][DATASET] Loading JSONs...")
    train_ex = load_vqav2_split(train_q, train_a)
    val_ex = load_vqav2_split(val_q, val_a)

    random.shuffle(train_ex)
    random.shuffle(val_ex)

    train_ex = _maybe_subsample(train_ex, max_train)
    val_ex = _maybe_subsample(val_ex, max_val)

    print(f"[{_ts()}][DATASET] Using TRAIN={len(train_ex)} | VAL={len(val_ex)}")

    ans2idx, _ = build_answer_vocab(train_ex, top_k=TOP_K_ANSWERS)
    word2idx, _ = build_question_vocab(train_ex)

    transform = _default_transform()

    train_ds = VQADataset(
        train_ex, train_img, ans2idx, word2idx,
        split_name="train2014",
        transform=transform,
        forget_answer=FORGET_ANSWER,
        max_len=MAX_QUESTION_LEN,
    )

    val_ds = VQADataset(
        val_ex, val_img, ans2idx, word2idx,
        split_name="val2014",
        transform=transform,
        forget_answer=FORGET_ANSWER,
        max_len=MAX_QUESTION_LEN,
    )

    train_idx = np.arange(len(train_ds))
    val_idx = np.arange(len(val_ds))

    # SPEED: don't call __getitem__ to get forget flag (it loads images).
    train_forget = train_ds.get_forget_mask_numpy()
    val_forget = val_ds.get_forget_mask_numpy()

    train_retain_ds = Subset(train_ds, train_idx[~train_forget])
    train_forget_ds = Subset(train_ds, train_idx[train_forget])

    val_forget_ds = Subset(val_ds, val_idx[val_forget])
    val_retain_ds = Subset(val_ds, val_idx[~val_forget])

    return (
        _make_loader(train_ds, True),
        _make_loader(train_retain_ds, True),
        _make_loader(train_forget_ds, True),
        _make_loader(val_ds, False),
        _make_loader(val_forget_ds, False),
        _make_loader(val_retain_ds, False),
        {
            "dataset": mode_name,
            "train_n": len(train_ds),
            "val_n": len(val_ds),
            "train_forget": int(train_forget.sum()),
            "val_forget": int(val_forget.sum()),
            "word2idx": word2idx,
            "ans2idx": ans2idx,
        }
    )


def build_vqav2_small(seed):
    return _build_vqav2(seed, "vqa_v2_small", 2000, 500)


def build_vqav2_20k(seed):
    return _build_vqav2(seed, "vqa_v2_20k", MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES)


def build_vqav2_full(seed):
    return _build_vqav2(seed, "vqa_v2_full", None, None)


def build_dataset(name, seed):
    if name == "vqa_v2_small":
        return build_vqav2_small(seed)
    elif name == "vqa_v2_20k":
        return build_vqav2_20k(seed)
    elif name == "vqa_v2_full":
        return build_vqav2_full(seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")
