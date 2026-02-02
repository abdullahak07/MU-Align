# mu_align/experiments/evaluate_all.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from models.utils import (
    set_seed,
    load_vqav2_split,
    build_answer_vocab,
    build_question_vocab,
    VQADataset,
    make_transform,
    build_loaders,
    DEVICE,
    standard_forget_retain_acc,
)
from models.vqa_model import VQAModel
from models.losses import mu_align_loss
from models.cka import pairwise_cka_dict, save_cka_heatmap

from attacks.classic_attacks import (
    attack1_loss_threshold,
    attack2_conf_threshold,
    attack3_entropy_threshold,
    attack4_logit_lr,
    attack5_rep_knn,
    attack7_margin_label_only,
    attack8_shadow_feature,
    attack9_attribute_inference,
    attack10_property_inference,
)
from attacks.neural_mia import run_neural_mia
from attacks.query_adapt_mia import run_query_adapt_mia
from attacks.inversion_attack import run_inversion_attack

from baselines.amnesiac import run_amnesiac
from baselines.fisher_forget import run_fisher_forget
from baselines.scrub import run_scrub
from baselines.sisa import run_sisa
from baselines.multidelete import run_multidelete


def run_evaluation(vqav2_root, save_dir="results"):
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Load splits & subsample
    # -----------------------------
    train_q = os.path.join(
        vqav2_root,
        "Train questions",
        "v2_Questions_Train_mscoco",
        "v2_OpenEnded_mscoco_train2014_questions.json",
    )
    val_q = os.path.join(
        vqav2_root,
        "Train questions",
        "v2_Questions_Val_mscoco",
        "v2_OpenEnded_mscoco_val2014_questions.json",
    )
    train_a = os.path.join(
        vqav2_root,
        "Train annotations",
        "v2_Annotations_Train_mscoco",
        "v2_mscoco_train2014_annotations.json",
    )
    val_a = os.path.join(
        vqav2_root,
        "Train annotations",
        "v2_Annotations_Val_mscoco",
        "v2_mscoco_val2014_annotations.json",
    )

    print("=== Loading VQAv2 train/val splits ===")
    train_ex = load_vqav2_split(train_q, train_a)
    val_ex = load_vqav2_split(val_q, val_a)

    MAX_TRAIN_SAMPLES = 2000
    MAX_VAL_SAMPLES = 500
    np.random.shuffle(train_ex)
    np.random.shuffle(val_ex)
    train_ex = train_ex[:MAX_TRAIN_SAMPLES]
    val_ex = val_ex[:MAX_VAL_SAMPLES]

    # -----------------------------
    # Build vocab + datasets
    # -----------------------------
    ans2idx, idx2ans = build_answer_vocab(train_ex)
    word2idx, _ = build_question_vocab(train_ex)
    vocab_size = len(word2idx)
    num_answers = len(ans2idx)

    transform = make_transform()
    train_img_dir = os.path.join(vqav2_root, "COCO Images", "train2014")
    val_img_dir = os.path.join(vqav2_root, "COCO Images", "val2014")

    train_ds = VQADataset(train_ex, train_img_dir, ans2idx, word2idx, split_name="train2014", transform=transform)
    val_ds = VQADataset(val_ex, val_img_dir, ans2idx, word2idx, split_name="val2014", transform=transform)

    loaders = build_loaders(train_ds, val_ds)
    train_loader_full = loaders["train_full"]
    train_loader_retain = loaders["train_retain"]
    train_loader_forget = loaders["train_forget"]
    val_loader_full = loaders["val_full"]
    val_loader_forget = loaders["val_forget"]

    # ------------------------------------------------------------------
    # ORIG
    # ------------------------------------------------------------------
    import torch.optim as optim
    import torch.nn as nn

    ce_loss = nn.CrossEntropyLoss()

    print("\n=== Training ORIG (D_r ∪ D_f) ===")
    model_orig = VQAModel(vocab_size, num_answers).to(DEVICE)
    opt = optim.Adam(model_orig.parameters(), lr=1e-3)
    from models.utils import train_epoch_ce, collect_outputs

    for ep in range(5):
        loss, acc = train_epoch_ce(model_orig, train_loader_full, opt, ce_loss)
        print(f"[ORIG] Epoch {ep+1}/5 loss={loss:.4f} acc={acc:.4f}")
    out_orig_val = collect_outputs(model_orig, val_loader_full)
    fa_o, ra_o = standard_forget_retain_acc(out_orig_val)
    print(f"ORIG VAL ForgetAcc={fa_o:.4f}, RetainAcc={ra_o:.4f}")

    # ------------------------------------------------------------------
    # RETR
    # ------------------------------------------------------------------
    print("\n=== Training RETR (D_r only) ===")
    model_retr = VQAModel(vocab_size, num_answers).to(DEVICE)
    opt = optim.Adam(model_retr.parameters(), lr=1e-3)
    for ep in range(5):
        loss, acc = train_epoch_ce(model_retr, train_loader_retain, opt, ce_loss)
        print(f"[RETR] Epoch {ep+1}/5 loss={loss:.4f} acc={acc:.4f}")
    out_retr_val = collect_outputs(model_retr, val_loader_full)
    fa_r, ra_r = standard_forget_retain_acc(out_retr_val)
    print(f"RETR VAL ForgetAcc={fa_r:.4f}, RetainAcc={ra_r:.4f}")

    # ------------------------------------------------------------------
    # MU-Align (GOOD)
    # ------------------------------------------------------------------
    print("\n=== Training MU-Align (GOOD) ===")
    model_mu = VQAModel(vocab_size, num_answers).to(DEVICE)
    model_mu.load_state_dict(model_orig.state_dict())
    opt_mu = optim.Adam(model_mu.parameters(), lr=5e-4)

    for ep in range(3):
        model_mu.train()
        total_loss = total_ce = total_unif = total_cssa = total_cmfc = 0.0
        total = 0
        for batch in train_loader_full:
            img = batch["image"].to(DEVICE)
            q = batch["question"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            forget = batch["forget"].to(DEVICE)
            retain_mask = forget == 0
            forget_mask = forget == 1

            logits, img_emb, txt_emb, fused_emb = model_mu.forward_decomposed(img, q)

            loss, ce_val, unif_val, cssa_val, cmfc_val = mu_align_loss(
                logits,
                img_emb,
                txt_emb,
                fused_emb,
                y,
                forget_mask,
                retain_mask,
                lambda_unif=1.0,
                lambda_cssa=0.1,
                lambda_cmfc=0.1,
            )

            opt_mu.zero_grad()
            loss.backward()
            opt_mu.step()

            bs = img.size(0)
            total_loss += loss.item() * bs
            total_ce += ce_val * bs
            total_unif += unif_val * bs
            total_cssa += cssa_val * bs
            total_cmfc += cmfc_val * bs
            total += bs

        print(
            f"[MU-Align] Epoch {ep+1}/3 loss={total_loss/total:.4f} "
            f"CE={total_ce/total:.4f} UNIF={total_unif/total:.4f} "
            f"CSSA={total_cssa/total:.4f} CMFC={total_cmfc/total:.4f}"
        )

    out_mu_val = collect_outputs(model_mu, val_loader_full)
    fa_m, ra_m = standard_forget_retain_acc(out_mu_val)
    print(f"MU-Align VAL ForgetAcc={fa_m:.4f}, RetainAcc={ra_m:.4f}")

    # ------------------------------------------------------------------
    # Extra baselines (enable/disable as needed)
    # ------------------------------------------------------------------
    print("\n=== Running extra baselines ===")
    model_am, out_am = run_amnesiac(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    model_ff, out_ff = run_fisher_forget(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    model_scrub, out_scrub = run_scrub(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)
    sisa_models, out_sisa = run_sisa(train_ds, val_loader_full, vocab_size, num_answers)
    model_md, out_md = run_multidelete(train_loader_full, train_loader_forget, val_loader_full, vocab_size, num_answers)

    # ------------------------------------------------------------------
    # Threat models / attacks on forgotten samples
    # ------------------------------------------------------------------
    print("\n=== Building member/non-member sets on forget samples ===")
    n_train_f = len(loaders["train_forget_ds"])
    n_val_f = len(loaders["val_forget_ds"])
    member_mask = np.array([1] * n_train_f + [0] * n_val_f, dtype=bool)

    concat_forget_ds = torch.utils.data.ConcatDataset(
        [loaders["train_forget_ds"], loaders["val_forget_ds"]]
    )
    concat_forget_loader = torch.utils.data.DataLoader(
        concat_forget_ds, batch_size=64, shuffle=False
    )

    def collect_forget_outputs(model):
        model.eval()
        all_logits, all_embeds, all_labels = [], [], []
        for batch in concat_forget_loader:
            img = batch["image"].to(DEVICE)
            q = batch["question"].to(DEVICE)
            logits, emb = model(img, q)
            all_logits.append(logits.cpu())
            all_embeds.append(emb.cpu())
            all_labels.append(batch["label"])
        return {
            "logits": torch.cat(all_logits),
            "embeds": torch.cat(all_embeds),
            "labels": torch.cat(all_labels),
        }

    out_orig_f = collect_forget_outputs(model_orig)
    out_retr_f = collect_forget_outputs(model_retr)
    out_mu_f = collect_forget_outputs(model_mu)

    print("\n=== Classic MIA attacks (forgotten samples) ===")
    for name, out in [("ORIG", out_orig_f), ("RETR", out_retr_f), ("MU", out_mu_f)]:
        print(f"\n{name}:")
        a1 = attack1_loss_threshold(out, member_mask)
        a2 = attack2_conf_threshold(out, member_mask)
        a3 = attack3_entropy_threshold(out, member_mask)
        a4 = attack4_logit_lr(out, member_mask)
        a5 = attack5_rep_knn(out, member_mask)
        a7 = attack7_margin_label_only(out, member_mask)
        a8 = attack8_shadow_feature(out, member_mask)
        print(f"  A1 Loss-thres     : {a1:.3f}")
        print(f"  A2 Conf-thres     : {a2:.3f}")
        print(f"  A3 Entropy-thres  : {a3:.3f}")
        print(f"  A4 Logit-LR       : {a4:.3f}")
        print(f"  A5 Rep-kNN        : {a5:.3f}")
        print(f"  A7 Margin (label) : {a7:.3f}")
        print(f"  A8 Shadow feat-LR : {a8:.3f}")

        auc_neural = run_neural_mia(out, member_mask)
        print(f"  Neural-MIA (MLP)  : {auc_neural:.3f}")

        auc_query = run_query_adapt_mia(model_orig, concat_forget_loader, member_mask)
        print(f"  Query-Adapt MIA   : {auc_query:.3f}")

    # Attribute / property / inversion on MU-Align
    print("\n=== Attribute / property / inversion attacks (MU-Align) ===")
    qtypes_forget = torch.cat(
        [
            torch.tensor(
                [train_ds[i]["qtype"].item() for i in loaders["train_forget_idx"]]
            ),
            torch.tensor(
                [
                    val_ds[i]["qtype"].item()
                    for i, m in enumerate(loaders["val_forget_mask"])
                    if m
                ]
            ),
        ]
    )
    n = len(qtypes_forget)
    indices = np.arange(n)
    tr_idx, te_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=qtypes_forget.numpy()
    )
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[tr_idx] = True
    te_mask = ~tr_mask

    acc_attr_mu = attack9_attribute_inference(out_mu_f, qtypes_forget, tr_mask, te_mask)
    print("Attr inference acc (MU-Align):", acc_attr_mu)

    prop_auc_mu = attack10_property_inference(out_mu_val, out_mu_val["forget"])
    print("Property AUC (MU-Align):", prop_auc_mu)

    inv_acc_mu = run_inversion_attack(out_mu_val, out_mu_val["labels"])
    print("Inversion linear acc (MU-Align):", inv_acc_mu)

    # ------------------------------------------------------------------
    # CKA diagnostics + plots
    # ------------------------------------------------------------------
    print("\n=== CKA similarity diagnostics ===")
    def collect_decomposed(model, loader):
        model.eval()
        all_img, all_txt, all_fused = [], [], []
        for batch in loader:
            img = batch["image"].to(DEVICE)
            q = batch["question"].to(DEVICE)
            with torch.no_grad():
                _, img_e, txt_e, fused_e = model.forward_decomposed(img, q)
            all_img.append(img_e.cpu())
            all_txt.append(txt_e.cpu())
            all_fused.append(fused_e.cpu())
        return {
            "img": torch.cat(all_img),
            "txt": torch.cat(all_txt),
            "fused": torch.cat(all_fused),
        }

    feats_orig = collect_decomposed(model_orig, val_loader_full)
    feats_mu = collect_decomposed(model_mu, val_loader_full)

    cka_orig = pairwise_cka_dict(feats_orig)
    cka_mu = pairwise_cka_dict(feats_mu)

    os.makedirs("plots", exist_ok=True)
    save_cka_heatmap(cka_orig, os.path.join("plots", "cka_orig.png"))
    save_cka_heatmap(cka_mu, os.path.join("plots", "cka_mu_align.png"))

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "ORIG": {"forget_acc": fa_o, "retain_acc": ra_o},
        "RETR": {"forget_acc": fa_r, "retain_acc": ra_r},
        "MU-Align": {"forget_acc": fa_m, "retain_acc": ra_m},
    }
    with open(os.path.join(save_dir, "mu_align_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary (VAL forget / retain accuracy) ===")
    print("ORIG   :", (fa_o, ra_o))
    print("RETR   :", (fa_r, ra_r))
    print("MU-Align:", (fa_m, ra_m))
    print(f"Saved summary to {os.path.join(save_dir, 'mu_align_summary.json')}")
