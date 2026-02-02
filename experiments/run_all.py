import json
import os
import time
from collections import defaultdict

import numpy as np

from .config import (
    SAVE_ROOT,
    DATASETS,
    SEEDS,
    METHODS,
)
from .datasets import build_dataset
from .methods import (
    run_orig,
    run_retr,
    run_mu_align,
    run_amnesiac_baseline,
    run_fisher_baseline,
    run_scrub_baseline,
    run_multidelete_baseline,
)


def _ts():
    return time.strftime("%H:%M:%S")


def _mean_std(x):
    x = [v for v in x if v is not None and (not (isinstance(v, float) and np.isnan(v)))]
    if len(x) == 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=0))}


def run_pipeline():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # results[dataset][method][seed] = dict(metrics...)
    results = defaultdict(lambda: defaultdict(dict))

    for dataset_name in DATASETS:
        for seed in SEEDS:
            print(f"\n[{_ts()}]==============================")
            print(f"[{_ts()}] DATASET={dataset_name} SEED={seed}")
            print(f"[{_ts()}]==============================")

            (
                train_full_loader,
                train_retain_loader,
                train_forget_loader,
                val_full_loader,
                val_forget_loader,
                val_retain_loader,
                meta
            ) = build_dataset(dataset_name, seed)

            vocab_size = len(meta["word2idx"])
            num_answers = len(meta["ans2idx"])

            # ORIG
            orig_model = None
            orig_out = None
            if "ORIG" in METHODS:
                orig_model, orig_out = run_orig(train_full_loader, val_full_loader, vocab_size, num_answers)
                results[dataset_name]["ORIG"][seed] = {
                    "forget_acc": float(orig_out.get("forget_acc")),
                    "retain_acc": float(orig_out.get("retain_acc")),
                }

            # RETR (teacher)
            retr_model = None
            retr_out = None
            if "RETR" in METHODS:
                retr_model, retr_out = run_retr(train_retain_loader, val_full_loader, vocab_size, num_answers)
                results[dataset_name]["RETR"][seed] = {
                    "forget_acc": float(retr_out.get("forget_acc")),
                    "retain_acc": float(retr_out.get("retain_acc")),
                }

            # MU_ALIGN (uses RETR as teacher)
            if "MU_ALIGN" in METHODS:
                if retr_model is None:
                    raise RuntimeError("MU_ALIGN requires RETR model as teacher. Ensure RETR is in METHODS.")
                mu_model, mu_out = run_mu_align(
                    train_full_loader,
                    train_forget_loader,
                    val_full_loader,
                    vocab_size,
                    num_answers,
                    retrained_teacher=retr_model,
                    init_state_dict=(orig_model.state_dict() if orig_model is not None else None),
                )
                results[dataset_name]["MU_ALIGN"][seed] = {
                    "forget_acc": float(mu_out.get("forget_acc")),
                    "retain_acc": float(mu_out.get("retain_acc")),
                }

            # Baselines
            if "MULTIDELETE" in METHODS:
                m, out = run_multidelete_baseline(train_full_loader, train_forget_loader, val_full_loader, vocab_size, num_answers)
                results[dataset_name]["MULTIDELETE"][seed] = {
                    "forget_acc": float(out.get("forget_acc")),
                    "retain_acc": float(out.get("retain_acc")),
                }

            if "AMNESIAC" in METHODS:
                m, out = run_amnesiac_baseline(train_full_loader, train_forget_loader, val_full_loader, vocab_size, num_answers)
                results[dataset_name]["AMNESIAC"][seed] = {
                    "forget_acc": float(out.get("forget_acc")),
                    "retain_acc": float(out.get("retain_acc")),
                }

            if "FISHER" in METHODS:
                m, out = run_fisher_baseline(train_full_loader, train_forget_loader, val_full_loader, vocab_size, num_answers)
                results[dataset_name]["FISHER"][seed] = {
                    "forget_acc": float(out.get("forget_acc")),
                    "retain_acc": float(out.get("retain_acc")),
                }

            if "SCRUB" in METHODS:
                m, out = run_scrub_baseline(train_full_loader, train_forget_loader, val_full_loader, vocab_size, num_answers)
                results[dataset_name]["SCRUB"][seed] = {
                    "forget_acc": float(out.get("forget_acc")),
                    "retain_acc": float(out.get("retain_acc")),
                }

    # Aggregate into schema expected by usenix_pareto_distance.py (json_nested)
    summary = {}
    for ds in results:
        summary[ds] = {}
        for method in results[ds]:
            fa_list, ra_list = [], []
            for seed in results[ds][method]:
                fa_list.append(results[ds][method][seed].get("forget_acc"))
                ra_list.append(results[ds][method][seed].get("retain_acc"))
            summary[ds][method] = {
                "forget_acc": _mean_std(fa_list),
                "retain_acc": _mean_std(ra_list),
                "seeds": list(results[ds][method].keys()),
            }

    out_path = os.path.join(SAVE_ROOT, "summary_all_methods.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{_ts()}] Saved utility summary: {out_path}")

    return summary
