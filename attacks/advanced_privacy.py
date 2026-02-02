# attacks/advanced_privacy.py
import os
import json
import csv
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------
# USER SETTINGS
# ---------------------------
RESULTS_DIR = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\mu_align\results\privacy_cache"
DATASET_FILTER = "vqa_v2_20k"   # set to None to run all datasets found
CV_FOLDS = 5
CV_RANDOM_STATE = 42
LOGREG_MAX_ITER = 20000

# Low-FPR points (USENIX-style reporting)
FPR_POINTS = [1e-3, 1e-2, 5e-2]

# Shadow / LiRA config
SHADOW_HOLDOUT_SEED = None  # None => automatically pick last seed as target per (dataset, method)
LIRA_EPS = 1e-6


# ---------------------------
# Utils
# ---------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def attacker_best_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    auc = roc_auc_score(y_true, scores)
    return float(max(auc, 1.0 - auc))


def best_direction_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    # Ensure AUC is >= 0.5 by flipping direction if needed
    auc = roc_auc_score(y_true, scores)
    return scores if auc >= 0.5 else -scores


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_targets=FPR_POINTS) -> dict:
    scores = best_direction_scores(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    out = {}
    for ft in fpr_targets:
        idx = np.searchsorted(fpr, ft, side="left")
        if idx <= 0:
            out[f"tpr@fpr={ft}"] = float(tpr[0])
        elif idx >= len(fpr):
            out[f"tpr@fpr={ft}"] = float(tpr[-1])
        else:
            f0, f1 = fpr[idx - 1], fpr[idx]
            t0, t1 = tpr[idx - 1], tpr[idx]
            if (f1 - f0) < 1e-12:
                out[f"tpr@fpr={ft}"] = float(t1)
            else:
                w = (ft - f0) / (f1 - f0)
                out[f"tpr@fpr={ft}"] = float(t0 + w * (t1 - t0))
    return out


def mean_std(vals):
    # ignore None and NaN
    cleaned = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        cleaned.append(v)

    if not cleaned:
        return {"mean": None, "std": None, "n": 0}

    arr = np.array(cleaned, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


# ---------------------------
# Discover runs from privacy_cache
# Naming must match:
#   {dataset}_{method}_seed{seed}_forget_outputs.npz
#   {dataset}_seed{seed}_member_mask.npy
# ---------------------------
def find_runs(results_dir: str, dataset_filter: str | None):
    KNOWN_METHODS = ["MU_ALIGN", "ORIG", "RETR", "AMNESIAC", "FISHER", "SCRUB", "MULTIDELETE", "SISA"]
    runs = []
    for fn in os.listdir(results_dir):
        if not fn.endswith("_forget_outputs.npz"):
            continue

        for method in KNOWN_METHODS:
            token = f"_{method}_seed"
            if token not in fn:
                continue

            prefix, rest = fn.split(token)
            dataset = prefix
            seed_str = rest.replace("_forget_outputs.npz", "")

            if not seed_str.isdigit():
                continue
            seed = int(seed_str)

            if dataset_filter and dataset != dataset_filter:
                continue

            npz_path = os.path.join(results_dir, fn)
            mask_path = os.path.join(results_dir, f"{dataset}_seed{seed}_member_mask.npy")
            if not os.path.exists(mask_path):
                continue

            runs.append((dataset, method, seed, npz_path, mask_path))
            break

    runs.sort(key=lambda x: (x[0], x[1], x[2]))
    return runs


def load_forget_outputs(npz_path: str, mask_path: str):
    data = np.load(npz_path)
    member = np.load(mask_path).astype(int)

    logits = data["logits"]
    labels = data["labels"].astype(int)
    embeds = data["embeds"]

    n = logits.shape[0]
    if labels.shape[0] != n or embeds.shape[0] != n or member.shape[0] != n:
        raise ValueError(f"Shape mismatch: {os.path.basename(npz_path)}")

    return {"logits": logits, "labels": labels, "embeds": embeds, "member": member}


# ---------------------------
# Feature / score construction
# ---------------------------
def scores_from_outputs(out: dict):
    logits = out["logits"]
    labels = out["labels"]
    embeds = out["embeds"]
    member = out["member"].astype(int)

    probs = softmax(logits)

    # standard scores
    loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-12)     # higher = more confident member typically
    loss_score = -loss                                                # we will use -loss as "member score"
    conf_score = probs.max(axis=1)                                    # higher = more member-like
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    entropy_score = -entropy                                          # lower entropy = more member-like
    norm_score = np.linalg.norm(embeds, axis=1)                       # can correlate with membership

    # centroid distance score
    centroid_score = None
    if np.sum(member == 1) > 0 and np.sum(member == 0) > 0:
        centroid = embeds[member == 1].mean(axis=0)
        dist = np.linalg.norm(embeds - centroid, axis=1)
        centroid_score = -dist                                        # closer to train-forget centroid = more member-like

    return {
        "member": member,
        "loss_score": loss_score,
        "conf_score": conf_score,
        "entropy_score": entropy_score,
        "norm_score": norm_score,
        "centroid_score": centroid_score,
    }


def make_attack_features(scores: dict):
    # Shadow attack uses a compact feature vector per example
    feats = [
        scores["loss_score"],
        scores["conf_score"],
        scores["entropy_score"],
        scores["norm_score"],
    ]
    if scores["centroid_score"] is not None:
        feats.append(scores["centroid_score"])
    X = np.stack(feats, axis=1).astype(np.float32)
    y = scores["member"].astype(int)
    return X, y


# ---------------------------
# Basic attacks (your earlier ones + centroid + TPR@FPR)
# ---------------------------
def run_basic_attacks(out: dict):
    s = scores_from_outputs(out)
    y = s["member"]

    res = {}

    # Loss AUC + TPR@FPR
    res["loss_auc"] = attacker_best_auc(y, s["loss_score"])
    for k, v in tpr_at_fpr(y, s["loss_score"]).items():
        res[f"loss_{k}"] = v

    # Linear probe on embeds (CV AUC) + TPR@FPR (fit full then compute scores for TPR)
    embeds = out["embeds"]
    if len(np.unique(y)) < 2:
        res["probe_auc"] = None
        for ft in FPR_POINTS:
            res[f"probe_tpr@fpr={ft}"] = None
    else:
        aucs = []
        skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
        for tr, te in skf.split(embeds, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(embeds[tr])
            Xte = scaler.transform(embeds[te])
            clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, solver="liblinear")
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        mean_auc = float(np.mean(aucs))
        res["probe_auc"] = float(max(mean_auc, 1.0 - mean_auc))

        # Fit on full for TPR@FPR reporting
        scaler = StandardScaler()
        Xall = scaler.fit_transform(embeds)
        clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, solver="liblinear")
        clf.fit(Xall, y)
        probe_scores = clf.predict_proba(Xall)[:, 1]
        tprs = tpr_at_fpr(y, probe_scores)
        for ft in FPR_POINTS:
            res[f"probe_tpr@fpr={ft}"] = tprs.get(f"tpr@fpr={ft}", None)

    # centroid AUC + TPR@FPR
    if s["centroid_score"] is None:
        res["centroid_auc"] = None
        for ft in FPR_POINTS:
            res[f"centroid_tpr@fpr={ft}"] = None
    else:
        res["centroid_auc"] = attacker_best_auc(y, s["centroid_score"])
        tprs = tpr_at_fpr(y, s["centroid_score"])
        for ft in FPR_POINTS:
            res[f"centroid_tpr@fpr={ft}"] = tprs.get(f"tpr@fpr={ft}", None)

    return res


# ---------------------------
# Shadow-model MIA (trained on shadow runs, tested on a held-out run)
# This is stronger than single-score, and is “drop-in” with your cached outputs.
# ---------------------------
def shadow_attack_for_group(group_runs):
    """
    group_runs: list of dicts each containing keys:
      dataset, method, seed, out (loaded outputs)
    Strategy:
      - choose one run as target, rest as shadow
      - build (X_shadow, y_shadow) from shadows
      - train attack model on shadows
      - evaluate on target -> AUC + TPR@0.1%FPR
    """
    if len(group_runs) < 2:
        return {"shadow_auc": None, "shadow_tpr@fpr=0.001": None, "shadow_target_seed": None}

    seeds = sorted([r["seed"] for r in group_runs])
    if SHADOW_HOLDOUT_SEED is not None and SHADOW_HOLDOUT_SEED in seeds:
        target_seed = SHADOW_HOLDOUT_SEED
    else:
        target_seed = seeds[-1]  # deterministic default

    target = None
    shadows = []
    for r in group_runs:
        if r["seed"] == target_seed:
            target = r
        else:
            shadows.append(r)

    if target is None or len(shadows) == 0:
        return {"shadow_auc": None, "shadow_tpr@fpr=0.001": None, "shadow_target_seed": None}

    # Build shadow training set
    Xs, ys = [], []
    for sh in shadows:
        sc = scores_from_outputs(sh["out"])
        X, y = make_attack_features(sc)
        Xs.append(X)
        ys.append(y)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)

    if len(np.unique(ys)) < 2:
        return {"shadow_auc": None, "shadow_tpr@fpr=0.001": None, "shadow_target_seed": target_seed}

    # Train attack model
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, solver="liblinear")
    clf.fit(Xs, ys)

    # Evaluate on target
    sc_t = scores_from_outputs(target["out"])
    Xt, yt = make_attack_features(sc_t)
    Xt = scaler.transform(Xt)
    p = clf.predict_proba(Xt)[:, 1]

    shadow_auc = attacker_best_auc(yt, p)
    tprs = tpr_at_fpr(yt, p)
    return {
        "shadow_auc": shadow_auc,
        "shadow_tpr@fpr=0.001": tprs.get("tpr@fpr=0.001", None),
        "shadow_target_seed": target_seed,
    }


# ---------------------------
# LiRA-style (GLOBAL) likelihood ratio using shadows
# Full LiRA needs stable example IDs across runs; you do not have them.
# This is still a strong likelihood-ratio attack and is easy to add safely.
# ---------------------------
def lira_global_for_group(group_runs):
    """
    Use shadow runs to estimate two Gaussians on a chosen scalar score:
      score | member=1  ~ N(mu_in,  sigma_in^2)
      score | member=0  ~ N(mu_out, sigma_out^2)
    Then for target scores, compute log-likelihood ratio:
      llr = log p(score|in) - log p(score|out)
    Evaluate attacker-best AUC and TPR@0.1%FPR on target run.

    We use loss_score (-loss) by default because it’s standard in MIA.
    """
    if len(group_runs) < 2:
        return {"lira_auc": None, "lira_tpr@fpr=0.001": None, "lira_target_seed": None}

    seeds = sorted([r["seed"] for r in group_runs])
    target_seed = SHADOW_HOLDOUT_SEED if (SHADOW_HOLDOUT_SEED in seeds) else seeds[-1]

    target = None
    shadows = []
    for r in group_runs:
        if r["seed"] == target_seed:
            target = r
        else:
            shadows.append(r)

    if target is None or len(shadows) == 0:
        return {"lira_auc": None, "lira_tpr@fpr=0.001": None, "lira_target_seed": None}

    # Collect shadow scores for global in/out distributions
    in_scores = []
    out_scores = []
    for sh in shadows:
        sc = scores_from_outputs(sh["out"])
        y = sc["member"]
        z = sc["loss_score"]  # LiRA uses loss-like statistics; we use -loss
        in_scores.append(z[y == 1])
        out_scores.append(z[y == 0])

    in_scores = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
    out_scores = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])

    if in_scores.size < 10 or out_scores.size < 10:
        return {"lira_auc": None, "lira_tpr@fpr=0.001": None, "lira_target_seed": target_seed}

    mu_in = float(np.mean(in_scores))
    mu_out = float(np.mean(out_scores))
    sd_in = float(np.std(in_scores) + LIRA_EPS)
    sd_out = float(np.std(out_scores) + LIRA_EPS)

    # Compute llr on target
    sc_t = scores_from_outputs(target["out"])
    y_t = sc_t["member"].astype(int)
    z_t = sc_t["loss_score"]

    # Gaussian logpdf up to constant: -0.5*((x-mu)/sd)^2 - log(sd)
    ll_in = -0.5 * ((z_t - mu_in) / sd_in) ** 2 - np.log(sd_in)
    ll_out = -0.5 * ((z_t - mu_out) / sd_out) ** 2 - np.log(sd_out)
    llr = ll_in - ll_out

    lira_auc = attacker_best_auc(y_t, llr)
    tprs = tpr_at_fpr(y_t, llr)
    return {
        "lira_auc": lira_auc,
        "lira_tpr@fpr=0.001": tprs.get("tpr@fpr=0.001", None),
        "lira_target_seed": target_seed,
    }


# ---------------------------
# Main
# ---------------------------
def main():
    runs = find_runs(RESULTS_DIR, DATASET_FILTER)
    if not runs:
        raise RuntimeError("No valid forget-output runs found in RESULTS_DIR.")

    # Load all runs once
    loaded = []
    for dataset, method, seed, npz_path, mask_path in runs:
        try:
            out = load_forget_outputs(npz_path, mask_path)
            loaded.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "npz_file": os.path.basename(npz_path),
                "out": out,
                "error": None,
            })
        except Exception as e:
            loaded.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "npz_file": os.path.basename(npz_path),
                "out": None,
                "error": str(e),
            })

    # Per-run computations + aggregation buckets
    per_run = []
    grouped = {}  # (dataset, method) -> dict(metric -> list)

    for r in loaded:
        row = {
            "dataset": r["dataset"],
            "method": r["method"],
            "seed": r["seed"],
            "npz_file": r["npz_file"],
            "error": r["error"],
        }
        if r["out"] is None:
            per_run.append(row)
            continue

        try:
            basic = run_basic_attacks(r["out"])
            row.update(basic)

            key = (r["dataset"], r["method"])
            grouped.setdefault(key, {})
            for mk, mv in basic.items():
                grouped[key].setdefault(mk, []).append(mv)

            row["error"] = None
        except Exception as e:
            row["error"] = str(e)

        per_run.append(row)

    # Add shadow + LiRA per (dataset, method) group
    shadow_summary = {}
    lira_summary = {}
    for (dataset, method), _metrics in grouped.items():
        group_runs = [rr for rr in loaded if rr["dataset"] == dataset and rr["method"] == method and rr["out"] is not None]
        sh = shadow_attack_for_group(group_runs)
        li = lira_global_for_group(group_runs)
        shadow_summary.setdefault(dataset, {})[method] = sh
        lira_summary.setdefault(dataset, {})[method] = li

        # also push into grouped so it appears in aggregate JSON
        key = (dataset, method)
        grouped[key].setdefault("shadow_auc", []).append(sh.get("shadow_auc"))
        grouped[key].setdefault("shadow_tpr@fpr=0.001", []).append(sh.get("shadow_tpr@fpr=0.001"))
        grouped[key].setdefault("lira_auc", []).append(li.get("lira_auc"))
        grouped[key].setdefault("lira_tpr@fpr=0.001", []).append(li.get("lira_tpr@fpr=0.001"))

    # Build aggregate
    aggregate = {}
    for (dataset, method), metrics in grouped.items():
        aggregate.setdefault(dataset, {})
        aggregate[dataset][method] = {k: mean_std(v) for k, v in metrics.items()}

    # Write outputs
    tag = DATASET_FILTER or "ALL"
    out_json = os.path.join(RESULTS_DIR, f"advanced_privacy_merged_{tag}.json")
    out_csv = os.path.join(RESULTS_DIR, f"advanced_privacy_merged_{tag}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_run": per_run,
                "aggregate": aggregate,
                "shadow_summary": shadow_summary,
                "lira_summary": lira_summary,
                "notes": {
                    "auc": "attacker-best AUC (closer to 0.5 is better)",
                    "tpr_at_fpr": "lower is better",
                    "shadow_attack": "trained on shadow runs (other seeds), tested on held-out seed",
                    "lira_global": "global Gaussian likelihood ratio on -loss using shadow runs; full LiRA needs stable example IDs",
                },
            },
            f,
            indent=2,
        )

    # CSV fields (keep stable)
    fieldnames = [
        "dataset", "method", "seed",
        "loss_auc", "probe_auc", "centroid_auc",
        "loss_tpr@fpr=0.001", "probe_tpr@fpr=0.001", "centroid_tpr@fpr=0.001",
        "shadow_auc", "shadow_tpr@fpr=0.001",
        "lira_auc", "lira_tpr@fpr=0.001",
        "npz_file", "error",
    ]

    # Enrich per-run rows with group-level shadow/lira so CSV is readable
    per_run_enriched = []
    for r in per_run:
        rr = dict(r)
        ds = rr.get("dataset")
        m = rr.get("method")
        if ds in aggregate and m in aggregate[ds]:
            rr["shadow_auc"] = aggregate[ds][m].get("shadow_auc", {}).get("mean", None)
            rr["shadow_tpr@fpr=0.001"] = aggregate[ds][m].get("shadow_tpr@fpr=0.001", {}).get("mean", None)
            rr["lira_auc"] = aggregate[ds][m].get("lira_auc", {}).get("mean", None)
            rr["lira_tpr@fpr=0.001"] = aggregate[ds][m].get("lira_tpr@fpr=0.001", {}).get("mean", None)
        per_run_enriched.append(rr)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_run_enriched:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    # Print summary (USENIX-style)
    print("\n=== ADVANCED PRIVACY (MERGED) ===")
    print("AUC is attacker-best (closer to 0.5 is better). Also reporting TPR@low FPR (lower is better).")

    for dataset in sorted(aggregate.keys()):
        print(f"\nDATASET: {dataset}")
        for method in sorted(aggregate[dataset].keys()):
            m = aggregate[dataset][method]

            def fmt_ms(obj):
                if obj is None or obj.get("mean") is None:
                    return "NA"
                return f"{obj['mean']:.4f}±{(obj['std'] if obj['std'] is not None else 0.0):.4f}"

            lossA = m.get("loss_auc")
            probeA = m.get("probe_auc")
            centA = m.get("centroid_auc")

            # low-fpr (0.1%)
            lt = m.get("loss_tpr@fpr=0.001", {}).get("mean", None)
            pt = m.get("probe_tpr@fpr=0.001", {}).get("mean", None)

            shA = m.get("shadow_auc")
            liA = m.get("lira_auc")

            # n (use probe_auc n if present else loss_auc n)
            n = max(m.get("probe_auc", {}).get("n", 0), m.get("loss_auc", {}).get("n", 0))

            print(
                f"  {method:22s} "
                f"lossAUC {fmt_ms(lossA)} | "
                f"probeAUC {fmt_ms(probeA)} | "
                f"centroidAUC {fmt_ms(centA)} | "
                f"shadowAUC {fmt_ms(shA)} | "
                f"liRA-AUC {fmt_ms(liA)} | "
                f"TPR@0.1%FPR loss {('NA' if lt is None else f'{lt:.4f}')} "
                f"probe {('NA' if pt is None else f'{pt:.4f}')} "
                f"(n={n})"
            )

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
