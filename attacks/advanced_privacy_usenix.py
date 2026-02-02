# attacks/advanced_privacy.py
import os
import json
import csv
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\mu_align\results\privacy_cache"
DATASET_FILTER = "vqa_v2_20k"   # set to None to run all

CV_FOLDS = 5
CV_RANDOM_STATE = 42
LOGREG_MAX_ITER = 20000

# USENIX-style reporting points
FPR_POINTS = [1e-3]  # you can extend to [1e-3, 1e-2, 5e-2] if you want


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def attacker_best_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    auc = roc_auc_score(y_true, scores)
    return float(max(auc, 1.0 - auc))


def best_direction_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
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
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "n": int(len(arr))}


def find_runs(results_dir):
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
            seed = int(rest.replace("_forget_outputs.npz", ""))

            if DATASET_FILTER and dataset != DATASET_FILTER:
                continue

            npz_path = os.path.join(results_dir, fn)
            mask_path = os.path.join(results_dir, f"{dataset}_seed{seed}_member_mask.npy")
            if not os.path.exists(mask_path):
                continue

            runs.append((dataset, method, seed, npz_path, mask_path))
            break

    runs.sort(key=lambda x: (x[0], x[1], x[2]))
    return runs


def load_forget_outputs(npz_path, mask_path):
    data = np.load(npz_path)
    member_mask = np.load(mask_path).astype(int)

    logits = data["logits"]
    labels = data["labels"].astype(int)
    embeds = data["embeds"]

    n = logits.shape[0]
    if not (labels.shape[0] == embeds.shape[0] == member_mask.shape[0] == n):
        raise ValueError(f"Shape mismatch in {os.path.basename(npz_path)}")

    return {"logits": logits, "labels": labels, "embeds": embeds, "member": member_mask}


def _cv_logreg_scores_auc(X, y):
    """
    Cross-validated logistic regression attacker.
    Returns:
      - attacker-best AUC (>= 0.5)
      - out-of-fold scores (for TPR@lowFPR)
    """
    y = np.asarray(y).astype(int).reshape(-1)
    X = np.asarray(X)

    if len(np.unique(y)) < 2:
        return 0.5, np.zeros_like(y, dtype=float)

    skf = StratifiedKFold(n_splits=int(CV_FOLDS), shuffle=True, random_state=int(CV_RANDOM_STATE))
    oof = np.zeros(len(y), dtype=float)
    aucs = []

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = LogisticRegression(max_iter=int(LOGREG_MAX_ITER), solver="liblinear")
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))

    m = float(np.mean(aucs))
    return float(max(m, 1.0 - m)), oof


def _lira_lite_auc(loss_score, member):
    """
    LiRA-lite (single-model):
      Fit 1D Gaussians to loss_score for in/out and use log-likelihood ratio as score.
    WARNING:
      True LiRA requires multiple shadow models trained on disjoint subsets.
      This is a lightweight surrogate useful for reporting, but do not oversell it as full LiRA.
    """
    y = member.astype(int)
    s = np.asarray(loss_score).reshape(-1)

    if len(np.unique(y)) < 2:
        return 0.5, np.zeros_like(s)

    in_s = s[y == 1]
    out_s = s[y == 0]

    mu_in, std_in = float(np.mean(in_s)), float(np.std(in_s) + 1e-6)
    mu_out, std_out = float(np.mean(out_s)), float(np.std(out_s) + 1e-6)

    def logpdf(x, mu, std):
        return -0.5 * np.log(2 * np.pi * std * std) - 0.5 * ((x - mu) ** 2) / (std * std)

    llr = logpdf(s, mu_in, std_in) - logpdf(s, mu_out, std_out)
    auc = attacker_best_auc(y, llr)
    return float(auc), llr


def run_attacks(out):
    logits = out["logits"]
    labels = out["labels"]
    embeds = out["embeds"]
    member = out["member"].astype(int)

    probs = softmax(logits)
    y = member

    # ---------- Basic scalar attacks ----------
    # Loss score (higher => more member) use -NLL
    nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12)
    loss_score = -nll
    loss_auc = attacker_best_auc(y, loss_score)
    loss_tpr = tpr_at_fpr(y, loss_score)

    # Embedding norm
    emb_norm = np.linalg.norm(embeds, axis=1)

    # Centroid distance (members centroid)
    if np.sum(y == 1) > 0 and np.sum(y == 0) > 0:
        centroid = embeds[y == 1].mean(axis=0)
        dist = np.linalg.norm(embeds - centroid, axis=1)
        centroid_score = -dist  # closer => more member
        centroid_auc = attacker_best_auc(y, centroid_score)
    else:
        centroid_score = np.zeros(len(y), dtype=float)
        centroid_auc = 0.5

    # ---------- Probe on embeddings (strong baseline) ----------
    probe_auc, probe_oof = _cv_logreg_scores_auc(embeds, y)
    probe_tpr = tpr_at_fpr(y, probe_oof)

    # ---------- Shadow-style attack (engineered features + CV) ----------
    conf = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]

    shadow_feats = np.stack([loss_score, conf, entropy, margin, emb_norm, centroid_score], axis=1)
    shadow_auc, _shadow_oof = _cv_logreg_scores_auc(shadow_feats, y)

    # ---------- LiRA-lite ----------
    lira_auc, _lira_scores = _lira_lite_auc(loss_score, y)

    return {
        "loss_auc": float(loss_auc),
        "probe_auc": float(probe_auc),
        "centroid_distance_auc": float(centroid_auc),
        "shadow_auc": float(shadow_auc),
        "lira_auc": float(lira_auc),
        f"loss_tpr@fpr={FPR_POINTS[0]}": float(loss_tpr[f"tpr@fpr={FPR_POINTS[0]}"]),
        f"probe_tpr@fpr={FPR_POINTS[0]}": float(probe_tpr[f"tpr@fpr={FPR_POINTS[0]}"]),
    }


def main():
    runs = find_runs(RESULTS_DIR)
    if not runs:
        raise RuntimeError("No valid forget-output runs found.")

    per_run = []
    grouped = {}

    for dataset, method, seed, npz_path, mask_path in runs:
        row = {"dataset": dataset, "method": method, "seed": seed, "npz_file": os.path.basename(npz_path)}
        try:
            out = load_forget_outputs(npz_path, mask_path)
            res = run_attacks(out)
            row.update(res)
            row["error"] = None

            key = (dataset, method)
            grouped.setdefault(key, {})
            for k in [
                "loss_auc", "probe_auc", "centroid_distance_auc",
                "shadow_auc", "lira_auc",
                f"loss_tpr@fpr={FPR_POINTS[0]}",
                f"probe_tpr@fpr={FPR_POINTS[0]}",
            ]:
                grouped[key].setdefault(k, []).append(row.get(k))

        except Exception as e:
            row["error"] = str(e)

        per_run.append(row)

    aggregate = {}
    for (dataset, method), metrics in grouped.items():
        aggregate.setdefault(dataset, {})
        aggregate[dataset][method] = {k: mean_std(v) for k, v in metrics.items()}

    out_json = os.path.join(RESULTS_DIR, f"advanced_privacy_merged_{DATASET_FILTER or 'ALL'}.json")
    out_csv = out_json.replace(".json", ".csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_run": per_run, "aggregate": aggregate}, f, indent=2)

    fieldnames = [
        "dataset", "method", "seed",
        "loss_auc", "probe_auc", "centroid_distance_auc",
        "shadow_auc", "lira_auc",
        f"loss_tpr@fpr={FPR_POINTS[0]}",
        f"probe_tpr@fpr={FPR_POINTS[0]}",
        "npz_file", "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_run:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    print("\n=== ADVANCED PRIVACY (MERGED) ===")
    print("AUC is attacker-best (closer to 0.5 is better). Also reporting TPR@low FPR (lower is better).")

    for dataset in sorted(aggregate):
        print(f"\nDATASET: {dataset}")
        for method in sorted(aggregate[dataset]):
            m = aggregate[dataset][method]
            print(
                f"  {method:22s} "
                f"lossAUC {m['loss_auc']['mean']:.4f}±{m['loss_auc']['std']:.4f} | "
                f"probeAUC {m['probe_auc']['mean']:.4f}±{m['probe_auc']['std']:.4f} | "
                f"centroidAUC {m['centroid_distance_auc']['mean']:.4f}±{m['centroid_distance_auc']['std']:.4f} | "
                f"shadowAUC {m['shadow_auc']['mean']:.4f}±{m['shadow_auc']['std']:.4f} | "
                f"liRA-AUC {m['lira_auc']['mean']:.4f}±{m['lira_auc']['std']:.4f} | "
                f"TPR@0.1%FPR loss {m[f'loss_tpr@fpr={FPR_POINTS[0]}']['mean']:.4f} "
                f"probe {m[f'probe_tpr@fpr={FPR_POINTS[0]}']['mean']:.4f} "
                f"(n={m['loss_auc']['n']})"
            )

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
