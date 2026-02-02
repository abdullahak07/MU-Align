# attacks/usenix_pareto_distance.py
import os
import json
import csv
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\mu_align\results\privacy_cache"
DATASET_FILTER = "vqa_v2_20k"  # set None to run all

UTILITY_PATH = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\mu_align\results\summary_all_methods.json"

METHODS_ORDER = ["ORIG", "RETR", "MU_ALIGN", "MULTIDELETE", "AMNESIAC", "FISHER", "SCRUB"]

# Low-FPR regime (USENIX-style)
FPR_POINTS = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2]

# Probe CV
CV_FOLDS = 5
CV_RANDOM_STATE = 42
LOGREG_MAX_ITER = 20000

# ======= Step-2 fix: HELD-OUT calibration split =======
# Calibrate thresholds on a subset of the evaluation examples,
# then report calibrated TPR on the held-out subset.
CALIBRATION_FRAC = 0.5
CALIBRATION_RANDOM_STATE = 12345  # controls deterministic split per run

# Pareto outputs
SAVE_PARETO = True
PARETO_DIR = RESULTS_DIR

# Table outputs (Step-4)
SAVE_USENIX_TABLES = True


# ============================================================
# BASIC HELPERS
# ============================================================
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


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_targets: List[float]) -> Dict[float, float]:
    """
    ROC-interp TPR@FPR (direction chosen to maximize AUC).
    """
    scores = best_direction_scores(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    out: Dict[float, float] = {}
    for ft in fpr_targets:
        ft = float(ft)
        idx = np.searchsorted(fpr, ft, side="left")
        if idx <= 0:
            out[ft] = float(tpr[0])
        elif idx >= len(fpr):
            out[ft] = float(tpr[-1])
        else:
            f0, f1 = fpr[idx - 1], fpr[idx]
            t0, t1 = tpr[idx - 1], tpr[idx]
            if (f1 - f0) < 1e-12:
                out[ft] = float(t1)
            else:
                w = (ft - f0) / (f1 - f0)
                out[ft] = float(t0 + w * (t1 - t0))
    return out


def _split_calib_test_indices(y: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic stratified split of indices into calib + test.
    """
    y = y.astype(int)
    n = len(y)
    idx = np.arange(n)

    rng = np.random.RandomState(seed)
    # stratify by y
    idx0 = idx[y == 0]
    idx1 = idx[y == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_cal = int(np.floor(len(idx0) * frac))
    n1_cal = int(np.floor(len(idx1) * frac))

    calib = np.concatenate([idx0[:n0_cal], idx1[:n1_cal]])
    test = np.concatenate([idx0[n0_cal:], idx1[n1_cal:]])
    rng.shuffle(calib)
    rng.shuffle(test)

    # safety
    if calib.size == 0 or test.size == 0:
        # fallback: no split
        return idx, idx
    return calib, test


def calibrated_tpr_at_fpr_heldout(
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr_targets: List[float],
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[float, float]:
    """
    Step-2 fix:
      - choose best direction using calibration data
      - set threshold via calibration NEG quantile
      - compute TPR on held-out test set
    """
    y = y_true.astype(int)

    y_cal = y[calib_idx]
    s_cal_raw = scores[calib_idx]
    # choose direction based on calibration slice
    s_cal = best_direction_scores(y_cal, s_cal_raw)

    # apply same direction to test slice
    # We must mirror the direction choice; easiest is:
    # if best_direction_scores flipped sign on calib, flip for test too.
    auc_cal = roc_auc_score(y_cal, s_cal_raw)
    flip = (auc_cal < 0.5)
    s_test = (-scores[test_idx]) if flip else (scores[test_idx])
    y_test = y[test_idx]

    neg_cal = s_cal[y_cal == 0]
    if neg_cal.size == 0:
        return {float(ft): None for ft in fpr_targets}

    out: Dict[float, float] = {}
    for ft in fpr_targets:
        ft = float(ft)
        q = 1.0 - ft
        q = min(max(q, 0.0), 1.0)
        thr = float(np.quantile(neg_cal, q))
        pred = (s_test >= thr).astype(int)

        tp = np.sum((pred == 1) & (y_test == 1))
        fn = np.sum((pred == 0) & (y_test == 1))
        out[ft] = float(tp / max(tp + fn, 1))
    return out


def mean_std(vals: List[float]) -> Dict[str, Any]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


# ============================================================
# RUN DISCOVERY
# ============================================================
def _parse_dataset_method(left: str, methods: List[str]) -> Optional[Tuple[str, str]]:
    """
    Robust parsing for dataset names with underscores.
    left = "{dataset}_{method}" where dataset may contain underscores.
    """
    for m in sorted(methods, key=len, reverse=True):
        suf = "_" + m
        if left.endswith(suf):
            dataset = left[: -len(suf)]
            method = m
            if dataset:
                return dataset, method
    return None


def find_runs(results_dir: str, dataset_filter: Optional[str]) -> List[Tuple[str, str, int, str, str]]:
    """
    Finds:
      {dataset}_{METHOD}_seed{seed}_forget_outputs.npz
      {dataset}_seed{seed}_member_mask.npy
    """
    runs: List[Tuple[str, str, int, str, str]] = []
    suffix = "_forget_outputs.npz"

    if not os.path.isdir(results_dir):
        return runs

    for fn in os.listdir(results_dir):
        if not fn.endswith(suffix):
            continue

        base = fn[: -len(suffix)]
        if "_seed" not in base:
            continue

        left, seed_part = base.rsplit("_seed", 1)
        if not seed_part.isdigit():
            continue
        seed = int(seed_part)

        parsed = _parse_dataset_method(left, METHODS_ORDER)
        if parsed is None:
            continue
        dataset, method = parsed

        if dataset_filter is not None and dataset != dataset_filter:
            continue

        npz_path = os.path.join(results_dir, fn)
        mask_path = os.path.join(results_dir, f"{dataset}_seed{seed}_member_mask.npy")
        if not os.path.exists(mask_path):
            continue

        runs.append((dataset, method, seed, npz_path, mask_path))

    runs.sort(key=lambda x: (x[0], x[1], x[2]))
    return runs


def load_npz_and_mask(npz_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    member = np.load(mask_path).astype(int)

    logits = data["logits"]
    labels = data["labels"].astype(int)
    embeds = data["embeds"]

    n = logits.shape[0]
    if labels.shape[0] != n or embeds.shape[0] != n or member.shape[0] != n:
        raise ValueError(f"Shape mismatch: {os.path.basename(npz_path)}")
    return logits, labels, embeds, member


# ============================================================
# ATTACK SCORES
# ============================================================
def compute_loss_score(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-12)
    return -loss  # higher => more likely member


def linear_probe_cv_auc(embeds: np.ndarray, member: np.ndarray) -> Optional[float]:
    y = member.astype(int)
    if len(np.unique(y)) < 2:
        return None

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

    m = float(np.mean(aucs))
    return float(max(m, 1.0 - m))


def fit_probe_full_scores(embeds: np.ndarray, member: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X = scaler.fit_transform(embeds)
    y = member.astype(int)
    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, solver="liblinear")
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]


def centroid_distance_auc(embeds: np.ndarray, member: np.ndarray) -> Optional[float]:
    if np.sum(member == 1) == 0 or np.sum(member == 0) == 0:
        return None
    centroid = embeds[member == 1].mean(axis=0)
    dist = np.linalg.norm(embeds - centroid, axis=1)
    return attacker_best_auc(member, -dist)


def shadow_feature_scores(logits: np.ndarray, labels: np.ndarray, embeds: np.ndarray, member: np.ndarray) -> np.ndarray:
    """
    Feature-based attacker:
      feats = [loss, max_conf, entropy, margin, emb_norm]
    """
    y = member.astype(int)

    probs = softmax(logits)
    max_conf = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = np.abs(top2[:, 1] - top2[:, 0])
    emb_norm = np.linalg.norm(embeds, axis=1)

    loss = -compute_loss_score(logits, labels)  # positive
    feats = np.stack([loss, max_conf, entropy, margin, emb_norm], axis=1)

    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, solver="liblinear")
    clf.fit(feats, y)
    return clf.predict_proba(feats)[:, 1]


def shadow_feature_auc(logits: np.ndarray, labels: np.ndarray, embeds: np.ndarray, member: np.ndarray) -> Optional[float]:
    y = member.astype(int)
    if len(np.unique(y)) < 2:
        return None
    scores = shadow_feature_scores(logits, labels, embeds, member)
    return attacker_best_auc(y, scores)


# ============================================================
# DISTANCE-TO-RETR
# ============================================================
def _cov_frob(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    return (x.T @ x) / float(x.shape[0] - 1)


def distance_to_retr(
    logits: np.ndarray,
    embeds: np.ndarray,
    retr_logits: np.ndarray,
    retr_embeds: np.ndarray,
) -> Dict[str, float]:
    if logits.shape != retr_logits.shape or embeds.shape != retr_embeds.shape:
        raise ValueError("distance_to_retr: shape mismatch with RETR arrays")

    lmse = float(np.mean((logits - retr_logits) ** 2))
    emse = float(np.mean((embeds - retr_embeds) ** 2))

    mu1 = embeds.mean(axis=0)
    mu2 = retr_embeds.mean(axis=0)
    mean_shift = float(np.linalg.norm(mu1 - mu2))

    c1 = _cov_frob(embeds).astype(np.float64)
    c2 = _cov_frob(retr_embeds).astype(np.float64)
    cov_frob = float(np.linalg.norm(c1 - c2, ord="fro"))

    return {
        "logits_mse_all": lmse,
        "embeds_mse_all": emse,
        "mean_shift_emb": mean_shift,
        "cov_frob_emb": cov_frob,
    }


# ============================================================
# UTILITY LOADER (JSON)
# ============================================================
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_utility_summary(path: str) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], str]:
    """
    util[dataset][method] = {"retain_acc": mean, "forget_acc": mean}
    """
    util: Dict[str, Dict[str, Dict[str, float]]] = {}
    if path is None or not os.path.exists(path):
        return util, "missing"

    ext = os.path.splitext(path)[1].lower()
    if ext != ".json":
        return util, "non_json_unsupported"

    try:
        obj = json.loads(_read_text(path))
    except Exception:
        return util, "json_failed"

    if not isinstance(obj, dict):
        return util, "json_bad_root"

    def _get_mean(x):
        if isinstance(x, dict) and "mean" in x:
            return x["mean"]
        return x

    for ds, by_method in obj.items():
        if not isinstance(by_method, dict):
            continue
        for m, stats in by_method.items():
            if not isinstance(stats, dict):
                continue
            ra = _get_mean(stats.get("retain_acc"))
            fa = _get_mean(stats.get("forget_acc"))
            if ra is None or fa is None:
                continue
            util.setdefault(ds, {})[m] = {"retain_acc": float(ra), "forget_acc": float(fa)}

    return util, "json_nested"


# ============================================================
# PARETO OUTPUT
# ============================================================
def save_pareto_plots(per_method_summary: Dict[str, Dict[str, Any]], out_prefix: str):
    if not SAVE_PARETO:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skipping pareto plots.")
        return

    methods = sorted(per_method_summary.keys())
    rows = []
    for m in methods:
        d = per_method_summary[m]
        rows.append({
            "method": m,
            "retain_acc": d.get("retain_acc"),
            "forget_acc": d.get("forget_acc"),
            "worstTPR@1e-05": d.get("worstTPR@1e-05"),
            "calWorstTPR_holdout@1e-05": d.get("calWorstTPR_holdout@1e-05"),
            "dist_logits_mse": d.get("dist_logits_mse"),
            "dist_embeds_mse": d.get("dist_embeds_mse"),
            "dist_mean_shift_emb": d.get("dist_mean_shift_emb"),
            "dist_cov_frob_emb": d.get("dist_cov_frob_emb"),
        })

    csv_path = out_prefix + "_pareto_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def _scatter(xkey, ykey, title, png_name, xlabel=None, ylabel=None):
        xs, ys, labs = [], [], []
        for r in rows:
            x = r.get(xkey, None)
            y = r.get(ykey, None)
            if x is None or y is None:
                continue
            if isinstance(x, float) and np.isnan(x):
                continue
            if isinstance(y, float) and np.isnan(y):
                continue
            xs.append(float(x))
            ys.append(float(y))
            labs.append(r["method"])

        if not xs:
            return

        plt.figure()
        plt.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labs):
            plt.text(x, y, lab, fontsize=8)
        plt.title(title)
        plt.xlabel(xlabel or xkey)
        plt.ylabel(ylabel or ykey)
        plt.tight_layout()
        plt.savefig(png_name, dpi=200)
        plt.close()

    _scatter(
        "retain_acc",
        "calWorstTPR_holdout@1e-05",
        "Pareto: Retain Utility vs Held-out CalWorstTPR@1e-5 (lower is better)",
        out_prefix + "_pareto_retain_vs_calWorstTPR_holdout_1e5.png",
        xlabel="retain_acc (higher better)",
        ylabel="calWorstTPR_holdout@1e-5 (lower better)",
    )

    _scatter(
        "retain_acc",
        "dist_embeds_mse",
        "Pareto: Retain Utility vs Distance-to-RETR (embeds MSE; lower is better)",
        out_prefix + "_pareto_retain_vs_distEmbMSE.png",
        xlabel="retain_acc (higher better)",
        ylabel="embeds_mse_all (lower better)",
    )

    print(f"[INFO] Saved Pareto CSV: {csv_path}")
    print(f"[INFO] Saved Pareto PNGs with prefix: {out_prefix}_*.png")


# ============================================================
# USENIX TABLE OUTPUTS (Step-4)
# ============================================================
def _write_table_csv(path: str, headers: List[str], rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in headers})


def save_usenix_tables(dataset: str, aggregate: Dict[str, Dict[str, Any]], out_dir: str):
    if not SAVE_USENIX_TABLES:
        return

    # Utility table
    util_rows = []
    for method, m in sorted(aggregate.items()):
        util_rows.append({
            "dataset": dataset,
            "method": method,
            "retain_acc_mean": m.get("retain_acc", {}).get("mean", None),
            "retain_acc_std":  m.get("retain_acc", {}).get("std", None),
            "forget_acc_mean": m.get("forget_acc", {}).get("mean", None),
            "forget_acc_std":  m.get("forget_acc", {}).get("std", None),
        })
    _write_table_csv(
        os.path.join(out_dir, f"tables_{dataset}_utility.csv"),
        ["dataset", "method", "retain_acc_mean", "retain_acc_std", "forget_acc_mean", "forget_acc_std"],
        util_rows
    )

    # Privacy (low-FPR) table: ROC + held-out calibrated
    priv_rows = []
    for method, m in sorted(aggregate.items()):
        row = {"dataset": dataset, "method": method}
        for ft in [1e-5, 1e-4, 1e-3]:
            row[f"worstTPR_mean@{ft}"] = m.get(f"worstTPR@{float(ft)}", {}).get("mean", None)
            row[f"worstTPR_std@{ft}"]  = m.get(f"worstTPR@{float(ft)}", {}).get("std", None)
            row[f"calWorstTPR_holdout_mean@{ft}"] = m.get(f"calWorstTPR_holdout@{float(ft)}", {}).get("mean", None)
            row[f"calWorstTPR_holdout_std@{ft}"]  = m.get(f"calWorstTPR_holdout@{float(ft)}", {}).get("std", None)
        priv_rows.append(row)

    _write_table_csv(
        os.path.join(out_dir, f"tables_{dataset}_privacy_lowfpr.csv"),
        list(priv_rows[0].keys()) if priv_rows else ["dataset", "method"],
        priv_rows
    )

    # Distance-to-RETR table
    dist_rows = []
    for method, m in sorted(aggregate.items()):
        dist_rows.append({
            "dataset": dataset,
            "method": method,
            "dist_embeds_mse_mean": m.get("dist_embeds_mse_all", {}).get("mean", None),
            "dist_embeds_mse_std":  m.get("dist_embeds_mse_all", {}).get("std", None),
            "mean_shift_mean":      m.get("dist_mean_shift_emb", {}).get("mean", None),
            "mean_shift_std":       m.get("dist_mean_shift_emb", {}).get("std", None),
            "cov_frob_mean":        m.get("dist_cov_frob_emb", {}).get("mean", None),
            "cov_frob_std":         m.get("dist_cov_frob_emb", {}).get("std", None),
        })
    _write_table_csv(
        os.path.join(out_dir, f"tables_{dataset}_distance.csv"),
        ["dataset", "method",
         "dist_embeds_mse_mean", "dist_embeds_mse_std",
         "mean_shift_mean", "mean_shift_std",
         "cov_frob_mean", "cov_frob_std"],
        dist_rows
    )


# ============================================================
# MAIN
# ============================================================
def main():
    runs = find_runs(RESULTS_DIR, DATASET_FILTER)
    if not runs:
        raise RuntimeError(f"No runs found in {RESULTS_DIR} for dataset={DATASET_FILTER}")

    util, util_mode = load_utility_summary(UTILITY_PATH)
    print(f"[INFO] Utility file: {UTILITY_PATH} (mode={util_mode})")

    # index[(dataset, seed, method)] -> (npz_path, mask_path)
    index: Dict[Tuple[str, int, str], Tuple[str, str]] = {}
    for ds, method, seed, npz_path, mask_path in runs:
        index[(ds, seed, method)] = (npz_path, mask_path)

    per_run: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for dataset, method, seed, npz_path, mask_path in runs:
        row: Dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "npz_file": os.path.basename(npz_path),
        }

        try:
            logits, labels, embeds, member = load_npz_and_mask(npz_path, mask_path)
            y = member.astype(int)
            if len(np.unique(y)) < 2:
                raise ValueError("member_mask has only one class; cannot compute AUC/TPR.")

            # Step-2 split (per run)
            split_seed = int(CALIBRATION_RANDOM_STATE + 1000 * seed + (hash(method) % 997))
            calib_idx, test_idx = _split_calib_test_indices(y, CALIBRATION_FRAC, split_seed)

            # --- Attacks ---
            loss_score = compute_loss_score(logits, labels)
            loss_auc = attacker_best_auc(y, loss_score)
            loss_tpr = tpr_at_fpr(y, loss_score, FPR_POINTS)
            loss_cal_holdout = calibrated_tpr_at_fpr_heldout(y, loss_score, FPR_POINTS, calib_idx, test_idx)

            probe_auc = linear_probe_cv_auc(embeds, y)
            probe_scores = fit_probe_full_scores(embeds, y)
            probe_tpr = tpr_at_fpr(y, probe_scores, FPR_POINTS)
            probe_cal_holdout = calibrated_tpr_at_fpr_heldout(y, probe_scores, FPR_POINTS, calib_idx, test_idx)

            shadow_auc = shadow_feature_auc(logits, labels, embeds, y)
            shadow_scores = shadow_feature_scores(logits, labels, embeds, y)
            shadow_tpr = tpr_at_fpr(y, shadow_scores, FPR_POINTS)
            shadow_cal_holdout = calibrated_tpr_at_fpr_heldout(y, shadow_scores, FPR_POINTS, calib_idx, test_idx)

            centroid_auc = centroid_distance_auc(embeds, y)

            # Worst-case TPR across attacks at each FPR (ROC interp)
            worst_tpr: Dict[float, float] = {}
            # Worst-case TPR across attacks (held-out calibrated)
            cal_worst_holdout: Dict[float, float] = {}

            for ft in FPR_POINTS:
                ft = float(ft)
                worst_tpr[ft] = float(max(loss_tpr[ft], probe_tpr[ft], shadow_tpr[ft]))

                vals_hold = []
                vals_hold.append(loss_cal_holdout.get(ft, None))
                vals_hold.append(probe_cal_holdout.get(ft, None))
                vals_hold.append(shadow_cal_holdout.get(ft, None))
                vals_hold = [v for v in vals_hold if v is not None and not (isinstance(v, float) and np.isnan(v))]
                cal_worst_holdout[ft] = float(max(vals_hold)) if vals_hold else None

            # --- Utility attach (method-level means) ---
            ra = None
            fa = None
            if dataset in util and method in util[dataset]:
                ra = util[dataset][method].get("retain_acc", None)
                fa = util[dataset][method].get("forget_acc", None)

            # --- Distance-to-RETR ---
            dist = {"logits_mse_all": None, "embeds_mse_all": None, "mean_shift_emb": None, "cov_frob_emb": None}
            if method != "RETR":
                key_retr = (dataset, seed, "RETR")
                if key_retr in index:
                    retr_npz, retr_mask = index[key_retr]
                    retr_logits, retr_labels, retr_embeds, _ = load_npz_and_mask(retr_npz, retr_mask)
                    if not np.array_equal(labels, retr_labels):
                        raise ValueError("Distance-to-RETR: labels mismatch (ordering differs). Rebuild cache consistently.")
                    dist = distance_to_retr(logits, embeds, retr_logits, retr_embeds)

            row.update({
                "retain_acc": ra,
                "forget_acc": fa,

                "loss_auc": loss_auc,
                "probe_auc": probe_auc,
                "shadow_auc": shadow_auc,
                "centroid_auc": centroid_auc,

                **{f"worstTPR@{float(ft)}": worst_tpr[float(ft)] for ft in FPR_POINTS},
                **{f"calWorstTPR_holdout@{float(ft)}": cal_worst_holdout[float(ft)] for ft in FPR_POINTS},

                "dist_logits_mse_all": dist["logits_mse_all"],
                "dist_embeds_mse_all": dist["embeds_mse_all"],
                "dist_mean_shift_emb": dist["mean_shift_emb"],
                "dist_cov_frob_emb": dist["cov_frob_emb"],

                "calib_frac": float(CALIBRATION_FRAC),
                "error": None,
            })

            # --- Aggregate ---
            agg.setdefault(dataset, {}).setdefault(method, {}).setdefault("loss_auc", []).append(loss_auc)
            if probe_auc is not None:
                agg[dataset][method].setdefault("probe_auc", []).append(probe_auc)
            if shadow_auc is not None:
                agg[dataset][method].setdefault("shadow_auc", []).append(shadow_auc)
            if centroid_auc is not None:
                agg[dataset][method].setdefault("centroid_auc", []).append(centroid_auc)

            if ra is not None:
                agg[dataset][method].setdefault("retain_acc", []).append(float(ra))
            if fa is not None:
                agg[dataset][method].setdefault("forget_acc", []).append(float(fa))

            for ft in FPR_POINTS:
                agg[dataset][method].setdefault(f"worstTPR@{float(ft)}", []).append(worst_tpr[float(ft)])
                agg[dataset][method].setdefault(f"calWorstTPR_holdout@{float(ft)}", []).append(cal_worst_holdout[float(ft)])

            # distance aggregates
            if dist["logits_mse_all"] is not None:
                agg[dataset][method].setdefault("dist_logits_mse_all", []).append(float(dist["logits_mse_all"]))
            if dist["embeds_mse_all"] is not None:
                agg[dataset][method].setdefault("dist_embeds_mse_all", []).append(float(dist["embeds_mse_all"]))
            if dist["mean_shift_emb"] is not None:
                agg[dataset][method].setdefault("dist_mean_shift_emb", []).append(float(dist["mean_shift_emb"]))
            if dist["cov_frob_emb"] is not None:
                agg[dataset][method].setdefault("dist_cov_frob_emb", []).append(float(dist["cov_frob_emb"]))

        except Exception as e:
            row["error"] = str(e)

        per_run.append(row)

    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset, by_method in agg.items():
        aggregate[dataset] = {}
        for method, metrics in by_method.items():
            aggregate[dataset][method] = {k: mean_std(v) for k, v in metrics.items()}

    out_json = os.path.join(RESULTS_DIR, f"usenix_privacy_utility_distance_{DATASET_FILTER or 'ALL'}.json")
    out_csv = os.path.join(RESULTS_DIR, f"usenix_privacy_utility_distance_{DATASET_FILTER or 'ALL'}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_run": per_run,
                "aggregate": aggregate,
                "utility_path": UTILITY_PATH,
                "utility_mode": util_mode,
                "fpr_points": FPR_POINTS,
                "calibration_frac": CALIBRATION_FRAC,
                "calibration_random_state": CALIBRATION_RANDOM_STATE,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "dataset", "method", "seed",
        "retain_acc", "forget_acc",
        "loss_auc", "probe_auc", "shadow_auc", "centroid_auc",
        *[f"worstTPR@{float(ft)}" for ft in FPR_POINTS],
        *[f"calWorstTPR_holdout@{float(ft)}" for ft in FPR_POINTS],
        "dist_logits_mse_all", "dist_embeds_mse_all", "dist_mean_shift_emb", "dist_cov_frob_emb",
        "calib_frac",
        "npz_file", "error",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_run:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    print("\n=== USENIX PRIVACY+UTILITY+LOWFPR + DIST-TO-RETR (MERGED) ===")
    print("AUC attacker-best (closer to 0.5 better). TPR@lowFPR lower is better.")
    print("Calibrated: thresholds learned on calib split; TPR reported on held-out test split.")
    print(f"[INFO] Utility file: {UTILITY_PATH} (mode={util_mode})")

    for dataset in sorted(aggregate.keys()):
        print(f"\nDATASET: {dataset}")
        per_method = {}

        for method in sorted(aggregate[dataset].keys()):
            m = aggregate[dataset][method]

            ra = m.get("retain_acc", {}).get("mean", None)
            fa = m.get("forget_acc", {}).get("mean", None)

            la = m.get("loss_auc", {}).get("mean", None)
            pa = m.get("probe_auc", {}).get("mean", None)
            sa = m.get("shadow_auc", {}).get("mean", None)
            ca = m.get("centroid_auc", {}).get("mean", None)

            ft0 = float(FPR_POINTS[0])
            worst_1e5 = m.get(f"worstTPR@{ft0}", {}).get("mean", None)
            cal_hold_1e5 = m.get(f"calWorstTPR_holdout@{ft0}", {}).get("mean", None)

            dlog = m.get("dist_logits_mse_all", {}).get("mean", None)
            demb = m.get("dist_embeds_mse_all", {}).get("mean", None)
            dmu = m.get("dist_mean_shift_emb", {}).get("mean", None)
            dcov = m.get("dist_cov_frob_emb", {}).get("mean", None)

            def _fmt(x, nd=6):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "NA"
                try:
                    return f"{float(x):.{nd}f}"
                except Exception:
                    return str(x)

            print(
                f"  {method:12s} "
                f"retainAcc {_fmt(ra,4)} | forgetAcc {_fmt(fa,4)} | "
                f"lossAUC {_fmt(la,4)} | probeAUC {_fmt(pa,4)} | shadowAUC {_fmt(sa,4)} | centroidAUC {_fmt(ca,4)} | "
                f"worstTPR@1e-5 {_fmt(worst_1e5,6)} | calWorstTPR_holdout@1e-5 {_fmt(cal_hold_1e5,6)} | "
                f"distLogitsMSE {_fmt(dlog,6)} | distEmbMSE {_fmt(demb,6)} | meanShift {_fmt(dmu,6)} | covFrob {_fmt(dcov,6)}"
            )

            per_method[method] = {
                "retain_acc": ra,
                "forget_acc": fa,
                "worstTPR@1e-05": worst_1e5,
                "calWorstTPR_holdout@1e-05": cal_hold_1e5,
                "dist_logits_mse": dlog,
                "dist_embeds_mse": demb,
                "dist_mean_shift_emb": dmu,
                "dist_cov_frob_emb": dcov,
            }

        out_prefix = os.path.join(PARETO_DIR, f"usenix_{dataset}_")
        save_pareto_plots(per_method, out_prefix)

        # Step-4 tables
        save_usenix_tables(dataset, aggregate[dataset], RESULTS_DIR)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")
    if SAVE_USENIX_TABLES and DATASET_FILTER:
        print(f"[INFO] Saved tables: tables_{DATASET_FILTER}_utility.csv / privacy_lowfpr.csv / distance.csv")


if __name__ == "__main__":
    main()
