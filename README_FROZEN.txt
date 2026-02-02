# MU-ALIGN-SAFE

This folder contains the **final, frozen, and USENIX-safe implementation** of **MU-ALIGN**, a **machine unlearning method** designed to achieve **strong low-FPR privacy guarantees** while preserving task utility.

This version is intended for:
- paper tables and figures
- final result generation
- artifact evaluation
- archival and reproducibility

⚠️ **Do not modify this folder once results are frozen.**

---

## 1. What MU-ALIGN Is

**MU-ALIGN is a machine unlearning algorithm**, not a metric.

It performs **selective forgetting** by explicitly targeting the signals exploited by **membership inference attacks (MIAs)** in the **low–false-positive–rate regime**, which is the regime required by security venues such as **USENIX Security**.

---

## 2. Core Idea

MU-ALIGN enforces forgetting while maintaining utility through **three coordinated mechanisms**:

1. **Forget-Side Suppression (Privacy)**
   - Uniform prediction enforcement on forgotten samples
   - Forget-only confidence capping
   - Forget-only embedding and logit norm suppression
   - Forget-only noise injection and clipping

2. **Retain-Side Stabilization (Utility)**
   - Cross-modal semantic alignment (CSSA)
   - Cross-modal feature consistency (CMFC)
   - CORAL covariance alignment
   - Knowledge distillation to a retrained (RETR) teacher

3. **Gradient Isolation**
   - Explicit **forget ⟂ retain gradient orthogonalization**
   - Prevents forgetting updates from harming retained representations

These mechanisms are **jointly optimized**, not applied as independent heuristics.

---

## 3. Why This Is Publishable (USENIX Perspective)

Security reviewers care about:

- **Low-FPR privacy**, not average AUC
- **Worst-case attackers**, not single probes
- **Held-out calibration**, not optimistic thresholds
- **Faithfulness to retraining**, not accuracy alone

MU-ALIGN satisfies all of these:

- Evaluates **worst-case TPR** across multiple attackers
- Reports **calibrated TPR on a held-out split**
- Includes **distance-to-retraining metrics**
- Demonstrates a **Pareto improvement** over SCRUB and other baselines

---

## 4. Dataset and Setting

- **Dataset**: VQA-v2 (20k subset)
- **Forgetting target**: answers equal to `"yes"`
- **Evaluation**: 3 random seeds (42, 123, 5508)
- **Training**: no retraining from scratch

---

## 5. Final Headline Results (Frozen)

Averaged over **3 seeds**:

- **calWorstTPR@1e-5 (held-out)**: ≈ **0.003–0.004**
- **worstTPR@1e-5 (ROC)**: ≈ **0.003–0.004**
- **Forget accuracy**: ≈ **0.0**
- **Retain accuracy**: ≈ **0.41–0.42**
- **Distance-to-RETR (Emb MSE)**: lower than SCRUB

MU-ALIGN **outperforms SCRUB** in:
- calibrated low-FPR privacy
- distance to retraining
- retain utility

---

## 6. Folder Contents

mu_align_safe/
├── experiments/
│ ├── config.py # final frozen hyperparameters
│ ├── methods.py # MU-ALIGN training logic (with grad-ortho)
│ └── run_all.py
├── models/
│ └── losses.py # final mu_align_loss signature
├── attacks/
│ └── usenix_pareto_distance.py # USENIX-style evaluation (held-out calibration)
├── results/
│ ├── summary_all_methods.json
│ └── privacy_cache/
│ ├── usenix_privacy_utility_distance_vqa_v2_20k.json
│ ├── usenix_privacy_utility_distance_vqa_v2_20k.csv
│ ├── tables_vqa_v2_20k_utility.csv
│ ├── tables_vqa_v2_20k_privacy_lowfpr.csv
│ └── tables_vqa_v2_20k_distance.csv
└── README.md ← this file



---

## 7. Reproducing Results

1. Train models:
```bash
python main.py

2.Run privacy + utility evaluation:
python attacks/usenix_pareto_distance.py

3.Outputs are saved to:
results/privacy_cache/

---


## 8. Important Notes

Evaluation-time privacy is disabled (PRIVACY_ON_EVAL = False)

All reported calibrated TPRs use held-out calibration

All metrics reflect worst-case attackers

This version is final and frozen


---



9. Terminology Guidance (for Writing)

Use the following terms consistently in the paper:

✅ unlearning method / algorithm

✅ low-FPR unlearning

❌ unlearning metric

❌ post-processing defense

Recommended phrasing:

“MU-ALIGN is a low-FPR machine unlearning algorithm that explicitly suppresses membership signals while preserving retraining faithfulness.”

---

10. Status

Status: ✅ Final
Ready for: USENIX Security submission, artifact evaluation, rebuttal

