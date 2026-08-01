# MU-ALIGN

<div align="center">

**Tail-Suppressed Multimodal Machine Unlearning with Improved Utility–Forgetting Trade-offs**

Abdullah Ahmad Khan · Hamid Laga · Mohammed Kaosar · Ferdous Sohel

[![Repository](https://img.shields.io/badge/GitHub-MU--Align-181717?logo=github)](https://github.com/abdullahak07/MU-Align)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)

</div>

---

## Overview

**MU-ALIGN** is an approximate multimodal machine-unlearning method that suppresses residual high-confidence behaviour on forgotten examples while preserving utility on retained data.

The method combines two coordinated optimisation paths.

### Forget path

- predictive uniformity, `L_unif`;
- confidence and logit-norm tail suppression, `L_tail`.

### Retain path

- supervised cross-entropy, `L_CE`;
- CORAL-inspired second-moment alignment, `L_align`;
- knowledge distillation from the frozen original model, `L_KD`.

The frozen original model acts as the teacher during unlearning. This requires no additional teacher-training stage, although teacher forward passes contribute to runtime.

---

## Method

The complete objective is:

```text
L = L_CE + lambda_a L_align + lambda_k L_KD
    + lambda_u L_unif + lambda_t L_tail
```

The paper configuration uses:

| Parameter | Value |
|---|---:|
| `lambda_u` | 1.0 |
| `lambda_t` | 1.0 |
| `lambda_a` | 0.1 |
| `lambda_k` | 0.5 |
| Distillation temperature `T` | 2 |
| Label smoothing | 0.05 |
| Unlearning epochs | 8 |
| Batch size | 32 |

---

## Main Results

### VQA-v2 answer-class deletion

Mean ± sample standard deviation over seeds 42, 123, and 5508:

| Method | ForgetAcc ↓ | RetainAcc ↑ |
|---|---:|---:|
| SCRUB (faithful) | 0.00036 ± 0.00063 | 0.3973 ± 0.0087 |
| RETR reference | 0.00000 ± 0.00000 | 0.5917 ± 0.0170 |
| **MU-ALIGN** | **0.00000 ± 0.00000** | **0.4533 ± 0.0066** |

At essentially matched aggregate forgetting, MU-ALIGN improves retained accuracy over faithful SCRUB by **5.60 percentage points**.

### Low-FPR membership inference

Under disjoint method-specific loss-based calibration:

| Method | TPR @ FPR 1e-2 ↓ | TPR @ FPR 5e-3 ↓ | TPR @ FPR 1e-3 ↓ |
|---|---:|---:|---:|
| ORIG | 0.01484 | 0.00911 | 0.00224 |
| RETR | 0.01022 | 0.00483 | 0.00098 |
| SCRUB | 0.01024 | 0.00549 | **0.00080** |
| **MU-ALIGN** | **0.00832** | **0.00474** | 0.00098 |

MU-ALIGN is lower than SCRUB at the reported FPR values `1e-2` and `5e-3`, while SCRUB is marginally lower at `1e-3`. Neither method uniformly dominates across all reported operating points.

### MLLMU-Bench generation family

| Method | Forget ↓ | Retain ↑ | Selective gap ↑ |
|---|---:|---:|---:|
| ORIG | 0.580 | 0.576 | -0.004 |
| RETR | 0.290 | 0.302 | +0.012 |
| Uniform-Target | 0.395 | 0.513 | +0.118 |
| **MU-ALIGN** | **0.362 ± 0.061** | **0.553 ± 0.014** | **+0.192 ± 0.050** |

This comparison is against the evaluated Uniform-Target baseline. Faithful generative SCRUB was not evaluated in this setting.

---

## Paper Figures

The repository includes the paper figures as PDF files. Click a figure name to open the vector-quality PDF.

| Figure | Description |
|---|---|
| [**flow.pdf**](flow.pdf) | MU-ALIGN pipeline: forget path, retain path, and joint optimisation |
| [**fig_vqa_corrected.pdf**](fig_vqa_corrected.pdf) | Seed-level VQA-v2 retained-accuracy comparison |
| [**fig_score_dist.pdf**](fig_score_dist.pdf) | Seed-42 loss-based MIA score distributions |
| [**fig_lowfpr.pdf**](fig_lowfpr.pdf) | Low-FPR membership-inference curves |
| [**fig_mllmu.pdf**](fig_mllmu.pdf) | MLLMU-Bench generation-family comparison |
| [**fig_oracle.pdf**](fig_oracle.pdf) | RETR-aligned oracle diagnostic, retained for archival comparison |

> GitHub does not reliably render PDF files inline inside a README. The links above open the original vector PDFs directly.

---

## Experimental Scope

### Primary VQA-v2 setting

The controlled model uses:

- a frozen ImageNet-pretrained ResNet-18 visual encoder;
- a GRU question encoder with hidden dimension 512;
- a two-layer multimodal fusion classifier;
- a 20,000-example VQA-v2 subset;
- an answer-class deletion protocol in which examples with majority answer `"yes"` form the forget set.

The primary comparison includes ORIG, RETR, faithful SCRUB, and MU-ALIGN.

This benchmark is a controlled answer-class deletion setting and should not be interpreted as evidence of arbitrary sample-level or subject-level erasure.

### Oracle diagnostic

The RETR-aligned oracle stress test is a development-seed diagnostic only. It uses RETR during optimisation and is not a deployable unlearning method.

The selected checkpoint passes five of eight gates but fails validation utility, RETR decision agreement, and noncollapse. This demonstrates that matching aggregate ForgetAcc does not by itself establish retraining-aligned behavioural equivalence.

---

## Installation

```bash
git clone https://github.com/abdullahak07/MU-Align.git
cd MU-Align
```

Create and activate a virtual environment.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The exact commands used to reproduce individual tables and figures should be documented alongside their corresponding scripts and configuration files in the archival release.

---

## Reproducibility Checklist

For each reported experiment, the repository should provide:

- configuration file;
- random seed;
- split manifest;
- checkpoint path or download instructions;
- evaluation command;
- raw metric output;
- generated table or figure.

Primary VQA-v2 seeds:

```text
42
123
5508
```

SCRUB hyperparameters were selected on seed 42 and frozen for seeds 123 and 5508.

---

## Recommended Repository Layout

```text
MU-Align/
├── data/
├── models/
├── methods/
│   ├── mu_align/
│   └── scrub/
├── experiments/
│   ├── vqav2/
│   └── mllmu/
├── evaluation/
│   ├── utility/
│   ├── membership_inference/
│   ├── tail_analysis/
│   └── oracle_audit/
├── configs/
├── results/
├── scripts/
├── flow.pdf
├── fig_vqa_corrected.pdf
├── fig_score_dist.pdf
├── fig_lowfpr.pdf
├── fig_mllmu.pdf
├── fig_oracle.pdf
├── requirements.txt
└── README.md
```

---

## Limitations

The current study has several important limitations:

- the primary benchmark uses answer-class deletion;
- the retained `"yes"`-answer diagnostic was not available under the evaluated split;
- the primary model is a compact ResNet-GRU-MLP multimodal classifier;
- only three primary seeds are reported;
- NPO is not included as a primary baseline;
- faithful generative SCRUB is not evaluated on MLLMU-Bench;
- the privacy evaluation is limited to loss-based attacks and confidence-tail diagnostics;
- the seed-42 tail analysis is exploratory;
- the RETR-assisted oracle audit is diagnostic and non-deployable.

---

## Citation

Publication metadata will be updated after acceptance. For the submitted manuscript:

```bibtex
@article{khan2026mualign,
  title   = {MU-ALIGN: Tail-Suppressed Multimodal Machine Unlearning with Improved Utility--Forgetting Trade-offs},
  author  = {Khan, Abdullah Ahmad and Laga, Hamid and Kaosar, Mohammed and Sohel, Ferdous},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Authors

- Abdullah Ahmad Khan
- Hamid Laga
- Mohammed Kaosar
- Ferdous Sohel

School of Information Technology  
Murdoch University  
Perth, Western Australia, Australia

---

## Contact

For questions, reproducibility concerns, or bug reports, open an issue:

https://github.com/abdullahak07/MU-Align/issues

---

## Licence

Add the final repository licence before archival release. Ensure that it is compatible with all included datasets, pretrained models, and third-party code.
