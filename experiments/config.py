# experiments/config.py
import os
import torch

# =================================================
# Paths
# =================================================
VQAV2_ROOT = r"D:\vqav2"
SAVE_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVE_ROOT, exist_ok=True)

PRIVACY_CACHE_DIR = os.path.join(SAVE_ROOT, "privacy_cache")
os.makedirs(PRIVACY_CACHE_DIR, exist_ok=True)

# =================================================
# Dataset modes
# =================================================
DATASETS = ["vqa_v2_20k"]

# =================================================
# Dataset caps
# =================================================
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 5000

# =================================================
# Core hyperparameters
# =================================================
FORGET_ANSWER = "yes"
TOP_K_ANSWERS = 1000
MAX_QUESTION_LEN = 16

LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================
# Batch size
# =================================================
BATCH_SIZE = 32

# =================================================
# Epochs
# =================================================
EPOCHS_ORIG = 6
EPOCHS_RETR = 6
EPOCHS_MUALIGN = 8

# =================================================
# DataLoader performance (Windows-safe)
# =================================================
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2

# =================================================
# Experiment grid
# =================================================
SEEDS = [42, 123, 5508]

# =================================================
# IMPORTANT: evaluation-time privacy must be OFF globally
# =================================================
PRIVACY_ON_TRAIN = False
PRIVACY_ON_EVAL = False
PRIVACY_EMB_CLIP = 0.0
PRIVACY_EMB_NOISE_STD = 0.0

# =================================================
# MU_ALIGN (alignment + uniformity)
# =================================================
MUALIGN_LAMBDA_UNIF = 6.0
MUALIGN_LAMBDA_CSSA = 0.01
MUALIGN_LAMBDA_CMFC = 0.01
MUALIGN_LAMBDA_CORAL = 0.02
MUALIGN_LABEL_SMOOTHING = 0.05

# =================================================
# MU_ALIGN: forget-only privacy in fused space (train-time)
# =================================================
# You already achieved held-out calWorstTPR@1e-5 <= 0.01.
# Now prioritize retainAcc by dialing back the most utility-harming knobs
# while preserving low-FPR via the tail caps below.
MUALIGN_FORGET_NOISE_STD = 0.10     # was 0.20 (too aggressive)
MUALIGN_FORGET_CLIP_NORM = 2.0      # was 1.2 (too tight)
MUALIGN_FORGET_SCRUB_ALPHA = 0.0    # disable scrub-like suppression inside MU_ALIGN

# =================================================
# MU_ALIGN low-FPR hardening: LOGIT confidence tails (forget-only)
# =================================================
# Keep these ON, but not overly strong.
MUALIGN_FORGET_CONF_CAP = 0.35      # was 0.30 (too strict)
MUALIGN_LAMBDA_FCONF = 8.0          # was 12.0

MUALIGN_FORGET_MARGIN_CAP = 1.0
MUALIGN_LAMBDA_FMARG = 0.0          # optional; keep off unless you see margin spikes

# =================================================
# MU_ALIGN low-FPR hardening: EMBEDDING tails (forget-only)
# =================================================
# Mildly constrain fused embedding norm to suppress probe/shadow attacks,
# but avoid crushing representation quality.
MUALIGN_FORGET_EMBNORM_CAP = 12.0   # was 10.0
MUALIGN_LAMBDA_FEMBNORM = 2.0       # was 3.0

# Optional logit norm cap (keep mild)
MUALIGN_FORGET_LOGITNORM_CAP = 25.0
MUALIGN_LAMBDA_FLOGITNORM = 0.5

# =================================================
# KD-to-teacher (retain stabilization)
# =================================================
MUALIGN_LAMBDA_KD = 0.6
MUALIGN_KD_TEMP = 2.0

# =================================================
# Methods
# =================================================
METHODS = [
    "RETR",
    "MU_ALIGN",
     "SCRUB",       # enable only for comparison runs
    # "AMNESIAC",
    # "FISHER",
    # "MULTIDELETE",
]
