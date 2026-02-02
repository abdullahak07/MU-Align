# mu_align/experiments/run_ablations.py

"""
Ablation runner: you can modify this to toggle each component:
  - no CSSA
  - no CM-FC
  - no uniform
  - random suppression
For now it's just a placeholder; reuse train_mu_align with different
hyperparameters and/or custom loss variants.
"""

def run_ablations():
    raise NotImplementedError(
        "Wire ablations by replacing mu_align_loss with partial variants."
    )


if __name__ == "__main__":
    run_ablations()
