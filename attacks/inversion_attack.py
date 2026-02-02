# mu_align/attacks/inversion_attack.py

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def inversion_linear_attack(outputs, test_size: float = 0.3, random_state: int = 42):
    """
    Linear inversion attack on embeddings.

    Idea:
      - Treat the joint embedding (image+text fused representation)
        as a lossy encoding of the underlying target y.
      - Train a simple linear classifier (Logistic Regression) to
        predict y from embeddings.
      - Higher accuracy ⇒ more information about y is still present
        in the representation ⇒ worse forgetting.

    Parameters
    ----------
    outputs : dict
        Expected keys:
          - "embeds": [N, D] tensor
          - "labels": [N] tensor

    test_size : float
        Fraction of data used as attacker test set.

    random_state : int
        Seeding for train/test split.

    Returns
    -------
    acc : float
        Accuracy of the linear inversion attacker on its test split.
        If only one class is present, returns NaN.
    """
    if "embeds" not in outputs or "labels" not in outputs:
        raise ValueError(
            "inversion_linear_attack expects outputs with keys "
            "'embeds' and 'labels'."
        )

    # Extract embeddings and labels
    emb = outputs["embeds"]
    labels = outputs["labels"]

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    emb = np.asarray(emb)
    labels = np.asarray(labels)

    # Guard: need at least 2 classes
    if len(np.unique(labels)) < 2:
        return float("nan")

    # Attacker train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        emb,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc


# ---------------------------------------------------------------------------
# Backwards-compatible alias if other code uses `inversion_attack`
# ---------------------------------------------------------------------------

def inversion_attack(*args, **kwargs):
    """
    Thin wrapper around inversion_linear_attack for backwards compatibility.
    """
    return inversion_linear_attack(*args, **kwargs)
