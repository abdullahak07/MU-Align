# attacks/privacy_separability_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============== CONFIG ==============
RESULTS_DIR = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\mu_align\results\privacy_cache"
DATASET = "vqa_v2_small"
SEED = 42

METHOD_A = "MU_ALIGN"
METHOD_B = "RETR"   # compare against baseline

OUT_DIR = os.path.join(RESULTS_DIR, "plots_separability")
# ====================================


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def load_npz(dataset, method, seed):
    npz_path = os.path.join(RESULTS_DIR, f"{dataset}_{method}_seed{seed}_forget_outputs.npz")
    mask_path = os.path.join(RESULTS_DIR, f"{dataset}_seed{seed}_member_mask.npy")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    d = np.load(npz_path)
    member = np.load(mask_path).astype(int)
    logits = d["logits"]
    labels = d["labels"].astype(int)
    embeds = d["embeds"]

    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-12)
    emb_norm = np.linalg.norm(embeds, axis=1)

    return {"loss": loss, "emb_norm": emb_norm, "embeds": embeds, "member": member}


def hist_plot(x_mem, x_non, title, xlabel, out_png):
    plt.figure()
    plt.hist(x_non, bins=40, alpha=0.6, label="non-member")
    plt.hist(x_mem, bins=40, alpha=0.6, label="member")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def pca_plot(embeds, member, title, out_png):
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(embeds)

    plt.figure()
    plt.scatter(z[member == 0, 0], z[member == 0, 1], s=8, alpha=0.6, label="non-member")
    plt.scatter(z[member == 1, 0], z[member == 1, 1], s=8, alpha=0.6, label="member")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    A = load_npz(DATASET, METHOD_A, SEED)
    B = load_npz(DATASET, METHOD_B, SEED)

    # --- Loss histograms ---
    hist_plot(
        A["loss"][A["member"] == 1], A["loss"][A["member"] == 0],
        title=f"{METHOD_A}: member vs non-member loss",
        xlabel="NLL loss",
        out_png=os.path.join(OUT_DIR, f"{DATASET}_{METHOD_A}_seed{SEED}_loss_hist.png"),
    )
    hist_plot(
        B["loss"][B["member"] == 1], B["loss"][B["member"] == 0],
        title=f"{METHOD_B}: member vs non-member loss",
        xlabel="NLL loss",
        out_png=os.path.join(OUT_DIR, f"{DATASET}_{METHOD_B}_seed{SEED}_loss_hist.png"),
    )

    # --- Embedding norm histograms ---
    hist_plot(
        A["emb_norm"][A["member"] == 1], A["emb_norm"][A["member"] == 0],
        title=f"{METHOD_A}: member vs non-member embedding norm",
        xlabel="||embedding||2",
        out_png=os.path.join(OUT_DIR, f"{DATASET}_{METHOD_A}_seed{SEED}_embnorm_hist.png"),
    )
    hist_plot(
        B["emb_norm"][B["member"] == 1], B["emb_norm"][B["member"] == 0],
        title=f"{METHOD_B}: member vs non-member embedding norm",
        xlabel="||embedding||2",
        out_png=os.path.join(OUT_DIR, f"{DATASET}_{METHOD_B}_seed{SEED}_embnorm_hist.png"),
    )

    # --- PCA scatter plots ---
    pca_plot(
        A["embeds"], A["member"],
        title=f"{METHOD_A}: PCA(embeddings) colored by membership",
        out_png=os.path.join(OUT_DIR, f"{DATASET}_{METHOD_A}_seed{SEED}_pca.png"),
    )
    pca_plot(
        B["embeds"], B["member"],
        title=f"{METHOD_B}: PCA(embeddings) colored by membership",
        out_png=os.path.join(OUT_DIR, f"{DATASET}_{METHOD_B}_seed{SEED}_pca.png"),
    )

    print("Saved plots to:", OUT_DIR)
