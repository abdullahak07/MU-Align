# models/vqa_model.py
import torch
import torch.nn as nn
import torchvision.models as models


class VQAModel(nn.Module):
    """
    VQA model exposing decomposed embeddings:
      - img_emb:  pre-fusion image embedding
      - txt_emb:  pre-fusion text embedding
      - fused_emb: fused embedding (pre-classifier)

    Default forward:
      forward(image, question) -> (logits, fused_emb)

    Dict forward:
      forward(image, question, return_dict=True) -> dict

    MU_ALIGN-only training-time privacy (forget-only, train-only):
      - SCRUB-like shrink (optional)
      - L2-clip
      - Gaussian noise
    """

    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        img_embed: int = 512,
        txt_embed: int = 512,
        fused_dim: int = 512,
        p_drop: float = 0.25,
    ):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(base.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.img_proj = nn.Linear(base.fc.in_features, img_embed)

        self.word_emb = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.gru = nn.GRU(300, txt_embed, batch_first=True)

        self.fusion = nn.Linear(img_embed + txt_embed, fused_dim)
        self.fused_ln = nn.LayerNorm(fused_dim)
        self.fused_drop = nn.Dropout(p_drop)
        self.cls = nn.Linear(fused_dim, num_answers)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)
        feats = feats.view(feats.size(0), -1)
        feats = torch.relu(self.img_proj(feats))
        return feats

    def encode_text(self, q: torch.Tensor) -> torch.Tensor:
        emb = self.word_emb(q)
        _, h = self.gru(emb)
        feats = torch.relu(h.squeeze(0))
        return feats

    def fuse(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.relu(self.fusion(torch.cat([img_feat, txt_feat], dim=1)))
        fused = self.fused_ln(fused)
        fused = self.fused_drop(fused)
        return fused

    def classify_from_fused(self, fused_emb: torch.Tensor) -> torch.Tensor:
        return self.cls(fused_emb)

    def apply_forget_privacy(
        self,
        fused: torch.Tensor,
        forget_mask,
        noise_std: float,
        clip_norm: float,
        scrub_alpha: float = 0.0,
    ) -> torch.Tensor:
        """
        Forget-only privacy transform (TRAIN-TIME ONLY):

        1) SCRUB-like shrink:
           z_f <- (1 - a) * z_f + a * mean(z_retain)   (computed on the same batch)
           This reduces embedding separability exploited by probe/shadow attacks.

        2) L2 clip:
           z_f <- z_f * min(1, clip_norm / ||z_f||)

        3) Add noise:
           z_f <- z_f + N(0, noise_std^2)

        Notes:
        - If the batch has no retain samples, scrub step is skipped (safe).
        - This function never runs in eval mode, ensuring evaluation is clean.
        """
        if (not self.training) or forget_mask is None:
            return fused
        if not torch.is_tensor(forget_mask) or (not forget_mask.any()):
            return fused

        # If all knobs off, do nothing
        if (noise_std is None or noise_std <= 0.0) and (clip_norm is None or clip_norm <= 0.0) and (scrub_alpha is None or scrub_alpha <= 0.0):
            return fused

        z = fused.clone()
        f = forget_mask.bool()
        r = (~f)

        # 1) SCRUB-like shrink toward retain centroid (batch-local)
        a = float(scrub_alpha) if scrub_alpha is not None else 0.0
        if a > 0.0 and r.any():
            retain_mean = z[r].mean(dim=0, keepdim=True)  # (1, d)
            z[f] = (1.0 - a) * z[f] + a * retain_mean

        # 2) L2 clip (forget-only)
        if clip_norm is not None and float(clip_norm) > 0.0:
            cn = float(clip_norm)
            norms = z[f].norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = (cn / norms).clamp_max(1.0)
            z[f] = z[f] * scale

        # 3) Noise (forget-only)
        if noise_std is not None and float(noise_std) > 0.0:
            ns = float(noise_std)
            z[f] = z[f] + ns * torch.randn_like(z[f])

        return z

    def forward(self, image: torch.Tensor, question: torch.Tensor, return_dict: bool = False):
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(question)
        fused_emb = self.fuse(img_emb, txt_emb)
        logits = self.classify_from_fused(fused_emb)

        if return_dict:
            return {
                "logits": logits,
                "img_emb": img_emb,
                "txt_emb": txt_emb,
                "fused_emb": fused_emb,
            }
        return logits, fused_emb

    def forward_decomposed(self, image: torch.Tensor, question: torch.Tensor):
        out = self.forward(image, question, return_dict=True)
        return out["logits"], out["img_emb"], out["txt_emb"], out["fused_emb"]
