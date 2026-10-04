"""Same noise equation across modality layouts; no pretrained codec or quality claim."""
import json

import torch

from fm_tutorial.diffusion.continuous import GaussianDiffusion
from fm_tutorial.diffusion.discrete import CategoricalDiffusion, mask_tokens


def main():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    gaussian = GaussianDiffusion(steps=100, schedule="cosine")
    # Random embedding table illustrates a representation, not a trained text codec.
    embedding = torch.nn.Embedding(16, 8)
    ids = torch.tensor([[1, 3, 5, 2], [1, 4, 6, 2]])
    fixtures = {
        "text_embeddings": embedding(ids).detach(),              # [B,L,d]
        "image_latent": torch.randn(2, 4, 8, 8),                  # [B,C,H,W]
        "table_numeric": torch.randn(2, 3),                      # [B,columns]
        "video_latent": torch.randn(2, 4, 4, 8, 8),              # [B,C,F,H,W]
        "4d_trajectories": torch.randn(2, 4, 16, 3),              # [B,F,N,xyz], aligned point IDs
    }
    report = {"scope": "synthetic tensor + oracle-noise diagnostic; no learned reconstruction or generation",
              "continuous": {}}
    for name, x0 in fixtures.items():
        t = torch.tensor([20, 70])
        noise = torch.randn_like(x0)
        xt = gaussian.q_sample(x0, t, noise)
        oracle_x0 = gaussian.predict_x0(xt, t, noise)
        report["continuous"][name] = {"shape": list(xt.shape),
                                      "oracle_max_error": float((x0 - oracle_x0).abs().max())}
    categories = torch.tensor([[0, 2], [1, 0]])
    categorical = CategoricalDiffusion(vocab_size=3, steps=10)
    t = torch.tensor([2, 8])
    corrupted = categorical.q_sample(categories, t)
    posterior = categorical.posterior_probs(categories, corrupted, t)
    report["table_categorical"] = {"shape": list(corrupted.shape),
                                   "posterior_normalized": bool(torch.allclose(posterior.sum(-1), torch.ones_like(categories, dtype=posterior.dtype)))}
    condition = torch.zeros_like(ids, dtype=torch.bool)
    condition[:, 0] = True
    masked, selected = mask_tokens(ids, torch.tensor([1.0, 1.0]), mask_id=0, condition_mask=condition)
    report["masked_text"] = {"shape": list(masked.shape), "masked_positions": int(selected.sum()),
                             "condition_preserved": bool(torch.equal(masked[condition], ids[condition]))}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
