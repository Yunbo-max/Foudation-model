"""Conditional diffusion on two synthetic 2D Gaussian classes; no benchmark claim."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fm_tutorial.diffusion.continuous import GaussianDiffusion
from fm_tutorial.diffusion.conditioning import FeatureConcat, AddCondition, FiLM, AdaLN, classifier_free_guidance


class ConditionalPointDenoiser(nn.Module):
    """Shared noisy-point network with an interchangeable global condition block.

    Time enters the data path separately from semantic class. Class 2 is a
    learned null condition, used in both dropout training and CFG sampling.
    """

    def __init__(self, method, diffusion_steps):
        super().__init__()
        self.steps = diffusion_steps
        self.input = nn.Linear(5, 32)
        self.label = nn.Embedding(3, 32)
        self.condition = {"cat": FeatureConcat, "add": AddCondition,
                          "film": FiLM, "adaln": AdaLN}[method](32, 32)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, xt, t, labels):
        time = (t.float() + 1) / self.steps
        time_features = torch.stack((time, (math.pi * time).sin(), (math.pi * time).cos()), -1)
        h = self.input(torch.cat((xt, time_features), -1))[:, None, :]
        return self.output(self.condition(h, self.label(labels))[:, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("cat", "add", "film", "adaln"), default="adaln")
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/conditioning-demo"))
    args = parser.parse_args()
    if min(args.train_steps, args.diffusion_steps, args.sample_steps, args.samples) < 1:
        parser.error("step counts and samples must be positive")
    if args.sample_steps > args.diffusion_steps:
        parser.error("sample steps must not exceed diffusion steps")
    if not 0 < args.condition_dropout < 1:
        parser.error("condition dropout must lie strictly in (0,1) to train both branches")
    if not math.isfinite(args.guidance_scale):
        parser.error("guidance scale must be finite")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    # Explicit toy schedule: a near-zero cosine endpoint amplifies small
    # epsilon errors in a short-trained model. This prior is approximate;
    # changing steps also changes terminal SNR because betas are not rescaled.
    process = GaussianDiffusion(steps=args.diffusion_steps, schedule="linear", beta_end=0.12)
    model = ConditionalPointDenoiser(args.method, args.diffusion_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    metrics, null_count = [], 0
    for step in range(1, args.train_steps + 1):
        labels = torch.randint(2, (64,))
        x0 = centers[labels] + 0.1 * torch.randn(64, 2)
        noise, t = torch.randn_like(x0), torch.randint(process.steps, (64,))
        dropped = torch.rand(64) < args.condition_dropout
        null_count += int(dropped.sum())
        # Drop only semantic class. Actual target/noise and timestep stay intact.
        training_labels = labels.masked_fill(dropped, 2)
        prediction = model(process.q_sample(x0, t, noise), t, training_labels)
        loss = F.mse_loss(prediction, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.append({"step": step, "epsilon_mse": float(loss.detach())})
    if not 0 < null_count < 64 * args.train_steps:
        raise RuntimeError("Both null and conditional training examples are required before CFG sampling; adjust dropout or train longer")
    model.eval()
    sampled, means, initial_noises = [], [], []
    for label in range(2):
        condition = torch.full((args.samples,), label, dtype=torch.long)
        null = torch.full_like(condition, 2)

        def predictor(x, t):
            # Two logical model branches, evaluated in one doubled batch.
            both = model(torch.cat((x, x)), torch.cat((t, t)), torch.cat((null, condition)))
            unconditional, conditional = both.chunk(2)
            return classifier_free_guidance(unconditional, conditional, args.guidance_scale)

        torch.manual_seed(args.seed + 1000)
        initial_noises.append(torch.randn(args.samples, 2))
        torch.manual_seed(args.seed + 1000)  # DDIM sample draws this same initial array.
        values = process.sample(predictor, (args.samples, 2), sampler="ddim", sampling_steps=args.sample_steps)
        sampled.append(values)
        means.append(values.mean(0).tolist())
    samples = torch.stack(sampled)
    if not torch.isfinite(samples).all():
        raise RuntimeError("Non-finite generated values: inspect learned field and sampler")
    report = {"scope": "synthetic two-class algorithm demonstration; no held-out quality metric",
              "method": args.method, "train_steps": args.train_steps, "diffusion_steps": args.diffusion_steps,
              "sample_steps": args.sample_steps, "guidance_scale": args.guidance_scale,
              "condition_dropout": args.condition_dropout, "null_condition_training_examples": null_count,
              "conditional_training_examples": 64 * args.train_steps - null_count,
              "schedule": {"name": "linear", "beta_start": 1e-4, "beta_end": 0.12,
                           "terminal_alpha_bar": float(process.alpha_bar[-1]),
                           "initial_prior": "standard Gaussian, approximate terminal marginal"},
              "device": "cpu", "batch_size": 64, "learning_rate": 0.002,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "paired_initial_noise_equal": bool(torch.equal(*initial_noises)), "seed": args.seed,
              "parameters": sum(p.numel() for p in model.parameters()), "sample_shape": list(samples.shape),
              "source_centers": centers.tolist(), "generated_class_means": means,
              "first_batch_loss": metrics[0], "last_batch_loss": metrics[-1],
              "sampler_callback_calls_per_class": args.sample_steps,
              "logical_prediction_branches_per_class": 2 * args.sample_steps}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "samples.npy", samples.cpu().numpy())
    with (args.out_dir / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epsilon_mse"])
        writer.writeheader()
        writer.writerows(metrics)
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "report": report}, args.out_dir / "denoiser.pt")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
