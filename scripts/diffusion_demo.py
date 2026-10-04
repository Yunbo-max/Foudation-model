"""Train small denoisers and execute samplers. Synthetic demos are not benchmarks.

Gaussian/score modes accept preprocessed float arrays [N,...] via --data-npy.
The default two-dimensional mixture and fixed-length text are explicit teaching
fixtures. All metrics below describe that fixture and this run only.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from fm_tutorial.diffusion.continuous import GaussianDiffusion, VPSDE
from fm_tutorial.diffusion.discrete import (
    CategoricalDiffusion, MaskedDenoiser, mask_tokens, masked_loss, sample_masked,
)
from fm_tutorial.diffusion.models import TimeMLP


def continuous_data(path, device):
    """Return a training batch function and the event shape, with provenance."""
    if path:
        array = np.load(path, allow_pickle=False)
        if (array.ndim < 2 or len(array) < 2 or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.complexfloating)):
            raise ValueError("--data-npy needs a numeric preprocessed array [N,...], N>=2")
        if not np.isfinite(array).all():
            raise ValueError("--data-npy must contain finite values")
        data = torch.tensor(array, dtype=torch.float32, device=device)

        def batch(count):
            return data[torch.randint(len(data), (count,), device=device)]

        provenance = {"data_source": "user-supplied preprocessed numeric array",
                      "data_path": str(Path(path).resolve()),
                      "data_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      "data_shape": list(array.shape), "preprocessing": "performed by user; no automatic fit"}
        return batch, tuple(data.shape[1:]), provenance

    angles = torch.arange(8, device=device) * (2 * math.pi / 8)
    means = torch.stack((angles.cos(), angles.sin()), -1)

    def batch(count):
        indices = torch.randint(8, (count,), device=device)
        return means[indices] + 0.05 * torch.randn(count, 2, device=device)

    return batch, (2,), {"data_source": "synthetic eight-Gaussian 2D teaching fixture",
                        "components": 8, "radius": 1.0, "component_std": 0.05}


def text_data(device, include_mask=True):
    """Fixed-length whitespace fixtures, with no claimed language capability."""
    lines = [
        "the small cat likes warm milk", "the happy dog likes cool water",
        "the small dog likes warm milk", "the happy cat likes cool water",
        "the young bird sees blue skies", "the small bird sees green trees",
        "the young cat sees green trees", "the happy dog sees blue skies",
    ]
    words = sorted(set(" ".join(lines).split()))
    vocabulary = (["<mask>"] if include_mask else []) + words
    lookup = {word: index for index, word in enumerate(vocabulary)}
    data = torch.tensor([[lookup[word] for word in line.split()] for line in lines], device=device)
    return data, vocabulary, lookup


def train_continuous(args, device):
    batch, event_shape, provenance = continuous_data(args.data_npy, device)
    model = TimeMLP(math.prod(event_shape), hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    gaussian = GaussianDiffusion(steps=args.diffusion_steps, schedule="cosine")
    vp = VPSDE()
    records = []
    for step in range(1, args.train_steps + 1):
        x0 = batch(args.batch_size)
        noise = torch.randn_like(x0)
        if args.mode == "gaussian":
            t = torch.randint(gaussian.steps, (len(x0),), device=device)
            xt = gaussian.q_sample(x0, t, noise)
            time_input = (t.float() + 1) / gaussian.steps
        else:
            time_input = 0.001 + 0.999 * torch.rand(len(x0), device=device)
            xt = vp.marginal(x0, time_input, noise)
        loss = F.mse_loss(model(xt, time_input), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        records.append({"step": step, "epsilon_mse": float(loss.detach())})
    model.eval()
    shape = (args.samples,) + event_shape
    with torch.no_grad():
        if args.mode == "gaussian":
            predictor = lambda x, t: model(x, (t.float() + 1) / gaussian.steps)
            samples = gaussian.sample(predictor, shape, device=str(device), sampler=args.sampler,
                                      sampling_steps=args.sample_steps if args.sampler == "ddim" else None)
            extra = {"alpha_bar_terminal": float(gaussian.alpha_bar[-1]),
                     "time_convention": "array 0..T-1 maps to mathematical 1..T",
                     "network_evaluations_per_sample_batch": args.sample_steps if args.sampler == "ddim" else gaussian.steps}
        else:
            score = lambda x, t: vp.score_from_epsilon(model(x, t), t)
            samples = vp.sample(score, shape, device=str(device), steps=args.sample_steps,
                                probability_flow=args.sampler == "ode", t_min=0.001)
            extra = {"solver": "Euler ODE" if args.sampler == "ode" else "Euler-Maruyama reverse SDE",
                     "t_min": 0.001, "network_evaluations_per_sample_batch": args.sample_steps}
    return model, records, samples, provenance | extra, None


def train_text(args, device):
    data, vocabulary, lookup = text_data(device, include_mask=args.mode == "masked")
    model = MaskedDenoiser(len(vocabulary), data.shape[1], dim=32, heads=4, layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    discrete = CategoricalDiffusion(len(vocabulary), steps=args.diffusion_steps)
    records = []
    for step in range(1, args.train_steps + 1):
        x0 = data[torch.randint(len(data), (args.batch_size,), device=device)]
        if args.mode == "masked":
            t = torch.rand(len(x0), device=device)
            xt, active = mask_tokens(x0, t, mask_id=0)
            loss = masked_loss(model(xt, t), x0, active)
        else:
            t = torch.randint(1, discrete.steps + 1, (len(x0),), device=device)
            xt = discrete.q_sample(x0, t)
            logits = model(xt, t.float() / discrete.steps)
            # Auxiliary clean-token CE: not the full D3PM variational bound.
            loss = F.cross_entropy(logits.float().reshape(-1, len(vocabulary)), x0.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        records.append({"step": step, "denoising_ce": float(loss.detach())})
    model.eval()
    calls = 0

    def counted_predictor(x, t):
        nonlocal calls
        calls += 1
        return model(x, t)

    with torch.no_grad():
        if args.mode == "masked":
            condition = torch.full((args.samples, data.shape[1]), 0, device=device)
            condition[:, 0], condition[:, 1] = lookup["the"], lookup["small"]
            condition_mask = torch.zeros_like(condition, dtype=torch.bool)
            condition_mask[:, :2] = True
            samples = sample_masked(counted_predictor, tuple(condition.shape), mask_id=0, steps=args.sample_steps,
                                    device=str(device), condition_ids=condition, condition_mask=condition_mask)
        else:
            predictor = lambda x, t: counted_predictor(x, t.float() / discrete.steps)
            samples = discrete.sample(predictor, (args.samples, data.shape[1]), device=str(device))
    metadata = {"data_source": "synthetic fixed-length whitespace text teaching fixture",
                "training_sentences": len(data), "vocabulary": vocabulary,
                "mask_id": 0 if args.mode == "masked" else None,
                "objective": "unweighted masked-token CE" if args.mode == "masked" else "auxiliary clean-token CE",
                "prompt": "the small" if args.mode == "masked" else None,
                "network_evaluations_per_sample_batch": calls}
    return model, records, samples, metadata, vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gaussian", "score", "masked", "categorical"), default="gaussian")
    parser.add_argument("--sampler", choices=("ddpm", "ddim", "sde", "ode"))
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-npy", help="Preprocessed float array [N,...] for Gaussian/score modes")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/diffusion"))
    args = parser.parse_args()
    for name in ("train_steps", "diffusion_steps", "sample_steps", "samples", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.diffusion_steps < 2:
        parser.error("--diffusion-steps must be at least 2")
    if args.mode == "gaussian":
        args.sampler = args.sampler or "ddim"
        if args.sampler not in {"ddpm", "ddim"}:
            parser.error("Gaussian mode uses ddpm or ddim")
        if args.sampler == "ddim" and args.sample_steps > args.diffusion_steps:
            parser.error("DDIM sample steps must not exceed diffusion steps")
    elif args.mode == "score":
        args.sampler = args.sampler or "ode"
        if args.sampler not in {"sde", "ode"}:
            parser.error("Score mode uses sde or ode")
    elif args.sampler is not None:
        parser.error("Text samplers are selected by --mode; omit --sampler")
    if args.data_npy and args.mode not in {"gaussian", "score"}:
        parser.error("--data-npy is only for continuous modes")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    run = train_continuous if args.mode in {"gaussian", "score"} else train_text
    model, records, samples, metadata, vocabulary = run(args, device)
    if not torch.isfinite(samples).all():
        raise RuntimeError("Sampler produced non-finite values; inspect learned field and numerical solver")
    np.save(args.out_dir / "samples.npy", samples.detach().cpu().numpy())
    if vocabulary:
        lines = [" ".join(vocabulary[index] for index in row) for row in samples.cpu().tolist()]
        (args.out_dir / "samples.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (args.out_dir / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {"mode": args.mode, "sampler": args.sampler, "train_steps": args.train_steps,
              "diffusion_steps": args.diffusion_steps, "sample_steps": args.sample_steps,
              "seed": args.seed, "device": args.device, "batch_size": args.batch_size,
              "learning_rate": 0.002, "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "parameters": sum(p.numel() for p in model.parameters()),
              "sample_shape": list(samples.shape), "elapsed_seconds": round(time.monotonic() - started, 4),
              "first_batch_loss": records[0], "last_batch_loss": records[-1],
              "evidence_scope": "algorithm teaching run; no held-out quality metric or paper reproduction",
              **metadata}
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "report": report}, args.out_dir / "denoiser.pt")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
