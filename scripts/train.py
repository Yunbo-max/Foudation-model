#!/usr/bin/env python3
"""Train on real prepared uint32 corpora; --steps is a cumulative update goal.

The config's max_steps fixes the LR schedule horizon, including across resume.
An input window has context_length tokens; internally shifted loss supervises
context_length-1 predictions. Validation always uses the same data slices.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_settings(raw):
    defaults = dict(batch_size=4, gradient_accumulation=1, learning_rate=3e-4,
                    weight_decay=0.1, warmup_steps=10, max_steps=200,
                    min_lr_ratio=0.1, eval_every=20, eval_batches=4,
                    gradient_clip=1.0, seed=42, mixed_precision=False)
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"unknown training settings: {sorted(unknown)}")
    settings = {**defaults, **raw}
    for name in ("batch_size", "gradient_accumulation", "max_steps", "eval_every", "eval_batches"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(settings["seed"], int) or not 0 <= settings["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (not isinstance(settings["warmup_steps"], int)
            or not 0 <= settings["warmup_steps"] < settings["max_steps"]):
        raise ValueError("warmup_steps must be in [0, max_steps)")
    for name in ("learning_rate", "weight_decay", "min_lr_ratio", "gradient_clip"):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if settings["learning_rate"] == 0 or settings["gradient_clip"] == 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    if settings["min_lr_ratio"] > 1:
        raise ValueError("min_lr_ratio must not exceed 1")
    if not isinstance(settings["mixed_precision"], bool):
        raise ValueError("mixed_precision must be true or false")
    return settings


def _learning_rate(step, settings):
    warmup = settings["warmup_steps"]
    if step <= warmup:
        return settings["learning_rate"] * step / max(1, warmup)
    progress = (step - warmup) / (settings["max_steps"] - warmup)
    ratio = settings["min_lr_ratio"] + (1 - settings["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
    return settings["learning_rate"] * ratio


def _open_tokens(path, context_length, vocab_size):
    import numpy as np
    path = Path(path)
    size = path.stat().st_size
    if size % 4:
        raise ValueError(f"{path}: byte length is not a multiple of uint32")
    if size // 4 < context_length:
        raise ValueError(f"{path}: need at least {context_length} tokens for one window")
    tokens = np.memmap(path, mode="r", dtype="<u4")
    for start in range(0, len(tokens), 1024 * 1024):
        chunk = tokens[start:start + 1024 * 1024]
        if (chunk >= vocab_size).any():
            raise ValueError(f"{path}: token outside effective tokenizer vocabulary")
        if (chunk == 257).any():
            raise ValueError(f"{path}: PAD must not occur in a packed training/validation corpus")
    return tokens


def _batch(tokens, starts, context_length, device):
    import numpy as np
    import torch
    # Copy avoids non-writable memmap warnings and gives torch int64 indices.
    values = np.stack([tokens[int(i):int(i) + context_length] for i in starts]).astype(np.int64)
    return torch.from_numpy(values).to(device)


def _capture_rng(device_type):
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device_type == "cuda" else [],
    }


def _restore_rng(state, device_type):
    import numpy as np
    import torch
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    if device_type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def train(config_path, data_dir, out_dir, steps=None, device="cpu", resume=None):
    """Train to a cumulative optimizer-update count and return final metadata.

    Resume requires the same effective hyperparameters, corpus/tokenizer hashes
    and device type. CPU reproducibility is regression-tested; CUDA execution
    still depends on PyTorch/kernel/hardware determinism. No data is fabricated.
    """
    import numpy as np
    import torch
    from fm_tutorial.model import ModelConfig, TransformerLM
    from fm_tutorial.tokenizer import ByteBPETokenizer

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if set(raw) - {"model", "training"}:
        raise ValueError("config must contain only model and training sections")
    settings = _training_settings(raw.get("training", {}))
    goal = settings["max_steps"] if steps is None else steps
    if isinstance(goal, bool) or not isinstance(goal, int) or not 1 <= goal <= settings["max_steps"]:
        raise ValueError("--steps must be a positive cumulative goal <= config training.max_steps")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    target_device = torch.device(device)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    # A template's nominal vocab never overrules the actual learned tokenizer.
    model_config = ModelConfig(**{**raw.get("model", {}), "vocab_size": tokenizer.vocab_size})
    effective_config = {"model": asdict(model_config), "training": settings}
    train_tokens = _open_tokens(data_dir / "train.bin", model_config.context_length, tokenizer.vocab_size)
    val_tokens = _open_tokens(data_dir / "val.bin", model_config.context_length, tokenizer.vocab_size)
    fingerprint = {
        "tokenizer_sha256": _hash_file(tokenizer_path),
        "train_sha256": _hash_file(data_dir / "train.bin"),
        "val_sha256": _hash_file(data_dir / "val.bin"),
        "train_tokens": len(train_tokens), "val_tokens": len(val_tokens),
        "token_dtype": "little-endian uint32",
    }
    saved = None
    if resume is not None:
        saved = torch.load(Path(resume), map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if saved["effective_config"] != effective_config:
            raise ValueError("resume configuration differs from checkpoint hyperparameters")
        if saved["data_fingerprint"] != fingerprint:
            raise ValueError("resume data/tokenizer fingerprint differs from checkpoint")
        if saved["device_type"] != device:
            raise ValueError("resume device type differs; exact RNG continuation requires the same type")
        if saved["step"] > goal:
            raise ValueError("--steps is cumulative and precedes the checkpoint step")
    elif (out_dir / "last.pt").exists():
        raise ValueError("output already has last.pt; use --resume or a new --out-dir")

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
    model = TransformerLM(model_config).to(target_device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": settings["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0}], lr=settings["learning_rate"])
    amp_enabled = device == "cuda" and settings["mixed_precision"]
    amp_dtype = torch.bfloat16 if not amp_enabled or torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    start_step, tokens_seen = 0, 0
    if saved is not None:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        start_step, tokens_seen = saved["step"], saved["tokens_seen"]
        _restore_rng(saved["rng"], device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"event": "start", "parameters": parameter_count,
                      "effective_vocab_size": tokenizer.vocab_size, "device": device,
                      "start_step": start_step, "target_step": goal}, ensure_ascii=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "effective_config.json").write_text(json.dumps(effective_config, indent=2) + "\n", encoding="utf-8")
    batch_size = settings["batch_size"]
    window = model_config.context_length
    # Fixed positions, independent of training RNG; small validation files may
    # reuse/overlap slices. Report this finite-slice loss, not a corpus benchmark.
    validation_starts = np.linspace(0, len(val_tokens) - window,
                                   batch_size * settings["eval_batches"], dtype=np.int64)

    def autocast_context():
        return torch.autocast(device_type=device, dtype=amp_dtype) if amp_enabled else nullcontext()

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for offset in range(0, len(validation_starts), batch_size):
                x = _batch(val_tokens, validation_starts[offset:offset + batch_size], window, target_device)
                with autocast_context():
                    losses.append(model(x, labels=x)["loss"].item())
        model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(step):
        checkpoint = {
            "format_version": 1, "step": step, "tokens_seen": tokens_seen,
            "model_config": asdict(model_config), "effective_config": effective_config,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(), "rng": _capture_rng(device),
            "data_fingerprint": fingerprint, "device_type": device,
            "torch_version": str(torch.__version__), "parameter_count": parameter_count,
        }
        temporary = out_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(out_dir / "last.pt")

    model.train()
    for step in range(start_step + 1, goal + 1):
        lr = _learning_rate(step, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for _ in range(settings["gradient_accumulation"]):
            starts = np.random.randint(0, len(train_tokens) - window + 1, size=batch_size)
            x = _batch(train_tokens, starts, window, target_device)
            with autocast_context():
                loss = model(x, labels=x)["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite loss at update {step}")
            mean_loss += loss.item() / settings["gradient_accumulation"]
            scaler.scale(loss / settings["gradient_accumulation"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
        if not torch.isfinite(gradient_norm).item():
            raise FloatingPointError(f"non-finite gradients at update {step}")
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += batch_size * (window - 1) * settings["gradient_accumulation"]
        record = {"step": step, "train_loss": mean_loss, "learning_rate": lr, "tokens_seen": tokens_seen}
        if step % settings["eval_every"] == 0 or step == goal:
            record["val_loss"] = evaluate()
            record["validation_targets"] = len(validation_starts) * (window - 1)
            save_checkpoint(step)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False))
    if start_step == goal:
        # A no-op resume can target a new output directory; keep the returned
        # artifact contract true without performing another optimizer update.
        save_checkpoint(start_step)
    return {"step": goal, "tokens_seen": tokens_seen, "parameters": parameter_count,
            "effective_vocab_size": tokenizer.vocab_size, "checkpoint": str(out_dir / "last.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cpu.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cpu"))
    parser.add_argument("--steps", type=int, help="cumulative optimizer-update goal; config max_steps fixes the LR horizon")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume with matching config/data/device")
    args = parser.parse_args()
    try:
        train(args.config, args.data_dir, args.out_dir, args.steps, args.device, args.resume)
    except (ImportError, ValueError, OSError, FloatingPointError) as exc:
        parser.exit(1, f"training failed: {exc}\n")


if __name__ == "__main__":
    main()
