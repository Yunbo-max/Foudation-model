#!/usr/bin/env python3
"""Generate from a tutorial checkpoint using its exact learned tokenizer."""
import argparse
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def generate_tokens(model, prompt_ids, max_new_tokens=64, temperature=0.8):
    """Return prompt plus generated ids; truncate each input window, never emit PAD."""
    import torch
    from fm_tutorial.model import EOS_ID, PAD_ID
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative (0 is greedy)")
    ids = list(prompt_ids)
    if not ids:
        raise ValueError("prompt must encode to at least one token")
    if any(not isinstance(i, int) or i < 0 or i >= model.config.vocab_size or i == PAD_ID for i in ids):
        raise ValueError("prompt contains invalid or PAD token ids")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-model.config.context_length:]], dtype=torch.long, device=device)
            logits = model(x)["logits"][0, -1].float().clone()
            logits[PAD_ID] = float("-inf")
            if temperature == 0:
                token = int(logits.argmax().item())
            else:
                token = int(torch.multinomial((logits / temperature).softmax(dim=-1), 1).item())
            ids.append(token)
            if token == EOS_ID:
                break
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import torch
        from fm_tutorial.model import ModelConfig, TransformerLM
        from fm_tutorial.tokenizer import ByteBPETokenizer
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        tokenizer = ByteBPETokenizer.load(args.tokenizer)
        tokenizer_hash = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        if tokenizer_hash != checkpoint["data_fingerprint"]["tokenizer_sha256"]:
            raise ValueError("tokenizer fingerprint differs from checkpoint")
        config = ModelConfig(**checkpoint["model_config"])
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("tokenizer vocabulary differs from checkpoint")
        model = TransformerLM(config).to(args.device)
        model.load_state_dict(checkpoint["model_state"])
        torch.manual_seed(args.seed)
        ids = generate_tokens(model, tokenizer.encode(args.prompt), args.max_new_tokens, args.temperature)
        print(tokenizer.decode(ids))
    except (ImportError, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"generation failed: {exc}\n")


if __name__ == "__main__":
    main()
