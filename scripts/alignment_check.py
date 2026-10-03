"""Deterministic hand-checks of loss structure; no training or benchmark data."""
import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser(
        description="Check shifted masks, SFT/DPO, GRPO group validity and gradients on hand-chosen tensors. This is a structural math diagnostic, not a training experiment or benchmark."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Tensor device; no model or dataset is loaded")
    args = parser.parse_args()
    try:
        import torch
        from fm_tutorial.alignment import completion_log_probs, dpo_loss, group_advantages, grpo_loss, sft_loss
    except ImportError as error:
        parser.error(f"Install this package with its PyTorch dependency first: {error}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    device = torch.device(args.device)
    # Positions 2 and 3 are completion target tokens. Position 0 is never selected.
    logits = torch.tensor([[[.8, .2], [.25, .75], [.6, .4], [.9, .1]]], dtype=torch.float64, device=device).log().requires_grad_()
    ids = torch.tensor([[0, 1, 1, 0]], device=device)
    completion_mask = torch.tensor([[False, False, True, True]], device=device)
    summed = completion_log_probs(logits, ids, completion_mask)
    sft = sft_loss(logits, ids, completion_mask)
    expected_sum = math.log(.75) + math.log(.6)
    if not math.isclose(summed.item(), expected_sum, abs_tol=1e-10):
        raise RuntimeError("completion target shift/sum check failed")
    if not math.isclose(sft.item(), -expected_sum / 2, abs_tol=1e-10):
        raise RuntimeError("SFT active-token mean check failed")
    sft.backward()
    if logits.grad[0, 0].abs().sum().item() != 0 or logits.grad[0, 3].abs().sum().item() != 0:
        raise RuntimeError("prompt/unused-position gradient check failed")
    zeros = torch.zeros(2, dtype=torch.float64, device=device)
    dpo = dpo_loss(zeros, zeros, zeros, zeros)
    if not math.isclose(dpo.item(), math.log(2), abs_tol=1e-10):
        raise RuntimeError("DPO equal-margin check failed")
    rewards = torch.tensor([[1., 3.], [7., 7.]], dtype=torch.float64, device=device)
    advantages, valid = group_advantages(rewards)
    current = torch.tensor([[[0.], [math.log(1.1)]], [[10.], [10.]]], dtype=torch.float64, device=device, requires_grad=True)
    token_mask = torch.ones_like(current, dtype=torch.bool) & valid[:, None, None]
    grpo = grpo_loss(current, torch.zeros_like(current), advantages, token_mask)
    if not math.isclose(grpo.item(), -.05, abs_tol=1e-10):
        raise RuntimeError("GRPO valid-group mask/normalization check failed")
    grpo.backward()
    if current.grad[1].abs().sum().item() != 0:
        raise RuntimeError("invalid-group gradient check failed")
    print(json.dumps({
        "status": "math_checks_passed",
        "scope": "deterministic structural diagnostic; no training or benchmark data",
        "device": str(device),
        "completion_log_prob_sum": summed.item(),
        "sft_active_token_mean": sft.item(),
        "dpo_equal_margins": dpo.item(),
        "advantages": advantages.tolist(),
        "valid_groups": valid.tolist(),
        "grpo_active_token_mean": grpo.item(),
        "masked_gradients_zero": True,
        "kl_note": "k3 equals policy KL in expectation only with appropriate current-policy sampling and support; old-policy rollouts need correction for that claim",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
