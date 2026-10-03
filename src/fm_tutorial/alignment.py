"""Original, small alignment-loss implementations for mathematical study.

These functions operate on supplied tensors. They do not collect preference
data, run an RL environment, or reproduce a published training experiment.
Probability, margin and reduction math promotes fp16/bf16 inputs to fp32;
float64 inputs retain their precision. Casts remain connected to policy gradients.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _finite_float(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")


def _loss_math(value: Tensor) -> Tensor:
    """Use at least fp32 for low-precision probability/reduction arithmetic."""
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _binary_mask(mask: Tensor, shape: torch.Size, device: torch.device) -> Tensor:
    if not isinstance(mask, Tensor) or mask.shape != shape or mask.device != device:
        raise ValueError("mask must have the requested shape and device")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask must contain only zero/one or boolean values")
    return mask.bool()


def _completion_tokens(logits: Tensor, input_ids: Tensor, completion_mask: Tensor) -> tuple[Tensor, Tensor]:
    _finite_float("logits", logits)
    if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1] < 2 or logits.shape[2] == 0:
        raise ValueError("logits must have shape [B,T,V] with B,V>0 and T>=2")
    if not isinstance(input_ids, Tensor) or input_ids.shape != logits.shape[:2]:
        raise ValueError("input_ids must have shape [B,T]")
    if input_ids.dtype not in (torch.int32, torch.int64) or input_ids.device != logits.device:
        raise ValueError("input_ids must be integer token IDs on the logits device")
    if bool(((input_ids < 0) | (input_ids >= logits.shape[-1])).any()):
        raise ValueError("input_ids contain an out-of-vocabulary token")
    mask = _binary_mask(completion_mask, input_ids.shape, logits.device)
    if bool(mask[:, 0].any()):
        raise ValueError("completion_mask[:,0] must be false: the first token has no preceding prediction")
    log_probs = F.log_softmax(_loss_math(logits[:, :-1]), dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids[:, 1:].long().unsqueeze(-1)).squeeze(-1)
    return token_log_probs, mask[:, 1:]


def completion_log_probs(logits: Tensor, input_ids: Tensor, completion_mask: Tensor) -> Tensor:
    """Sum completion log-probabilities per sequence, returning shape [B].

    Mask positions identify *target tokens*: logits[:,t-1] predict input_ids[:,t].
    Therefore completion_mask[:,0] must be false. Unselected prompt/pad targets
    contribute zero. A sequence with no selected completion targets sums to zero.
    DPO below consumes these sequence sums, rather than a length-normalized mean.
    fp16/bf16 logits produce fp32 scores to avoid long-sequence sum overflow;
    float64 logits produce float64 scores.
    """
    token_log_probs, mask = _completion_tokens(logits, input_ids, completion_mask)
    return token_log_probs.masked_fill(~mask, 0).sum(dim=-1)


def sft_loss(logits: Tensor, input_ids: Tensor, completion_mask: Tensor) -> Tensor:
    """Mean negative log-probability over all selected completion target tokens.

    This active-token mean weights longer completions by their token count.
    A globally empty completion selection raises ValueError, rather than silently
    pretending that an SFT minibatch with no targets supplies a learning signal.
    """
    token_log_probs, mask = _completion_tokens(logits, input_ids, completion_mask)
    if not bool(mask.any()):
        raise ValueError("SFT requires at least one selected completion target")
    return -token_log_probs[mask].mean()


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    ref_chosen: Tensor,
    ref_rejected: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Mean stable DPO loss from paired sequence-sum log-probabilities.

    The reference scores are detached. Increasing the policy chosen-vs-rejected
    margin lowers this loss; reference parameters receive no gradient.
    Half-precision input scores are promoted before margin subtraction.
    """
    values = (policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    for name, value in zip(("policy_chosen", "policy_rejected", "ref_chosen", "ref_rejected"), values):
        _finite_float(name, value)
    if policy_chosen.ndim != 1 or policy_chosen.numel() == 0:
        raise ValueError("DPO inputs must be nonempty per-sequence vectors")
    if any(value.shape != policy_chosen.shape or value.device != policy_chosen.device for value in values):
        raise ValueError("DPO pairs must share a shape and device")
    if isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = (_loss_math(policy_chosen) - _loss_math(policy_rejected)) - (_loss_math(ref_chosen.detach()) - _loss_math(ref_rejected.detach()))
    return -F.logsigmoid(beta * margin).mean()


def group_advantages(rewards: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Normalize rewards within each prompt group, returning (A, valid_groups).

    rewards has shape [G,K]. Population std (unbiased=False) is used because all
    K sampled rewards constitute this rollout group. A constant or K=1 group
    supplies no relative ranking: its advantage is zero and validity is false.
    Rewards are detached; rewards themselves are not policy advantages until
    this within-group centering/scaling step. Combine valid_groups[:,None,None]
    with the completion mask before passing grouped [G,K,T] tensors to grpo_loss.
    """
    _finite_float("rewards", rewards)
    if rewards.ndim != 2 or 0 in rewards.shape:
        raise ValueError("rewards must have nonempty shape [G,K]")
    if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    rewards = _loss_math(rewards.detach())
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    valid = std.squeeze(-1) > 0
    advantages = centered / std.clamp_min(eps)
    return advantages.masked_fill(~valid[:, None], 0), valid


def grpo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_epsilon: float = 0.2,
    ref_log_probs: Tensor | None = None,
    kl_beta: float = 0.0,
) -> Tensor:
    """Masked active-token mean of a group-relative PPO clipped surrogate.

    Token scores may be [N,T] or [G,K,T]. Advantages have the leading sequence
    shape or the full token shape. Old scores, reference scores and advantages
    are detached. The mask must already exclude invalid constant-reward groups;
    otherwise those groups dilute the mean. This active-token aggregation is an
    explicit teaching choice, not a full reproduction of the original GRPO
    completion-normalized/group-averaged implementation.

    Optional k3 penalty: exp(ref_logp-logp) - (ref_logp-logp) - 1. Its expectation
    equals KL(policy||reference) only under current-policy sampling and suitable
    support. On old-policy PPO rollouts, without importance correction, it is a
    sampled surrogate, not an unbiased current-policy KL estimate.

    An all-empty mask returns gradient-connected zero. Active exponential
    overflow raises ValueError; silently clipping log-ratios would change PPO.
    """
    _finite_float("log_probs", log_probs)
    _finite_float("old_log_probs", old_log_probs)
    _finite_float("advantages", advantages)
    if log_probs.ndim not in (2, 3) or 0 in log_probs.shape:
        raise ValueError("log_probs must have nonempty shape [N,T] or [G,K,T]")
    if old_log_probs.shape != log_probs.shape or old_log_probs.device != log_probs.device:
        raise ValueError("old_log_probs must share the policy score shape and device")
    if advantages.device != log_probs.device:
        raise ValueError("advantages must be on the policy score device")
    if advantages.shape == log_probs.shape[:-1]:
        advantages = advantages.unsqueeze(-1).expand_as(log_probs)
    elif advantages.shape != log_probs.shape:
        raise ValueError("advantages must have sequence shape or full token shape")
    selected = _binary_mask(mask, log_probs.shape, log_probs.device)
    if isinstance(clip_epsilon, bool) or not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0,1)")
    if isinstance(kl_beta, bool) or not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta must be finite and nonnegative")
    if ref_log_probs is not None:
        _finite_float("ref_log_probs", ref_log_probs)
        if ref_log_probs.shape != log_probs.shape or ref_log_probs.device != log_probs.device:
            raise ValueError("reference scores must share the policy score shape and device")
    elif kl_beta > 0:
        raise ValueError("positive kl_beta requires ref_log_probs")
    # Select before exponentiation so masked scores cannot overflow or affect loss.
    current = _loss_math(log_probs[selected])
    if current.numel() == 0:
        return current.sum()
    ratio = (current - _loss_math(old_log_probs.detach()[selected])).exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("active policy ratios overflowed; inspect log-probabilities and update size")
    advantage = _loss_math(advantages.detach()[selected])
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    losses = -surrogate
    if kl_beta > 0:
        delta = _loss_math(ref_log_probs.detach()[selected]) - current
        losses = losses + kl_beta * (delta.expm1() - delta)
    if not bool(torch.isfinite(losses).all()):
        raise ValueError("active PPO/KL terms overflowed; inspect log-probabilities and policy update size")
    return losses.mean()
