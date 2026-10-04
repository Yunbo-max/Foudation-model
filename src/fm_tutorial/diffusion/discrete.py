"""Small, original categorical and absorbing-mask diffusion teaching examples.

Categorical matrices use row vectors: Q[t][i, j] = q(x_t=j | x_{t-1}=i).
Their mathematical time is 1..T, with Q[0] = Q_bar[0] = I for clean data.
The dense matrices and per-clean posterior mixtures are deliberately a
small-vocabulary reference, not a scalable language-model implementation.

The absorbing-mask path is separate: its continuous time in [0, 1] is the
probability that a non-condition token is masked. Its loss is unweighted CE
over the realized masked tokens, not MDLM's time-weighted continuous ELBO.

Sources: https://arxiv.org/abs/2107.03006 (D3PM),
https://arxiv.org/abs/2406.07524 (MDLM).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must be (batch, length)")
    for value in shape:
        _positive_integer(value, "shape dimension")
    return shape[0], shape[1]


def _ids(ids: Tensor, vocab_size: int | None = None, name: str = "ids") -> None:
    if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
        raise ValueError(f"{name} must be a nonempty long tensor [B, L]")
    if (ids < 0).any() or (vocab_size is not None and (ids >= vocab_size).any()):
        raise ValueError(f"{name} contains an invalid token id")


def _condition_mask(mask: Tensor | None, ids: Tensor) -> Tensor:
    if mask is None:
        return torch.zeros_like(ids, dtype=torch.bool)
    if mask.shape != ids.shape or mask.dtype != torch.bool:
        raise ValueError("condition_mask must be bool [B, L] matching tokens")
    return mask.to(ids.device)


def _discrete_times(t: int | Tensor, batch: int, steps: int,
                    device: torch.device, minimum: int = 0) -> Tensor:
    times = torch.as_tensor(t, device=device)
    if times.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("categorical times must be integers")
    if times.ndim == 0:
        times = times.expand(batch)
    if times.shape != (batch,) or (times < minimum).any() or (times > steps).any():
        raise ValueError(f"categorical times must be scalar or [B], in {minimum}..{steps}")
    return times.long()


def _mask_times(t: Tensor, batch: int, device: torch.device) -> Tensor:
    times = torch.as_tensor(t, device=device)
    if times.shape != (batch,) or not times.is_floating_point():
        raise ValueError("mask times must be floating-point [B]")
    times = times.float()
    if not torch.isfinite(times).all() or (times < 0).any() or (times > 1).any():
        raise ValueError("mask times must be finite probabilities in [0, 1]")
    return times


def _logits(model: Callable[[Tensor, Tensor], Tensor], ids: Tensor, t: Tensor,
            vocab_size: int | None = None) -> Tensor:
    logits = model(ids, t)
    if not isinstance(logits, Tensor) or logits.ndim != 3 or logits.shape[:2] != ids.shape:
        raise ValueError("model must return logits [B, L, V]")
    if not logits.is_floating_point() or logits.shape[-1] < 2:
        raise ValueError("model logits must be floating-point with at least two classes")
    if vocab_size is not None and logits.shape[-1] != vocab_size:
        raise ValueError("model vocabulary does not match the categorical process")
    logits = logits.float()
    if logits.device != ids.device or not torch.isfinite(logits).all():
        raise ValueError("model logits must be finite and on the tokens' device")
    return logits


class CategoricalDiffusion(nn.Module):
    """Uniform row-stochastic transitions Q_t = (1-beta_t) I + beta_t / V.

    ``q_probs``/``q_sample`` additionally accept t=0 for the clean identity.
    ``posterior_probs``/``reverse_probs`` require t=1..T. Positive betas give
    every observed class positive evidence for every clean class, avoiding
    undefined posteriors. Dense storage costs O(T V^2); reverse mixtures
    additionally materialize O(B L V^2) values.
    """

    def __init__(self, vocab_size: int, steps: int = 20,
                 beta_start: float = 0.01, beta_end: float = 0.2):
        super().__init__()
        _positive_integer(vocab_size, "vocab_size")
        _positive_integer(steps, "steps")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        if not all(math.isfinite(beta) and 0 < beta <= 1 for beta in (beta_start, beta_end)):
            raise ValueError("betas must be finite and in (0, 1]")
        self.vocab_size, self.steps = vocab_size, steps
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        if not (betas / vocab_size > 0).all():
            raise ValueError("betas are too small to retain full support in float32")
        identity = torch.eye(vocab_size, dtype=torch.float32)
        uniform = torch.full_like(identity, 1.0 / vocab_size)
        transitions = [identity]
        cumulative = [identity]
        for beta in betas:
            transition = (1 - beta) * identity + beta * uniform
            transitions.append(transition)
            cumulative.append(cumulative[-1] @ transition)
        self.register_buffer("betas", torch.cat((torch.zeros(1), betas)))
        self.register_buffer("Q", torch.stack(transitions))
        self.register_buffer("Q_bar", torch.stack(cumulative))

    def q_probs(self, x0: Tensor, t: int | Tensor) -> Tensor:
        """Return q(x_t | x0), shape [B, L, V], including the t=0 identity."""
        _ids(x0, self.vocab_size, "x0")
        times = _discrete_times(t, x0.shape[0], self.steps, x0.device)
        matrices = self.Q_bar.to(x0.device)[times]
        batch = torch.arange(x0.shape[0], device=x0.device)[:, None]
        return matrices[batch, x0]

    def q_sample(self, x0: Tensor, t: int | Tensor) -> Tensor:
        """Independently sample each position from the exact forward marginal."""
        probabilities = self.q_probs(x0, t)
        return torch.multinomial(probabilities.reshape(-1, self.vocab_size), 1).reshape_as(x0)

    def posterior_probs(self, x0: Tensor, xt: Tensor, t: int | Tensor) -> Tensor:
        """Exact q(x_{t-1}=k | x_t=j, x0=i) proportional to Q_bar[i,k] Q[k,j]."""
        _ids(x0, self.vocab_size, "x0")
        _ids(xt, self.vocab_size, "xt")
        if xt.shape != x0.shape or xt.device != x0.device:
            raise ValueError("x0 and xt must have matching shapes and devices")
        times = _discrete_times(t, x0.shape[0], self.steps, x0.device, minimum=1)
        batch = torch.arange(x0.shape[0], device=x0.device)[:, None]
        prior = self.Q_bar.to(x0.device)[times - 1][batch, x0]
        likelihood = self.Q.to(x0.device)[times].transpose(1, 2)[batch, xt]
        joint = prior * likelihood
        return joint / joint.sum(-1, keepdim=True)

    def reverse_probs(self, x0_probs: Tensor, xt: Tensor, t: int | Tensor) -> Tensor:
        """Marginalize *normalized* clean-conditioned posteriors using p(x0|xt).

        This educational parameterization is sum_i p_theta(i|xt) q(prev|xt,i).
        It differs from original D3PM Eq. (4), which normalizes a mixture of
        joint terms under a differently parameterized clean predictor. Mixing
        our joint terms before normalizing would incorrectly reweight p(x0|xt)
        by each clean class's evidence q(xt|x0).
        """
        _ids(xt, self.vocab_size, "xt")
        if x0_probs.shape != (*xt.shape, self.vocab_size) or not x0_probs.is_floating_point():
            raise ValueError("x0_probs must be floating-point [B, L, V]")
        probabilities = x0_probs.float()
        if probabilities.device != xt.device or not torch.isfinite(probabilities).all():
            raise ValueError("x0_probs must be finite and on xt's device")
        if (probabilities < 0).any() or not torch.allclose(
                probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=1e-5):
            raise ValueError("x0_probs must be normalized probabilities")
        times = _discrete_times(t, xt.shape[0], self.steps, xt.device, minimum=1)
        batch = torch.arange(xt.shape[0], device=xt.device)[:, None]
        # Axes below: [batch, position, candidate clean class, previous class].
        prior = self.Q_bar.to(xt.device)[times - 1][:, None, :, :]
        likelihood = self.Q.to(xt.device)[times].transpose(1, 2)[batch, xt]
        joint = prior * likelihood.unsqueeze(-2)
        conditional = joint / joint.sum(-1, keepdim=True)
        return (probabilities.unsqueeze(-1) * conditional).sum(-2)

    @torch.no_grad()
    def sample(self, model: Callable[[Tensor, Tensor], Tensor],
               shape: Sequence[int], device: str | torch.device = "cpu") -> Tensor:
        """Reverse T..1 from the uniform prior; model receives long time [B].

        Uniform is this kernel's limiting stationary prior. At finite T,
        Q_bar[T] need not equal it, so finite-prior mismatch remains unless
        the schedule mixes completely. The caller chooses the model's mode.
        """
        batch, length = _shape(shape)
        xt = torch.randint(self.vocab_size, (batch, length), device=device)
        for step in range(self.steps, 0, -1):
            times = torch.full((batch,), step, dtype=torch.long, device=xt.device)
            clean_probs = _logits(model, xt, times, self.vocab_size).softmax(-1)
            previous_probs = self.reverse_probs(clean_probs, xt, times)
            xt = torch.multinomial(previous_probs.reshape(-1, self.vocab_size), 1).reshape_as(xt)
        return xt


def mask_tokens(x0: Tensor, t: Tensor, mask_id: int,
                condition_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Mask eligible tokens independently with probability t[B] in [0, 1].

    Clean tokens must exclude the absorbing mask symbol. Condition positions
    are immutable and are excluded from the returned masked-position mask.
    """
    _ids(x0, name="x0")
    if isinstance(mask_id, bool) or not isinstance(mask_id, int) or mask_id < 0:
        raise ValueError("mask_id must be a nonnegative integer")
    if (x0 == mask_id).any():
        raise ValueError("clean x0 must not contain the mask symbol")
    times = _mask_times(t, x0.shape[0], x0.device)
    conditions = _condition_mask(condition_mask, x0)
    masked = (torch.rand(x0.shape, device=x0.device) < times[:, None]) & ~conditions
    return x0.masked_fill(masked, mask_id), masked


def masked_loss(logits: Tensor, x0: Tensor, masked_positions: Tensor) -> Tensor:
    """Unweighted float32 CE averaged over realized masked positions only.

    This teaching loss is not the MDLM continuous ELBO (which includes a
    schedule/time weight and a different token normalization). Empty batches
    return a graph-connected zero without summing potentially large logits.
    """
    _ids(x0, name="x0")
    if logits.ndim != 3 or logits.shape[:2] != x0.shape or not logits.is_floating_point():
        raise ValueError("logits must be floating-point [B, L, V] matching x0")
    if logits.device != x0.device or logits.shape[-1] < 2 or (x0 >= logits.shape[-1]).any():
        raise ValueError("logits and x0 must share a device and valid vocabulary")
    if masked_positions.shape != x0.shape or masked_positions.dtype != torch.bool:
        raise ValueError("masked_positions must be bool [B, L] matching x0")
    selected = masked_positions.to(x0.device)
    masked_logits = logits.float()[selected]
    if not selected.any():
        return masked_logits.sum()
    return F.cross_entropy(masked_logits, x0[selected])


@torch.no_grad()
def sample_masked(model: Callable[[Tensor, Tensor], Tensor], shape: Sequence[int],
                  mask_id: int, steps: int = 20, device: str | torch.device = "cpu",
                  condition_ids: Tensor | None = None,
                  condition_mask: Tensor | None = None) -> Tensor:
    """Sample a linear absorbing process, revealing masks with (t-s)/t.

    Start with all generated positions masked. Previously visible tokens are
    never re-masked or revised, and s=0 forces the final clean endpoint. Only
    entries selected by condition_mask are read from condition_ids. Model
    output determines V; mask_id may be anywhere in this vocabulary and is
    excluded before drawing clean tokens. Model time is float32 [B] in (0,1].
    """
    batch, length = _shape(shape)
    _positive_integer(steps, "steps")
    if isinstance(mask_id, bool) or not isinstance(mask_id, int) or mask_id < 0:
        raise ValueError("mask_id must be a nonnegative integer")
    xt = torch.full((batch, length), mask_id, dtype=torch.long, device=device)
    if (condition_ids is None) != (condition_mask is None):
        raise ValueError("condition_ids and condition_mask must be provided together")
    conditions = _condition_mask(condition_mask, xt)
    if condition_ids is not None:
        if condition_ids.shape != xt.shape or condition_ids.dtype != torch.long:
            raise ValueError("condition_ids must be long [B, L] matching shape")
        prompt = condition_ids.to(xt.device)[conditions]
        if (prompt < 0).any() or (prompt == mask_id).any():
            raise ValueError("condition tokens must be clean nonnegative ids")
        xt[conditions] = prompt
    for step in range(steps, 0, -1):
        masked = (xt == mask_id) & ~conditions
        if not masked.any():
            break
        # Linear mask marginal m(t)=t. Its conditional retention is s/t.
        t, s = step / steps, (step - 1) / steps
        times = torch.full((batch,), t, dtype=torch.float32, device=xt.device)
        logits = _logits(model, xt, times)
        if mask_id >= logits.shape[-1] or (xt >= logits.shape[-1]).any():
            raise ValueError("mask_id and visible tokens must belong to model vocabulary")
        reveal = masked if step == 1 else masked & (torch.rand(xt.shape, device=xt.device) < (t - s) / t)
        if reveal.any():
            clean_logits = logits[reveal].clone()
            clean_logits[:, mask_id] = -torch.inf
            xt[reveal] = torch.multinomial(clean_logits.softmax(-1), 1).squeeze(-1)
    return xt


class MaskedDenoiser(nn.Module):
    """Tiny bidirectional Transformer encoder with position and time inputs.

    No causal mask is supplied: each masked position can use visible tokens
    on either side. Float mask probabilities or integer categorical times
    are accepted; callers consistently choose their process's time units.
    """

    def __init__(self, vocab_size: int, max_length: int, dim: int = 32,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        for value, name in ((vocab_size, "vocab_size"), (max_length, "max_length"),
                            (dim, "dim"), (heads, "heads"), (layers, "layers")):
            _positive_integer(value, name)
        if vocab_size < 2 or dim % heads:
            raise ValueError("vocab_size must be >=2 and dim divisible by heads")
        self.vocab_size, self.max_length = vocab_size, max_length
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_length, dim)
        self.time_embedding = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4 * dim,
                                           dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim),
                                            enable_nested_tensor=False)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, ids: Tensor, t: Tensor) -> Tensor:
        _ids(ids, self.vocab_size)
        if ids.shape[1] > self.max_length:
            raise ValueError("sequence exceeds max_length")
        times = torch.as_tensor(t, device=ids.device).float()
        if times.shape != (ids.shape[0],) or not torch.isfinite(times).all() or (times < 0).any():
            raise ValueError("model times must be finite nonnegative [B]")
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = (self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
                  + self.time_embedding(times[:, None])[:, None, :])
        return self.output(self.encoder(hidden))
