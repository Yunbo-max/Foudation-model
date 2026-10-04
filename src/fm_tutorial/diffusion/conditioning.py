"""Small, inspectable conditioning mechanisms for diffusion feature tensors.

Token features use [B, N, D]; global conditions use [B, K], and condition
sequences use [B, M, K]. Conditions may already combine timestep, class, or
text embeddings. These blocks do not encode those inputs, train a denoiser,
or implement a complete DiT or ControlNet. Move modules and inputs to the
same device/dtype, and supply positional/modality embeddings when needed.

Classifier-free guidance combines predictions at sampling time. Its usual
Gaussian-diffusion interpretation applies to epsilon or score predictions;
mixing categorical logits does not give that same interpretation.
"""
import math
from numbers import Real

import torch
from torch import nn


__all__ = [
    "FeatureConcat", "AddCondition", "FiLM", "AdaLN", "AdaLNZeroBlock",
    "CrossAttentionCondition", "PrefixCondition", "ZeroSpatialResidual",
    "classifier_free_guidance",
]


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _dimensions(d_model, cond_dim, heads=None):
    _positive_integer("d_model", d_model)
    _positive_integer("cond_dim", cond_dim)
    if heads is not None:
        _positive_integer("heads", heads)
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")


def _floating_shape(tensor, name, rank, width):
    if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
            or tensor.ndim != rank or tensor.shape[-1] != width
            or any(size <= 0 for size in tensor.shape)):
        raise ValueError(f"{name} must be a nonempty floating rank-{rank} tensor with final dimension {width}")


def _same_device_dtype(left, right):
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("feature and condition tensors must have matching device and dtype")


def _check_tokens(h, c, d_model, cond_dim, condition_rank):
    _floating_shape(h, "h", 3, d_model)
    _floating_shape(c, "c", condition_rank, cond_dim)
    if c.shape[0] != h.shape[0]:
        raise ValueError("feature and condition batches must match")
    _same_device_dtype(h, c)


def _modulate(h, scale, shift):
    return (1 + scale[:, None, :]) * h + shift[:, None, :]


class FeatureConcat(nn.Module):
    """Broadcast c to all tokens, concatenate features, then project to D.

    A single linear projection of [h; c] equals W_h h + W_c c + b.
    A shared activation after these equal affine outputs preserves equality.
    Distinct nonlinear branches or different parameter constraints can differ.
    """

    def __init__(self, d_model, cond_dim):
        super().__init__()
        _dimensions(d_model, cond_dim)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.proj = nn.Linear(d_model + cond_dim, d_model)

    def forward(self, h, c):
        _check_tokens(h, c, self.d_model, self.cond_dim, 2)
        repeated = c[:, None, :].expand(-1, h.shape[1], -1)
        return self.proj(torch.cat((h, repeated), dim=-1))


class AddCondition(nn.Module):
    """Add a projected global condition to every token: h + W_c c + b."""

    def __init__(self, d_model, cond_dim):
        super().__init__()
        _dimensions(d_model, cond_dim)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.proj = nn.Linear(cond_dim, d_model)

    def forward(self, h, c):
        _check_tokens(h, c, self.d_model, self.cond_dim, 2)
        return h + self.proj(c)[:, None, :]


class FiLM(nn.Module):
    """Feature-wise affine modulation: (1 + scale(c)) * h + shift(c).

    This convention learns a scale offset around one. The linear head uses
    ordinary initialization, so a fresh module already depends on c.
    """

    def __init__(self, d_model, cond_dim):
        super().__init__()
        _dimensions(d_model, cond_dim)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.modulation = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, h, c):
        _check_tokens(h, c, self.d_model, self.cond_dim, 2)
        scale, shift = self.modulation(c).chunk(2, dim=-1)
        return _modulate(h, scale, shift)


class AdaLN(nn.Module):
    """Apply FiLM to LayerNorm(h), normalizing the D features of each token.

    LayerNorm has no learned affine parameters; the condition supplies them.
    Unlike FiLM, this approximately removes each token's input offset and
    positive scale (the normalization epsilon prevents exact invariance).
    """

    def __init__(self, d_model, cond_dim):
        super().__init__()
        _dimensions(d_model, cond_dim)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, h, c):
        _check_tokens(h, c, self.d_model, self.cond_dim, 2)
        scale, shift = self.modulation(c).chunk(2, dim=-1)
        return _modulate(self.norm(h), scale, shift)


class AdaLNZeroBlock(nn.Module):
    """A DiT-style attention/MLP block with six-way zero modulation.

    The head emits shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp,
    gate_mlp. Both residual gates start at zero, making the block initially
    the identity. Gates are unconstrained real values. At the first backward
    pass, gates can learn while the attention/MLP receive zero gradients.
    This implements the block pattern, not DiT's patch or output machinery.
    """

    def __init__(self, d_model, cond_dim, heads=4):
        super().__init__()
        _dimensions(d_model, cond_dim, heads)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.norm_attn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, h, c):
        _check_tokens(h, c, self.d_model, self.cond_dim, 2)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(c).chunk(6, dim=-1)
        )
        tokens = _modulate(self.norm_attn(h), scale_attn, shift_attn)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        h = h + gate_attn[:, None, :] * attended
        tokens = _modulate(self.norm_mlp(h), scale_mlp, shift_mlp)
        return h + gate_mlp[:, None, :] * self.mlp(tokens)


class CrossAttentionCondition(nn.Module):
    """Residual attention with Q from data and K/V from condition tokens.

    padding_mask is boolean [B, M], with True meaning ignored condition
    padding. Every batch row must retain at least one condition token.
    No position encoding is added here.
    """

    def __init__(self, d_model, cond_dim, heads=4):
        super().__init__()
        _dimensions(d_model, cond_dim, heads)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.attn = nn.MultiheadAttention(
            d_model, heads, kdim=cond_dim, vdim=cond_dim, batch_first=True,
        )

    def forward(self, h, c_tokens, padding_mask=None):
        _check_tokens(h, c_tokens, self.d_model, self.cond_dim, 3)
        if padding_mask is not None:
            if (not isinstance(padding_mask, torch.Tensor)
                    or padding_mask.dtype != torch.bool
                    or padding_mask.shape != c_tokens.shape[:2]
                    or padding_mask.device != h.device):
                raise ValueError("padding_mask must be boolean [B, M] on the feature device")
            if padding_mask.all(dim=1).any():
                raise ValueError("each row must have at least one unmasked condition token")
        attended, _ = self.attn(
            h, c_tokens, c_tokens, key_padding_mask=padding_mask, need_weights=False,
        )
        return h + attended


class PrefixCondition(nn.Module):
    """Prepend projected condition tokens, self-attend, return data tokens.

    Attention is bidirectional over both condition and data tokens, with a
    residual over the complete sequence. This is not a causal LM prefix.
    Conditions must be unpadded. The caller supplies any required position
    or modality embeddings; this block adds neither.
    """

    def __init__(self, d_model, cond_dim, heads=4):
        super().__init__()
        _dimensions(d_model, cond_dim, heads)
        self.d_model, self.cond_dim = d_model, cond_dim
        self.proj = nn.Linear(cond_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)

    def forward(self, h, c_tokens):
        _check_tokens(h, c_tokens, self.d_model, self.cond_dim, 3)
        prefix_length = c_tokens.shape[1]
        tokens = torch.cat((self.proj(c_tokens), h), dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return (tokens + attended)[:, prefix_length:, :]


class ZeroSpatialResidual(nn.Module):
    """Add a hint branch through a zero-initialized 1x1 output convolution.

    h is [B, C, H, W], hint is [B, K, H, W]. Initially the output equals h;
    the zero output map learns before gradients reach the hint encoder.
    This illustrates a ControlNet-like zero-convolution connection, without
    a copied/frozen denoiser or multiple-scale control branches.
    """

    def __init__(self, channels, hint_channels):
        super().__init__()
        _positive_integer("channels", channels)
        _positive_integer("hint_channels", hint_channels)
        self.channels, self.hint_channels = channels, hint_channels
        self.hint_encoder = nn.Sequential(
            nn.Conv2d(hint_channels, channels, kernel_size=3, padding=1), nn.SiLU(),
        )
        self.zero_conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, h, hint):
        for tensor, name, channels in ((h, "h", self.channels),
                                       (hint, "hint", self.hint_channels)):
            if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                    or tensor.ndim != 4 or tensor.shape[1] != channels
                    or any(size <= 0 for size in tensor.shape)):
                raise ValueError(f"{name} must be nonempty floating [B, {channels}, H, W]")
        if h.shape[0] != hint.shape[0] or h.shape[2:] != hint.shape[2:]:
            raise ValueError("features and hints must have matching batch and spatial dimensions")
        _same_device_dtype(h, hint)
        return h + self.zero_conv(self.hint_encoder(hint))


def classifier_free_guidance(uncond, cond, scale=1.0):
    """Return uncond + scale * (cond - uncond), retaining autograd.

    scale=0 selects unconditional prediction; scale=1 selects conditional;
    scale>1 extrapolates. Predictions must share shape, device, and floating
    dtype. scale is a finite real number or floating scalar tensor. Callers
    supply predictions from the same model using conditional and null inputs;
    learning that null behavior requires condition dropout during training.
    """
    if (not isinstance(uncond, torch.Tensor) or not isinstance(cond, torch.Tensor)
            or not uncond.is_floating_point() or not cond.is_floating_point()
            or uncond.shape != cond.shape):
        raise ValueError("predictions must be floating tensors with identical shape")
    _same_device_dtype(uncond, cond)
    if isinstance(scale, torch.Tensor):
        if (scale.ndim != 0 or not scale.is_floating_point()
                or scale.device != uncond.device or not torch.isfinite(scale)):
            raise ValueError("scale must be a finite floating scalar tensor on the prediction device")
    elif isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
        raise ValueError("scale must be a finite real number")
    return uncond + scale * (cond - uncond)
