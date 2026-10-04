"""A tiny time-conditioned denoiser for synthetic diffusion demonstrations."""
import math

import torch
from torch import nn


class TimeMLP(nn.Module):
    """Flatten each sample, append three time features, and restore its shape.

    input_dim is the product of the nonbatch dimensions. t is floating [B] in
    [0,1]. For GaussianDiffusion's integer array times use (t.float()+1)/steps
    in the training code and in a sampling wrapper. This global MLP has no
    spatial or temporal inductive bias and is not an image/video architecture.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 2:
            raise ValueError("x must be floating data with shape [batch, ...]")
        if x.shape[0] <= 0 or math.prod(x.shape[1:]) != self.input_dim:
            raise ValueError("product of nonbatch dimensions must equal input_dim")
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.shape != (x.shape[0],):
            raise ValueError("t must be floating with shape [batch]")
        if not torch.isfinite(t).all() or not ((t >= 0) & (t <= 1)).all():
            raise ValueError("normalized times must lie in [0,1]")
        t = t.to(device=x.device, dtype=x.dtype)
        features = torch.stack((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=-1)
        flat = torch.cat((x.reshape(x.shape[0], self.input_dim), features), dim=-1)
        return self.net(flat).reshape_as(x)
