"""Original, small PyTorch implementations of Gaussian diffusion and a VP SDE.

Gaussian array index 0 denotes the first noisy state (paper time 1). The clean
state is separate, with alpha_bar=1; DDIM represents it with previous_t=-1.
Networks predict epsilon for GaussianDiffusion and the score for VPSDE.
These float32 teaching samplers use fixed variances and simple integrators.
"""
import math

import torch
from torch import nn


def _check_data(x):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError("data must be a floating-point tensor")
    if x.ndim < 2 or any(size <= 0 for size in x.shape):
        raise ValueError("data must have nonempty shape [batch, ...]")


def _check_like(value, x, name):
    _check_data(value)
    if value.shape != x.shape or value.device != x.device:
        raise ValueError(f"{name} must have the data shape and device")


def _noise_like(x, noise):
    if noise is None:
        return torch.randn_like(x)
    _check_like(noise, x, "noise")
    return noise.to(dtype=x.dtype)


def _shape(shape):
    shape = tuple(shape)
    if len(shape) < 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("shape must be positive integer dimensions [batch, ...]")
    return shape


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _eta(eta):
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must lie in [0, 1]")


def _float_times(t, x):
    _check_data(x)
    if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.shape != (x.shape[0],):
        raise ValueError("t must be a floating-point tensor of shape [batch]")
    if not torch.isfinite(t).all() or not ((t > 0) & (t <= 1)).all():
        raise ValueError("VP times must lie in (0, 1]")
    return t.to(device=x.device, dtype=x.dtype)


def _expand(values, x):
    return values.reshape((-1,) + (1,) * (x.ndim - 1))


class GaussianDiffusion(nn.Module):
    """DDPM with posterior variance, plus DDIM with an optional skipped grid.

    ``schedule='linear'`` linearly interpolates the given beta endpoints without
    rescaling them when steps changes. Cosine uses s=0.008 and beta<=0.999, as
    in Improved DDPM; beta_start/beta_end only affect the linear schedule.
    Models receive integer array times [B], and return an epsilon tensor [B,...].
    """

    def __init__(self, steps=100, beta_start=1e-4, beta_end=0.02, schedule="cosine"):
        super().__init__()
        _positive_integer(steps, "steps")
        if not (math.isfinite(beta_start) and math.isfinite(beta_end)
                and 0 < beta_start <= beta_end < 1):
            raise ValueError("beta endpoints must satisfy 0 < beta_start <= beta_end < 1")
        self.steps = steps
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        elif schedule == "cosine":
            times = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            curve = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
            betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        betas = betas.float()
        alpha_bar = (1 - betas).cumprod(0)
        if not ((betas > 0) & (betas < 1)).all() or not ((alpha_bar > 0) & (alpha_bar < 1)).all():
            raise ValueError("schedule must retain 0 < beta, alpha_bar < 1 in float32")
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)

    def _times(self, t, x, *, clean_allowed=False):
        _check_data(x)
        if not isinstance(t, torch.Tensor) or t.dtype != torch.long or t.shape != (x.shape[0],):
            raise ValueError("t must be a torch.long tensor of shape [batch]")
        minimum = -1 if clean_allowed else 0
        if not ((t >= minimum) & (t < self.steps)).all():
            raise ValueError(f"time indices must lie in [{minimum}, {self.steps - 1}]")
        return t.to(device=x.device)

    def _extract(self, values, t, x):
        return _expand(values.to(device=x.device, dtype=x.dtype)[t], x)

    def q_sample(self, x0, t, noise=None):
        """Draw xt = sqrt(alpha_bar[t])*x0 + sqrt(1-alpha_bar[t])*noise."""
        t = self._times(t, x0)
        a = self._extract(self.alpha_bar, t, x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * _noise_like(x0, noise)

    def predict_x0(self, xt, t, eps):
        """Invert the forward equation using predicted epsilon, without clipping."""
        t = self._times(t, xt)
        _check_like(eps, xt, "epsilon")
        a = self._extract(self.alpha_bar, t, xt)
        return (xt - (1 - a).sqrt() * eps) / a.sqrt()

    def posterior(self, x0, xt, t):
        """Return q(x_previous|xt,x0) mean and broadcastable scalar variance.

        Variance has shape [B,1,...]. At array t=0, the previous state is exactly
        x0 and its posterior variance is exactly zero.
        """
        t = self._times(t, xt)
        _check_like(x0, xt, "x0")
        beta = self._extract(self.betas, t, xt)
        a = self._extract(self.alpha_bar, t, xt)
        a_previous = self._extract(self.alpha_bar, (t - 1).clamp(min=0), xt)
        at_clean = _expand(t == 0, xt)
        a_previous = torch.where(at_clean, torch.ones_like(a_previous), a_previous)
        mean = (beta * a_previous.sqrt() / (1 - a)) * x0
        mean = mean + ((1 - a_previous) * (1 - beta).sqrt() / (1 - a)) * xt
        variance = beta * (1 - a_previous) / (1 - a)
        # Explicit clean endpoint also avoids coefficient roundoff at t=0.
        return torch.where(at_clean, x0, mean), variance

    def p_sample(self, model, xt, t, noise=None):
        """One adjacent DDPM step; no noise is added at array t=0."""
        t = self._times(t, xt)
        x0 = self.predict_x0(xt, t, model(xt, t))
        mean, variance = self.posterior(x0, xt, t)
        if noise is not None:
            _check_like(noise, xt, "noise")
        if not (t > 0).any():
            return mean
        return mean + variance.sqrt() * _noise_like(xt, noise)

    def ddim_step(self, model, xt, t, previous_t, eta=0.0, noise=None):
        """One DDIM jump, using alpha_bar at the selected previous_t.

        previous_t must be smaller than t for every batch element; -1 means the
        clean endpoint. eta=0 is deterministic given xt, and eta=1 matches the
        posterior-variance DDPM update when using adjacent times.
        """
        _eta(eta)
        t = self._times(t, xt)
        previous_t = self._times(previous_t, xt, clean_allowed=True)
        if not (previous_t < t).all():
            raise ValueError("previous_t must be strictly smaller than t")
        eps = model(xt, t)
        x0 = self.predict_x0(xt, t, eps)
        a = self._extract(self.alpha_bar, t, xt)
        b = self._extract(self.alpha_bar, previous_t.clamp(min=0), xt)
        b = torch.where(_expand(previous_t == -1, xt), torch.ones_like(b), b)
        variance = eta**2 * (1 - b) / (1 - a) * (1 - a / b)
        sigma = variance.clamp(min=0).sqrt()
        sample = b.sqrt() * x0 + (1 - b - variance).clamp(min=0).sqrt() * eps
        if noise is not None:
            _check_like(noise, xt, "noise")
        if eta > 0 and (previous_t >= 0).any():
            sample = sample + sigma * _noise_like(xt, noise)
        return sample

    @torch.no_grad()
    def sample(self, model, shape, device="cpu", sampler="ddpm", sampling_steps=None, eta=0.0):
        """Sample from a standard Gaussian prior using a fixed time grid.

        DDPM visits every adjacent step. DDIM can use 1..steps evaluations, with
        an evenly spaced integer grid starting at steps-1 and ending at 0
        (one evaluation jumps directly to clean). No model mode is changed.
        eta controls only DDIM; DDPM always uses the posterior variance.
        A short linear schedule need not end near the standard Gaussian prior.
        """
        shape = _shape(shape)
        _eta(eta)
        if sampler not in ("ddpm", "ddim"):
            raise ValueError("sampler must be 'ddpm' or 'ddim'")
        count = self.steps if sampling_steps is None else sampling_steps
        _positive_integer(count, "sampling_steps")
        if count > self.steps:
            raise ValueError("sampling_steps cannot exceed diffusion steps")
        if sampler == "ddpm" and count != self.steps:
            raise ValueError("DDPM requires all steps; use DDIM for skipped steps")
        xt = torch.randn(shape, device=device)
        if sampler == "ddpm":
            for index in range(self.steps - 1, -1, -1):
                t = torch.full((shape[0],), index, dtype=torch.long, device=device)
                xt = self.p_sample(model, xt, t)
        else:
            indices = torch.linspace(self.steps - 1, 0, count).round().long().tolist()
            for current, previous in zip(indices, indices[1:] + [-1]):
                t = torch.full((shape[0],), current, dtype=torch.long, device=device)
                previous_t = torch.full_like(t, previous)
                xt = self.ddim_step(model, xt, t, previous_t, eta=eta)
        return xt


class VPSDE:
    """Variance-preserving SDE with beta(t)=beta_min+(beta_max-beta_min)*t.

    Forward drift is -beta(t)*x/2 and diffusion amplitude is sqrt(beta(t)).
    Float times [B] lie in (0,1]. Score networks return grad_x log p_t(x),
    not epsilon; score_from_epsilon converts an epsilon predictor's output.
    """

    def __init__(self, beta_min=0.1, beta_max=20.0):
        if not (math.isfinite(beta_min) and math.isfinite(beta_max)
                and 0 < beta_min <= beta_max):
            raise ValueError("require 0 < beta_min <= beta_max")
        self.beta_min, self.beta_max = beta_min, beta_max

    def _integrated_beta(self, t):
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()

    def marginal_stats(self, x0, t):
        """Return analytical conditional mean [B,...] and std [B,1,...]."""
        t = _float_times(t, x0)
        integrated = _expand(self._integrated_beta(t), x0)
        mean = torch.exp(-0.5 * integrated) * x0
        # -expm1(-B) is stable when B is small near the clean endpoint.
        std = (-torch.expm1(-integrated)).sqrt()
        return mean, std

    def marginal(self, x0, t, noise=None):
        """Sample the analytical perturbation kernel at t, preserving shape."""
        mean, std = self.marginal_stats(x0, t)
        return mean + std * _noise_like(x0, noise)

    def score_from_epsilon(self, eps, t):
        """Convert predicted epsilon to score=-epsilon/std(t)."""
        t = _float_times(t, eps)
        std = _expand((-torch.expm1(-self._integrated_beta(t))).sqrt(), eps)
        return -eps / std

    def reverse_step(self, score_model, xt, t, dt, probability_flow=False, noise=None):
        """Euler step with negative dt, or Euler-Maruyama for the reverse SDE.

        Reverse SDE drift: f-g^2*score; ODE drift: f-g^2*score/2.
        Brownian noise has amplitude g*sqrt(-dt). The ODE draws no noise.
        """
        t = _float_times(t, xt)
        if not math.isfinite(dt) or dt >= 0 or (t + dt < 0).any():
            raise ValueError("dt must be negative and must not cross time zero")
        score = score_model(xt, t)
        _check_like(score, xt, "score")
        beta = _expand(self.beta_min + (self.beta_max - self.beta_min) * t, xt)
        factor = 0.5 if probability_flow else 1.0
        drift = -0.5 * beta * xt - factor * beta * score
        mean = xt + drift * dt
        if noise is not None:
            _check_like(noise, xt, "noise")
        if probability_flow:
            return mean
        return mean + (-dt * beta).sqrt() * _noise_like(xt, noise)

    @torch.no_grad()
    def sample(self, score_model, shape, device="cpu", steps=100, probability_flow=False, t_min=0.001):
        """Integrate from t=1 to t_min, returning the state at t_min.

        Initialization is N(0,I), an approximate terminal prior when integrated
        beta is large. This teaching Euler solver does no final denoising,
        adaptive error control, corrector steps, or likelihood computation.
        """
        shape = _shape(shape)
        _positive_integer(steps, "steps")
        if not math.isfinite(t_min) or not 0 < t_min < 1:
            raise ValueError("t_min must lie in (0,1)")
        xt = torch.randn(shape, device=device)
        grid = torch.linspace(1, t_min, steps + 1, device=device)
        for index in range(steps):
            t = grid[index].expand(shape[0])
            dt = (grid[index + 1] - grid[index]).item()
            xt = self.reverse_step(score_model, xt, t, dt, probability_flow=probability_flow)
        return xt
