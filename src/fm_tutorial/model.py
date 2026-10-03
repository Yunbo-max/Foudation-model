"""An original, deliberately small decoder-only Transformer for the tutorial.

Attention is written as explicit matrix operations so its causal/key-padding
mask is visible. It materializes O(T²) attention and is not a production kernel.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

EOS_ID = 256
PAD_ID = 257


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 352
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size <= PAD_ID:
            raise ValueError("vocab_size must contain byte tokens, EOS=256 and PAD=257")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2 for next-token training")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x):
        # Promotion precedes squaring; a float16 square can overflow first.
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension, theta):
        super().__init__()
        frequency = theta ** (-torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension)
        self.register_buffer("frequency", frequency, persistent=False)

    def forward(self, x):
        # x: batch × heads × time × head_dimension. Buffers follow .to(device).
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.frequency.float())
        cosine = angles.cos().to(x.dtype)[None, None, :, :]
        sine = angles.sin().to(x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dimension = config.d_model // config.n_heads
        kv_dimension = self.n_kv_heads * self.head_dimension
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dimension, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dimension, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dimension, config.rope_theta)
        self.dropout = config.dropout

    def forward(self, x, valid_keys):
        batch, time, width = x.shape
        q = self.q_proj(x).view(batch, time, self.n_heads, self.head_dimension).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dimension).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dimension).transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)
        # Each KV head serves a contiguous group of query heads.
        repeats = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        # Matmul is autocast-eligible even with explicitly float32 operands.
        # Disable ambient AMP here so large finite QK scores cannot overflow
        # fp16 before softmax; the value/output projections still use AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dimension)
            causal = torch.ones((time, time), device=x.device, dtype=torch.bool).tril()
            allowed = causal[None, None, :, :] & valid_keys[:, None, None, :]
            # Finite sentinel plus post-softmax mask handles an all-PAD row safely.
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=-1).masked_fill(~allowed, 0.0)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        result = torch.matmul(weights.to(v.dtype), v)
        result = result.transpose(1, 2).contiguous().view(batch, time, width)
        return self.out_proj(result)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid_keys):
        x = x + self.dropout(self.attention(self.attention_norm(x), valid_keys))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids, labels=None):
        """Return logits and optional mean loss, shifting labels internally.

        Supply labels with exactly the input's shape. Logit at t predicts label
        at t+1; the last logit and first label do not contribute. PAD targets
        are ignored; all-ignored targets yield a differentiable zero.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape (batch, time)")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token ids")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input length exceeds configured context_length")
        if labels is not None:
            if labels.shape != input_ids.shape or labels.device != input_ids.device:
                raise ValueError("labels must have the same shape and device as input_ids")
            if labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("labels must be integer token ids")
        valid_keys = input_ids.ne(PAD_ID)
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, valid_keys)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            targets = labels[:, 1:].long()
            if targets.numel() == 0 or not targets.ne(PAD_ID).any():
                # The zero-loss branch must also promote before reduction:
                # finite fp16 logits can overflow their sum before *0.
                loss = logits.float().sum() * 0.0
            else:
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                                       targets.reshape(-1), ignore_index=PAD_ID)
        return {"logits": logits, "loss": loss}
