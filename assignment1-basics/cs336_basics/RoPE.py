import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        dim_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dim_idx / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        angles = torch.outer(positions, inv_freq)
       
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)