import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")