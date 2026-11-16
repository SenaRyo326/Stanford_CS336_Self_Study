import torch
import torch.nn as nn
from cs336_basics.linear import Linear
class SwiGLU(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
        super().__init__()
        if d_ff is None:
            d_ff = max(64, round((8 / 3 * d_model) / 64) * 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor :


        return self.w2((self.w1(x)*torch.sigmoid(self.w1(x)))*(self.w3(x)))

        
    