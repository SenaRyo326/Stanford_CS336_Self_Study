import torch
from cs336_basics.CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU

class Transformer(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float ,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.Norm1 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.Norm2 = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.Attention = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype) 
        self.FFN = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor | None = None)->torch.Tensor:
        y = x + self.Attention(self.Norm1(x), token_positions)
        return y + self.FFN(self.Norm2(y))
