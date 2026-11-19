import torch
from cs336_basics.linear import Linear
from cs336_basics.RoPE import RotaryPositionalEmbedding
from cs336_basics.ScaledDotProductAttention import ScaledDotProductAttention
import einops

class  CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,
                d_model : int,
                num_heads : int,
                max_seq_len : int | None = None,
                theta : float | None = None,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
                ):
        super().__init__()
        self.q_proj = Linear(d_model,d_model,device,dtype)
        self.k_proj = Linear(d_model,d_model,device,dtype)
        self.v_proj = Linear(d_model,d_model,device,dtype)
        self.o_proj = Linear(d_model,d_model,device,dtype)
        self.num_heads = num_heads
        self.d_k = d_model / num_heads

        if max_seq_len is not None and theta is not None:
            self.RoPE = RotaryPositionalEmbedding(theta , self.d_k , max_seq_len , device)
        else: self.RoPE = None

    def forward(self,
                x : torch.Tensor,
                token_positions: torch.Tensor | None = None
    ) -> torch.Tensor :
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q , "... seq (h d)-> ... h seq d" , h = self.num_heads)
        k = einops.rearrange(k , "... seq (h d)-> ... h seq d" , h = self.num_heads)
        v = einops.rearrange(v , "... seq (h d)-> ... h seq d" , h = self.num_heads)

        if self.RoPE is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)

            q = self.RoPE(q, token_positions)
            k = self.RoPE(k, token_positions)

        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device), diagonal=1)

        scores = ScaledDotProductAttention (q, k, v, mask)

        return self.o_proj(einops.rearrange(scores , "... h seq d -> ... seq (h d)", h = self.num_heads))

        


        



        