import torch
from cs336_basics.embedding import Embedding
from cs336_basics.TransformerBlock import Transformer
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float ,
            vocab_size: int,
            num_layers: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.Embed = Embedding(vocab_size, d_model, device, dtype)
        self.Transformer = torch.nn.ModuleList([Transformer(d_model,num_heads,d_ff,max_seq_len,theta,device,dtype) for _ in range(num_layers)])
        self.Norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.Out = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.Embed(x)
        for Transform in self.Transformer: x = Transform(x)
        x = self.Norm(x)
        x = self.Out(x)
        return x