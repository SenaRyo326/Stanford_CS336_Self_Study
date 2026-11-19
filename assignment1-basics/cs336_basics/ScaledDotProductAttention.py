import torch
import einops
import math
from cs336_basics.softmax import softmax
def ScaledDotProductAttention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask : torch.Tensor | None = None 
) -> torch.Tensor:
    score = einops.einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(q.shape[-1])
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    return einops.einsum(softmax(score, -1), v, "...  q k, ... k v -> ... q v")

