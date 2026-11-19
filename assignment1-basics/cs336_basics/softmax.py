import torch
def softmax(
        x: torch.Tensor,
        dim: int
)-> torch.Tensor:
    max = x.amax(dim=dim, keepdim=True)
    x = torch.exp(x-max)
    sum = torch.sum(x, dim=dim, keepdim=True)
    return x / sum

    
    