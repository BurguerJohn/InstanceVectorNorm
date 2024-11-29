import torch
import torch.nn as nn

class InstanceNormVector2D(nn.Module):
    def __init__(self, eps=1e-5):
        super(InstanceNormVector2D, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)  # Mean over height and width
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # Variance over height and width

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x / x.norm(p=2, dim=[1,2,3], keepdim=True).clamp(min=self.eps)
      
        return x
        

t = torch.rand(1, 3, 128, 128)

inst = InstanceNormVector2D()

print(inst(t))