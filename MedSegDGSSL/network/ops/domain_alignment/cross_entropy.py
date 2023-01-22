import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropyDistance(nn.Module):
    """CrossEntropy Distance

    Calculate the distance between two distribution using 
    """
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6
        self.activation = F.softmax
    
    def forward(self, x, y):
        x = self.activation(x, dim=-1)
        y = self.activation(y, dim=-1)
        loss = torch.mean(-torch.mm(x, torch.log(torch.clip(y.t(), min=self.eps)))-
                          torch.mm(y, torch.log(torch.clip(x.t(), min=self.eps))))
        return loss
