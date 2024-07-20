import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class DefuzzyLinearLayer(torch.nn.Module):
    """
    WIP, use it on own risk
    """

    def __init__(self, initial_consequences):
        """
            z = sum(Z_i*firing_i)/sum(firing_i)
        """
        super().__init__()
        
        self.size_out = initial_consequences.shape[0]
        self.size_in = initial_consequences.shape[1]
        self.Z = nn.Parameter(initial_consequences.reshape((1, self.size_out, self.size_in)))

    @classmethod
    def from_dimensions(cls, size_in, size_out):
        Z = torch.rand(size_out, size_in)
        return cls(Z)

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        
        rep_z = self.Z.expand((batch_size, self.size_out, self.size_in))
        
        norm_inp = input / input.sum(-1).reshape((batch_size, 1)).expand((batch_size, self.size_in))
        
        return torch.squeeze(torch.bmm(rep_z, norm_inp.reshape(batch_size, self.size_in, 1)), 2)

