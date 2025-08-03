import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class DefuzzyMaxLayer(torch.nn.Module):

    def __init__(self, initial_consequences, trainable):
        """
            z = max(Z_i*firing_i)
        """
        super().__init__()
        
        self.size_in = initial_consequences.shape[0]
        self.size_out = initial_consequences.shape[1]

    
        self.Z = nn.Parameter(initial_consequences, requires_grad=trainable)

    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True):
        Z = torch.rand(size_in, size_out)
        return cls(Z, trainable)
    
    def forward(self, fz: Tensor) -> Tensor:
        batch_size = fz.shape[0]
        
        rep_fz = fz.reshape((batch_size, self.size_in, 1)).expand((batch_size, self.size_in, self.size_out))

        rep_z = self.Z.expand((batch_size, self.size_in, self.size_out))
        
        return (rep_fz*rep_z).max(1).values

