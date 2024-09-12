import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class DefuzzyLinearLayer(torch.nn.Module):

    def __init__(self, initial_consequences, trainable, with_norm):
        """
            z = sum(Z_i*firing_i)/sum(firing_i)
        """
        super().__init__()
        self.with_norm = with_norm
        self.size_out = initial_consequences.shape[0]
        self.size_in = initial_consequences.shape[1]
        self.Z = nn.Parameter(initial_consequences.reshape((1, self.size_out, self.size_in)), requires_grad=trainable)

    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True, with_norm=True):
        Z = torch.rand(size_out, size_in)
        return cls(Z, trainable, with_norm)
    
    @classmethod
    def from_array(cls, initial_array, trainable=True, with_norm=True):
        Z = torch.FloatTensor(np.array(initial_array))
        return cls(Z, trainable, with_norm)

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        
        rep_z = self.Z.expand((batch_size, self.size_out, self.size_in))
        
        if self.with_norm:
            norm_inp = input / input.sum(-1).reshape((batch_size, 1)).expand((batch_size, self.size_in))
            return torch.squeeze(torch.bmm(rep_z, norm_inp.reshape(batch_size, self.size_in, 1)), 2)
        else:
            return torch.squeeze(torch.bmm(rep_z, input.reshape(batch_size, self.size_in, 1)), 2)

