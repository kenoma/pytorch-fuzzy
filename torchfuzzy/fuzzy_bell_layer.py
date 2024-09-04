import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class FuzzyBellLayer(torch.nn.Module):

    def __init__(self, initial_centers, initial_scales, initial_pow, trainable=True):
        """
        mu_j(x,a,c) =1/( 1 + || a . x ||^(2*b) )
        """
        super().__init__()

        if np.shape(initial_centers) != np.shape(initial_scales):
            raise Exception("initial_centers shape does not match initial_scales")

        sizes = np.shape(initial_centers)
        self.size_out, self.size_in, *_ = sizes

        diags = []
        for s,c in zip(initial_scales, initial_centers):
            diags.append(np.insert(np.diag(s), self.size_in, c, axis = 1))
        a = torch.FloatTensor(np.array(diags))

        const_row = np.zeros(self.size_in+1)
        const_row[self.size_in] = 1
        const_row = np.array([const_row]*self.size_out)
        const_row = np.reshape(const_row, (self.size_out, 1, self.size_in+1))
        self.c_r = nn.Parameter(torch.FloatTensor(const_row), requires_grad=False)
        self.c_one = nn.Parameter(torch.FloatTensor([1]), requires_grad=False)
        self.A = nn.Parameter(a, requires_grad=trainable) 
        self.B = nn.Parameter(initial_pow, requires_grad=trainable)

    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True):
        initial_centers = torch.randn((size_out, size_in))
        initial_scales = torch.ones((size_out, size_in))
        initial_pow = torch.ones(size_out)
        return cls(initial_centers, initial_scales, initial_pow, trainable)

    @classmethod
    def from_centers(cls, initial_centers, trainable=True):
        initial_centers =  np.multiply(-1, initial_centers)
        sizes = np.shape(initial_centers)
        initial_scales = torch.ones(sizes)
        initial_pow = torch.ones(initial_centers.shape[0])
        return cls(initial_centers, initial_scales, initial_pow, trainable)
    
    @classmethod
    def from_centers_and_scales(cls, initial_centers, initial_scales, trainable=True):
        initial_centers =  np.multiply(-1, initial_centers)
        initial_pow = torch.ones(initial_centers.shape[0])
        return cls(initial_centers, initial_scales, initial_pow, trainable)

    @classmethod
    def from_centers_scales_and_pow(cls, initial_centers, initial_scales, initial_pow, trainable=True):
        initial_centers =  np.multiply(-1, initial_centers)
        return cls(initial_centers, initial_scales, torch.tensor(initial_pow), trainable)

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        ta = torch.cat([self.A, self.c_r],1)
        repeated_one = self.c_one.repeat(batch_size,1)
        ext_x = torch.cat([input, repeated_one], 1)
        #reshaped_x = torch.reshape(ext_x, (1, self.size_in+1))
        tx = torch.transpose(ext_x, 0, 1)
        mul = torch.matmul(ta, tx)
        exponents = torch.norm(mul[:,:self.size_in], p=2, dim=1)
        memberships =1 / (1 + torch.pow(exponents.transpose(0,1), self.B))
        return memberships
