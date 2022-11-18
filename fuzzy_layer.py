import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class FuzzyLayer(torch.nn.Module):

    def __init__(self, initial_centers, initial_scales):
        """
        mu_j(x,a,c) = exp(-|| a . x ||^2)
        """
        super().__init__()

        if np.shape(initial_centers) != np.shape(initial_scales):
            raise Exception("initial_centers shape does not match initial_scales")

        sizes = np.shape(initial_centers)
        self.size_out, self.size_in = sizes[0], sizes[1]

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
        self.A = nn.Parameter(a, requires_grad=True) 

    @classmethod
    def fromdimentions(cls, size_in, size_out):
        initial_centers = torch.randn((size_out, size_in))
        initial_scales = torch.ones((size_out, size_in))
        return cls(initial_centers, initial_scales)

    @classmethod
    def fromcenters(cls, initial_centers):
        initial_centers =  np.multiply(-1, initial_centers)
        sizes = np.shape(initial_centers)
        initial_scales = torch.ones(sizes)
        return cls(initial_centers, initial_scales)

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        ta = torch.cat([self.A, self.c_r],1)
        repeated_one = self.c_one.repeat(batch_size,1)
        ext_x = torch.cat([input, repeated_one], 1)
        #reshaped_x = torch.reshape(ext_x, (1, self.size_in+1))
        tx = torch.transpose(ext_x, 0, 1)
        mul = torch.matmul(ta, tx)
        exponents = torch.norm(mul[:,:self.size_in], p=2, dim=1)
        memberships = torch.exp(-exponents)
        return memberships.transpose(0,1)