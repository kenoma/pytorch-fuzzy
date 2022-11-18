import torch
import torch.nn as nn
import numpy as np

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
        a = torch.FloatTensor(diags)

        const_row = np.zeros(self.size_in+1)
        const_row[self.size_in] = 1
        const_row = np.array([const_row]*self.size_out)
        const_row = np.reshape(const_row, (self.size_out, 1, self.size_in+1))
        self.c_r = torch.FloatTensor(const_row)
        self.c_one = torch.FloatTensor([1])
        self.A = nn.Parameter(a) 

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

    def forward(self, x):
        ta = torch.cat([self.A, self.c_r],1)
        tx = torch.transpose(torch.reshape(torch.cat([x,self.c_one]), (1,self.size_in+1)),0,1)
        mul = torch.matmul(ta, tx)
        exponents = torch.norm(mul[:,:self.size_in], p=2, dim=1)
        memberships = torch.exp(-exponents)
        return memberships.flatten()