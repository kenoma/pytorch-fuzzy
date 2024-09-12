import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class FuzzyBellLayer(torch.nn.Module):

    def __init__(self, initial_centers, initial_scales, initial_pow, trainable=True):
        """
        mu_j(x,a,c) =1/( 1 + || (x - c) / a ||^(2*b) )
        """
        super().__init__()

        if np.shape(initial_centers) != np.shape(initial_scales):
            raise Exception("initial_centers shape does not match initial_scales")

        sizes = np.shape(initial_centers)
        self.size_out, self.size_in, *_ = sizes
        self.a = nn.Parameter(initial_scales, requires_grad=trainable) 
        self.b = nn.Parameter(initial_pow, requires_grad=trainable)
        self.c = nn.Parameter(initial_centers, requires_grad=trainable)

    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True):
        initial_centers = torch.randn((size_out, size_in))
        initial_scales = torch.ones((size_out, size_in))
        initial_pow = torch.ones(size_out)
        return cls(initial_centers, initial_scales, initial_pow, trainable)

    @classmethod
    def from_centers(cls, initial_centers, trainable=True):
        initial_centers =  torch.FloatTensor(initial_centers)
        initial_scales = torch.ones_like(initial_centers)
        initial_pow = torch.ones(initial_centers.shape[0])
        return cls(initial_centers, initial_scales, initial_pow, trainable)
    
    @classmethod
    def from_centers_and_scales(cls, initial_centers, initial_scales, trainable=True):
        initial_centers =  torch.FloatTensor(initial_centers)
        initial_scales =  torch.FloatTensor(initial_scales)
        initial_pow = torch.ones(initial_centers.shape[0])
        return cls(initial_centers, initial_scales, initial_pow, trainable)

    @classmethod
    def from_centers_scales_and_pow(cls, initial_centers, initial_scales, initial_pow, trainable=True):
        initial_centers =  torch.FloatTensor(initial_centers)
        initial_scales =  torch.FloatTensor(initial_scales)
        initial_pow = torch.FloatTensor(initial_pow)
        
        return cls(initial_centers, initial_scales, initial_pow, trainable)

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        
        rep_x = input.repeat_interleave(self.size_out, 0).reshape((batch_size, self.size_out, self.size_in))
        rep_a = self.a.where(self.a > 1e-5, 1e-5)
        rep_a = rep_a.repeat(batch_size, 1).reshape((batch_size, self.size_out, self.size_in))
        rep_b = self.b.abs().repeat(batch_size, 1).reshape((batch_size, -1))
        rep_c = self.c.repeat(batch_size, 1).reshape((batch_size, self.size_out, self.size_in))
        
        fat = ((rep_x - rep_c) / rep_a).norm(2, dim =-1).pow(2 * rep_b)
        
        return 1/(1+fat)
    
    def get_centroids(self):
        return self.c.detach().clone()

