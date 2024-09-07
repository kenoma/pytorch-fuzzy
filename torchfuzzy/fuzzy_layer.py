import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class FuzzyLayer(torch.nn.Module):

    def __init__(self, initial_centers, initial_scales, trainable=True):
        """
        mu_j(x,a,c) = exp(-|| a . x ||^2)
        """
        super().__init__()

        if np.shape(initial_centers) != np.shape(initial_scales):
            raise Exception("initial_centers shape does not match initial_scales")

        sizes = np.shape(initial_centers)
        self.size_out, self.size_in, *_ = sizes

        const_row = np.zeros(self.size_in+1)
        const_row[self.size_in] = 1
        const_row = np.array([const_row]*self.size_out)
        const_row = np.reshape(const_row, (self.size_out, 1, self.size_in+1))
        self.c_r = nn.Parameter(torch.FloatTensor(const_row), requires_grad=False)
        self.c_one = nn.Parameter(torch.FloatTensor([1]), requires_grad=False)
        
        self.scales = nn.Parameter(initial_scales, requires_grad=trainable)
        
        self.rots = []
        for i in range(self.size_in - 1):
            self.rots.append(nn.Parameter(torch.zeros((self.size_out, self.size_in - i - 1), requires_grad=trainable)))
        self.rots = nn.ParameterList(self.rots)
        self.centroids = nn.Parameter(initial_centers.reshape((self.size_out, self.size_in, 1)), requires_grad=trainable)    
        


    @classmethod
    def from_dimensions(cls, size_in, size_out, trainable=True):
        initial_centers = torch.randn((size_out, size_in))
        initial_scales = torch.ones((size_out, size_in))
        return cls(initial_centers, initial_scales, trainable)

    @classmethod
    def from_centers(cls, initial_centers, trainable=True):
        initial_centers = torch.FloatTensor(np.multiply(-1, initial_centers))
        initial_scales = torch.ones_like(initial_centers)
        return cls(initial_centers, initial_scales, trainable)
    
    @classmethod
    def from_centers_and_scales(cls, initial_centers, initial_scales, trainable=True):
        initial_centers = torch.FloatTensor(np.multiply(-1, initial_centers))
        initial_scales = torch.FloatTensor(initial_scales)
        return cls(initial_centers, initial_scales, trainable)

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        A = self.get_scales_and_rot()
        A = torch.cat((A, self.centroids), 2)
        ta = torch.cat([A, self.c_r], 1)
        repeated_one = self.c_one.repeat(batch_size, 1)
        ext_x = torch.cat([input, repeated_one], 1)
        #reshaped_x = torch.reshape(ext_x, (1, self.size_in+1))
        tx = torch.transpose(ext_x, 0, 1)
        mul = torch.matmul(ta, tx)
        exponents = torch.norm(mul[:,:self.size_in], p=2, dim=1)
        memberships = torch.exp(-exponents)
        return memberships.transpose(0,1)
    
    def set_requires_grad_rot(self, requires_grad):
        for i in range(self.size_in - 1):
            self.rots[i].requires_grad = requires_grad

    def set_requires_grad_scales(self, requires_grad):
        self.scales.requires_grad = requires_grad

    def set_requires_grad_centroids(self, requires_grad):
        self.centroids.requires_grad = requires_grad
    
    def get_scales_and_rot(self):
        scales = self.scales
        A = torch.diag_embed(scales, 0)
        for i in range(self.size_in - 1):
            r = self.rots[i]
            A = A + torch.diag_embed(r, i+1)
            A = A + torch.diag_embed(r, i+1, -1, -2)
        return A
    
    def get_centroids(self):
        lh = self.get_scales_and_rot()
        rh = self.centroids.squeeze(-1)
        return torch.linalg.solve(lh, -rh)
            
    def get_transformation_matrix_eigenvals(self):
        A = self.get_scales_and_rot()
        return torch.linalg.eigvals(A)
    
    def get_transformation_matrix(self):
        A = self.get_scales_and_rot()
        A = torch.cat((A, self.centroids), 2)
        ta = torch.cat([A, self.c_r], 1)
        return ta
    
    


