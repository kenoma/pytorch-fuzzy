import torch
from torch import Tensor
from _fuzzy_layer_base import _FuzzyLayerBase

class FuzzyCauchyLayer(_FuzzyLayerBase):
    """mu(x) = 1 / (1 + ||A(x-c)||^2)"""
    
    def _membership(self, rx2: Tensor) -> Tensor:
        return 1.0 / (1.0 + rx2)