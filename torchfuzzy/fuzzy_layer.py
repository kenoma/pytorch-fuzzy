import torch
from torch import Tensor
from _fuzzy_layer_base import _FuzzyLayerBase

class FuzzyLayer(_FuzzyLayerBase):
    """mu(x) = exp(-||A(x-c)||^2)."""

    def _membership(self, rx2: Tensor) -> Tensor:
        return torch.exp(-rx2)