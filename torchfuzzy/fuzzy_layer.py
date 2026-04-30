import torch
from torch import Tensor
from torchfuzzy.fuzzy_layer_base import FuzzyLayerBase

class FuzzyLayer(FuzzyLayerBase):
    """mu(x) = exp(-||A(x-c)||^2)."""

    def _membership(self, rx2: Tensor) -> Tensor:
        return torch.exp(-rx2)