from torch import Tensor
from .fuzzy_layer_base import FuzzyLayerBase

class FuzzyCauchyLayer(FuzzyLayerBase):
    """mu(x) = 1 / (1 + ||A(x-c)||^2)"""

    def _membership(self, rx2: Tensor) -> Tensor:
        return 1.0 / (1.0 + rx2)