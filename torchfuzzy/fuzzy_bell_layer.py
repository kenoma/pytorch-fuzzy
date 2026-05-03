from __future__ import annotations
from typing import Literal, Optional
import torch
from torch import nn, Tensor
from .fuzzy_layer_base import FuzzyLayerBase

BParam = Literal["raw", "softplus", "exp"]
class FuzzyBellLayer(FuzzyLayerBase):
    """
    mu(x) = 1 / (1 + ||A(x-c)||^{2b})

    Параметризация b управляется аргументом `b_parametrization`:
      - "raw"      : b = b_raw                     (без ограничений, может быть <= 0)
      - "softplus" : b = softplus(b_raw) + eps     (гарантия, что b > 0)
      - "exp"      : b = exp(b_raw)                (мультипликативная)
    """

    _EPS = 1e-6

    def __init__(
        self,
        initial_centers: Tensor,
        initial_scales: Tensor,
        initial_pow: Optional[Tensor] = None,
        trainable: bool = True,
        b_parametrization: BParam = "softplus",
        **kw,
    ):
        super().__init__(initial_centers, initial_scales, trainable=trainable, **kw)

        if b_parametrization not in ("raw", "softplus", "exp"):
            raise ValueError(f"unknown b_parametrization: {b_parametrization!r}")
        self.b_parametrization = b_parametrization

        if initial_pow is None:
            init_b = torch.ones(self.size_out)
        else:
            init_b = torch.as_tensor(initial_pow, dtype=torch.float32)
            if init_b.shape != (self.size_out,):
                raise ValueError(
                    f"initial_pow must have shape ({self.size_out},), "
                    f"got {tuple(init_b.shape)}"
                )
            if b_parametrization != "raw" and (init_b <= 0).any():
                raise ValueError("initial_pow must be positive for softplus/exp")

        # Инвертируем параметризацию, чтобы после forward получилось ровно initial_pow
        self._b_raw = nn.Parameter(
            self._inverse_param(init_b), requires_grad=trainable
        )

    def _inverse_param(self, b: Tensor) -> Tensor:
        if self.b_parametrization == "raw":
            return b.clone()
        if self.b_parametrization == "exp":
            return torch.log(b)
        
        y = (b - self._EPS).clamp_min(1e-8)
        return torch.log(torch.expm1(y))

    @property
    def b(self) -> Tensor:
        if self.b_parametrization == "raw":
            return self._b_raw
        if self.b_parametrization == "exp":
            return torch.exp(self._b_raw)
        return nn.functional.softplus(self._b_raw) + self._EPS

    def _membership(self, rx2: Tensor) -> Tensor:
        return 1.0 / (1.0 + (rx2 + self._EPS).pow(self.b))

    def _prune_extra(self, kept_idx: Tensor) -> None:
        self._replace_param("_b_raw", self._b_raw.data[kept_idx])

    def freeze(self, centers=False, scales=False, rot=False, powers=False):
        super().freeze(centers=centers, scales=scales, rot=rot)
        if powers:
            self._b_raw.requires_grad_(False)
        return self

    def set_requires_grad_powers(self, flag: bool):
        self._b_raw.requires_grad_(flag)

    @classmethod
    def from_centers_scales_and_pow(
        cls, centers, scales, powers, trainable: bool = True,
        b_parametrization: BParam = "softplus", **kw,
    ):
        return cls(
            torch.as_tensor(centers, dtype=torch.float32),
            torch.as_tensor(scales,  dtype=torch.float32),
            torch.as_tensor(powers,  dtype=torch.float32),
            trainable=trainable,
            b_parametrization=b_parametrization,
            **kw,
        )