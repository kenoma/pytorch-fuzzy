from typing import Union
import numpy as np
import torch
from torch import nn, Tensor

class DefuzzyNWLayer(nn.Module):
    """
    Слой дефаззификации (Надарая-Ватсон):

        z = sum_i (Z_i * firing_i) / sum_i (firing_i)

    где firing_i — степени активации правил (входы слоя),
    а Z_i — обучаемые "consequences" (заключения).
    При with_norm=False слой вырождается в nn.Linear(bias=False)
    """

    def __init__(
        self,
        initial_consequences: Tensor,
        trainable: bool = True,
        with_norm: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if initial_consequences.ndim != 2:
            raise ValueError(
                f"initial_consequences must be 2D (size_out, size_in), "
                f"got shape {tuple(initial_consequences.shape)}"
            )

        self.with_norm = with_norm
        self.eps = eps
        self.size_out, self.size_in = initial_consequences.shape

        # shape: (size_out, size_in) — broadcast по батчу даст matmul
        self.Z = nn.Parameter(
            initial_consequences.clone().float(),
            requires_grad=trainable,
        )

    @classmethod
    def from_dimensions(
        cls,
        size_in: int,
        size_out: int,
        trainable: bool = True,
        with_norm: bool = True,
    ) -> "DefuzzyNWLayer":
        Z = torch.rand(size_out, size_in)
        return cls(Z, trainable=trainable, with_norm=with_norm)

    @classmethod
    def from_array(
        cls,
        initial_array: Union[np.ndarray, Tensor, list],
        trainable: bool = True,
        with_norm: bool = True,
    ) -> "DefuzzyNWLayer":
        Z = torch.as_tensor(initial_array, dtype=torch.float32)
        return cls(Z, trainable=trainable, with_norm=with_norm)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, size_in) — firings
        returns: (batch_size, size_out)
        """
        if x.ndim != 2 or x.shape[1] != self.size_in:
            raise ValueError(
                f"Expected input of shape (batch, {self.size_in}), "
                f"got {tuple(x.shape)}"
            )

        if self.with_norm:
            denom = x.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            x = x / denom

        # (batch, size_in) @ (size_in, size_out) -> (batch, size_out)
        return x @ self.Z.t()

    def extra_repr(self) -> str:
        return (
            f"size_in={self.size_in}, size_out={self.size_out}, "
            f"with_norm={self.with_norm}, trainable={self.Z.requires_grad}"
        )
