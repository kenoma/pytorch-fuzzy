from typing import Union
import numpy as np
import torch
from torch import nn, Tensor

class DefuzzyMaxLayer(nn.Module):
    """
    Слой дефаззификации по максимуму.

    Два режима:

    * winner_take_all=False (по умолчанию) — поэлементный максимум:
            z_j = max_i ( Z_{j,i} * firing_i )
      Дифференцируем через субградиент torch.max.

    * winner_take_all=True — «победитель забирает всё»:
            i* = argmax_i (firing_i)
            z_j = Z_{j, i*}
      Forward — жёсткий one-hot по argmax.
      Backward — straight-through estimator через softmax(firings / tau),
      так что градиент течёт и по Z, и по входу x.

    Z имеет форму (size_out, size_in) — как в DefuzzyNWLayer.
    """

    def __init__(
        self,
        initial_consequences: Tensor,
        trainable: bool = True,
        winner_take_all: bool = False,
        tau: float = 1.0,
    ) -> None:
        super().__init__()

        if initial_consequences.ndim != 2:
            raise ValueError(
                f"initial_consequences must be 2D (size_out, size_in), "
                f"got shape {tuple(initial_consequences.shape)}"
            )
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")

        self.winner_take_all = winner_take_all
        self.tau = tau
        self.size_out, self.size_in = initial_consequences.shape

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
        winner_take_all: bool = False,
        tau: float = 1.0,
    ) -> "DefuzzyMaxLayer":
        Z = torch.rand(size_out, size_in)
        return cls(Z, trainable=trainable,
                   winner_take_all=winner_take_all, tau=tau)

    @classmethod
    def from_array(
        cls,
        initial_array: Union[np.ndarray, Tensor, list],
        trainable: bool = True,
        winner_take_all: bool = False,
        tau: float = 1.0,
    ) -> "DefuzzyMaxLayer":
        Z = torch.as_tensor(initial_array, dtype=torch.float32)
        return cls(Z, trainable=trainable,
                   winner_take_all=winner_take_all, tau=tau)

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

        if self.winner_take_all:
            # soft для backward
            soft = torch.softmax(x / self.tau, dim=-1)        # (B, size_in)
            # hard one-hot для forward
            idx = x.argmax(dim=-1, keepdim=True)              # (B, 1)
            hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
            # Straight-through: forward == hard, grad идёт через soft
            weights = hard + soft - soft.detach()             # (B, size_in)
            # (B, size_in) @ (size_in, size_out) -> (B, size_out)
            return weights @ self.Z.t()

        # элементный max по правилам
        weighted = x.unsqueeze(1) * self.Z.unsqueeze(0)        # (B, size_out, size_in)
        return weighted.max(dim=-1).values

    def extra_repr(self) -> str:
        return (
            f"size_in={self.size_in}, size_out={self.size_out}, "
            f"winner_take_all={self.winner_take_all}, tau={self.tau}, "
            f"trainable={self.Z.requires_grad}"
        )
