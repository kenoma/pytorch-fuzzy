from __future__ import annotations
from typing import Iterable, Literal, Optional, Sequence
import torch
from torch import nn, Tensor

class _FuzzyLayerBase(nn.Module):
    """
    Базовый слой нечёткой логики:

        y_j  = A_j (x - c_j)                # с учётом input_mask
        rx2_j = || y_j ||^2
        mu_j(x) = _membership(rx2_j)        # определяется в подклассе
        out_j = mu_j * active_mask_j        # маскирование неактивных нейронов
    """

    def __init__(
        self,
        initial_centers: Tensor,            # (size_out, size_in)
        initial_scales: Tensor,             # (size_out, size_in)
        trainable: bool = True,
        input_mask: Optional[Tensor] = None,    # (size_out, size_in) bool/float
        active_mask: Optional[Tensor] = None,   # (size_out,)        bool/float
    ):
        super().__init__()
        if initial_centers.shape != initial_scales.shape:
            raise ValueError(
                f"centers {tuple(initial_centers.shape)} != "
                f"scales {tuple(initial_scales.shape)}"
            )
        self.size_out, self.size_in = initial_centers.shape

        self.centers = nn.Parameter(initial_centers.float(), requires_grad=trainable)
        self.scales  = nn.Parameter(initial_scales.float(),  requires_grad=trainable)

        n = self.size_in
        n_off = n * (n - 1) // 2
        self.rot = nn.Parameter(
            torch.zeros(self.size_out, n_off), requires_grad=trainable
        )

        iu = torch.triu_indices(n, n, offset=1)
        self.register_buffer("_iu_row", iu[0], persistent=False)
        self.register_buffer("_iu_col", iu[1], persistent=False)

        # --- masks ----------------------------------------------------------
        # active_mask: (size_out,) float в {0,1} — включён/выключен нейрон
        if active_mask is None:
            am = torch.ones(self.size_out)
        else:
            am = torch.as_tensor(active_mask).float().view(-1)
            if am.shape != (self.size_out,):
                raise ValueError(
                    f"active_mask shape must be ({self.size_out},), "
                    f"got {tuple(am.shape)}"
                )
        self.register_buffer("active_mask", am)

        # input_mask: (size_out, size_in) float в {0,1} — какой признак "видит" нейрон
        if input_mask is None:
            self.register_buffer("input_mask", None, persistent=False)
        else:
            im = torch.as_tensor(input_mask).float()
            if im.shape != (self.size_out, self.size_in):
                raise ValueError(
                    f"input_mask shape must be ({self.size_out}, {self.size_in}), "
                    f"got {tuple(im.shape)}"
                )
            self.register_buffer("input_mask", im)

    def _build_A(self) -> Tensor:
        """Симметричная матрица (size_out, n, n)."""
        A = torch.diag_embed(self.scales)
        if self.size_in > 1:
            A = A.clone()
            A[:, self._iu_row, self._iu_col] = self.rot
            A[:, self._iu_col, self._iu_row] = self.rot
        return A

    def _rx2(self, x: Tensor) -> Tensor:
        A    = self._build_A()                                  # (out, n, n)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)       # (batch, out, n)
        if self.input_mask is not None:
            diff = diff * self.input_mask.unsqueeze(0)
        mul  = torch.matmul(A.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
        return (mul * mul).sum(dim=-1)                          # (batch, out)

    def forward(self, x: Tensor) -> Tensor:
        out = self._membership(self._rx2(x))                    # (batch, out)
        # multiply сохраняет граф и обнуляет градиент неактивных нейронов
        if not torch.all(self.active_mask == 1):
            out = out * self.active_mask
        return out

    def _membership(self, rx2: Tensor) -> Tensor:               # override me
        raise NotImplementedError

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def active_indices(self) -> Tensor:
        return torch.nonzero(self.active_mask, as_tuple=False).view(-1)

    def set_active(self, indices: Sequence[int], active: bool = True) -> None:
        """Включает/выключает указанные выходные нейроны."""
        idx = torch.as_tensor(list(indices), dtype=torch.long,
                              device=self.active_mask.device)
        self.active_mask[idx] = float(bool(active))

    def reset_mask(self) -> None:
        self.active_mask.fill_(1.0)

    @torch.no_grad()
    def prune_inactive(self) -> "_FuzzyLayerBase":
        """
        Физически удаляет замаскированные нейроны из всех параметров/буферов.
        Уменьшает `size_out` и экономит память/вычисления.
        Возвращает self (in-place).
        """
        keep = self.active_mask.bool()
        if keep.all():
            return self
        kept_idx = torch.nonzero(keep, as_tuple=False).view(-1)

        self._replace_param("centers", self.centers.data[kept_idx])
        self._replace_param("scales",  self.scales.data[kept_idx])
        self._replace_param("rot",     self.rot.data[kept_idx])

        # подклассы могут переопределить _prune_extra, чтобы удалить свои параметры
        self._prune_extra(kept_idx)

        self.active_mask = self.active_mask[kept_idx].contiguous()
        if self.input_mask is not None:
            self.input_mask = self.input_mask[kept_idx].contiguous()

        self.size_out = int(keep.sum().item())
        return self

    def _prune_extra(self, kept_idx: Tensor) -> None:
        """Hook для подклассов — удалить их собственные (size_out,)-параметры."""
        pass

    def _replace_param(self, name: str, data: Tensor) -> None:
        old: nn.Parameter = getattr(self, name)
        new = nn.Parameter(data.clone(), requires_grad=old.requires_grad)
        delattr(self, name)
        self.register_parameter(name, new)

    def get_centroids(self) -> Tensor:
        return self.centers.detach().clone()

    def get_transformation_matrix(self) -> Tensor:
        return self._build_A()

    def get_transformation_matrix_eigenvals(self) -> Tensor:
        return torch.linalg.eigvalsh(self._build_A())

    def freeze(self, centers=False, scales=False, rot=False) -> "_FuzzyLayerBase":
        if centers: self.centers.requires_grad_(False)
        if scales:  self.scales.requires_grad_(False)
        if rot:     self.rot.requires_grad_(False)
        return self

    def set_requires_grad_centroids(self, flag: bool): self.centers.requires_grad_(flag)
    def set_requires_grad_scales(self,    flag: bool): self.scales.requires_grad_(flag)
    def set_requires_grad_rot(self,       flag: bool): self.rot.requires_grad_(flag)

    @classmethod
    def from_dimensions(cls, size_in: int, size_out: int, trainable: bool = True, **kw):
        return cls(
            torch.randn(size_out, size_in),
            torch.ones(size_out, size_in),
            trainable=trainable,
            **kw,
        )

    @classmethod
    def from_centers(cls, centers, scales=None, trainable: bool = True, **kw):
        centers = torch.as_tensor(centers, dtype=torch.float32)
        scales  = (torch.ones_like(centers) if scales is None
                   else torch.as_tensor(scales, dtype=torch.float32))
        return cls(centers, scales, trainable=trainable, **kw)

    from_centers_and_scales = from_centers