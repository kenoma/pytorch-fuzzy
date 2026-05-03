from __future__ import annotations

import math
import pytest
import torch
from torch import nn

from torchfuzzy import FuzzyLayer, FuzzyBellLayer, FuzzyCauchyLayer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def small_dims():
    return dict(size_in=4, size_out=3, batch=8)


@pytest.fixture
def gauss_layer(small_dims):
    return FuzzyLayer.from_dimensions(
        size_in=small_dims["size_in"],
        size_out=small_dims["size_out"],
    )


@pytest.fixture
def bell_layer(small_dims):
    return FuzzyBellLayer.from_dimensions(
        size_in=small_dims["size_in"],
        size_out=small_dims["size_out"],
    )


class TestConstruction:

    def test_shapes(self, gauss_layer, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        assert gauss_layer.centers.shape == (n_out, n_in)
        assert gauss_layer.scales.shape == (n_out, n_in)
        assert gauss_layer.rot.shape == (n_out, n_in * (n_in - 1) // 2)
        assert gauss_layer.active_mask.shape == (n_out,)

    def test_trainable_flag(self, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        layer = FuzzyLayer(
            torch.randn(n_out, n_in), torch.ones(n_out, n_in), trainable=False
        )
        for p in layer.parameters():
            assert p.requires_grad is False

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="centers.*scales"):
            FuzzyLayer(torch.randn(3, 4), torch.ones(3, 5))

    def test_bad_input_mask_raises(self):
        with pytest.raises(ValueError, match="input_mask"):
            FuzzyLayer(
                torch.randn(3, 4), torch.ones(3, 4),
                input_mask=torch.ones(3, 5),
            )

    def test_bad_active_mask_raises(self):
        with pytest.raises(ValueError, match="active_mask"):
            FuzzyLayer(
                torch.randn(3, 4), torch.ones(3, 4),
                active_mask=torch.ones(7),
            )

    def test_bad_initial_pow_shape(self):
        with pytest.raises(ValueError, match="initial_pow"):
            FuzzyBellLayer(
                torch.randn(3, 4), torch.ones(3, 4),
                initial_pow=torch.ones(5),
            )

    def test_cr_cone_are_not_parameters(self, gauss_layer):
        """Регресс: нетренируемые константы не должны попадать в parameters()."""
        names = {n for n, _ in gauss_layer.named_parameters()}
        assert "c_r" not in names and "c_one" not in names


class TestFactories:

    def test_from_centers_defaults(self):
        c = torch.randn(5, 3)
        layer = FuzzyLayer.from_centers(c)
        assert torch.allclose(layer.centers, c)
        assert torch.allclose(layer.scales, torch.ones_like(c))

    def test_from_centers_accepts_list(self):
        layer = FuzzyLayer.from_centers([[0.0, 1.0], [2.0, -1.0]])
        assert layer.size_out == 2 and layer.size_in == 2

    def test_from_centers_and_scales_behaves_as_from_centers(self):
        torch.manual_seed(0)
        c = torch.randn(3, 2)
        s = torch.rand(3, 2) + 0.1

        a = FuzzyLayer.from_centers(c, s)
        b = FuzzyLayer.from_centers_and_scales(c, s)

        x = torch.randn(5, 2)
        assert torch.allclose(a(x), b(x), atol=1e-6)
        
    def test_bell_from_centers_scales_and_pow(self):
        powers = torch.tensor([0.5, 1.0, 2.0])
        bell = FuzzyBellLayer.from_centers_scales_and_pow(
            torch.randn(3, 2), torch.ones(3, 2), powers,
            b_parametrization="softplus",
        )
        assert torch.allclose(bell.b.detach(), powers, atol=1e-5)

class TestForwardMath:

    def test_gaussian_formula(self, small_dims):
        n_in, n_out, bs = small_dims.values()
        centers = torch.randn(n_out, n_in)
        scales  = torch.rand(n_out, n_in) + 0.5
        layer = FuzzyLayer(centers, scales)
        x = torch.randn(bs, n_in)

        # эталон: diagonal A => A(x-c) = scales*(x-c)
        diff = x.unsqueeze(1) - centers.unsqueeze(0)        # (bs, out, n)
        y = diff * scales.unsqueeze(0)                      # (bs, out, n)
        expected = torch.exp(-(y * y).sum(-1))
        assert torch.allclose(layer(x), expected, atol=1e-6)

    def test_bell_formula(self, small_dims):
        n_in, n_out, bs = small_dims.values()
        centers = torch.randn(n_out, n_in)
        scales  = torch.rand(n_out, n_in) + 0.5
        powers  = torch.rand(n_out) + 0.5
        bell = FuzzyBellLayer(
            centers, scales, powers, b_parametrization="raw",
        )
        x = torch.randn(bs, n_in)

        diff = x.unsqueeze(1) - centers.unsqueeze(0)
        y = diff * scales.unsqueeze(0)
        rx2 = (y * y).sum(-1)
        expected = 1.0 / (1.0 + (rx2 + bell._EPS).pow(powers))
        assert torch.allclose(bell(x), expected, atol=1e-6)

    def test_membership_in_unit_interval(self, gauss_layer, bell_layer, small_dims):
        x = torch.randn(small_dims["batch"], small_dims["size_in"]) * 3
        for layer in (gauss_layer, bell_layer):
            out = layer(x)
            assert (out >= 0).all() and (out <= 1 + 1e-6).all()

    def test_center_gives_one(self, small_dims):
        """mu(c) должно равняться 1 (с точностью eps)."""
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        centers = torch.randn(n_out, n_in)
        gauss = FuzzyLayer(centers, torch.ones(n_out, n_in))
        out = gauss(centers)                                # (out, out)
        # на диагонали — значение в собственном центре
        assert torch.allclose(out.diag(), torch.ones(n_out), atol=1e-5)

    def test_output_shape(self, gauss_layer, small_dims):
        x = torch.randn(small_dims["batch"], small_dims["size_in"])
        assert gauss_layer(x).shape == (small_dims["batch"],
                                        small_dims["size_out"])

    def test_size_in_one(self):
        """Вырожденный случай: нет внедиагональных элементов."""
        layer = FuzzyLayer(torch.zeros(2, 1), torch.ones(2, 1))
        assert layer.rot.numel() == 0
        out = layer(torch.zeros(3, 1))
        assert torch.allclose(out, torch.ones(3, 2))


class TestGradients:

    def test_gradients_flow_to_all_params(self, bell_layer, small_dims):
        x = torch.randn(small_dims["batch"], small_dims["size_in"])
        loss = bell_layer(x).sum()
        loss.backward()
        for name, p in bell_layer.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"nan/inf in grad of {name}"

    def test_trainable_false_no_grad(self, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        layer = FuzzyLayer(
            torch.randn(n_out, n_in), torch.ones(n_out, n_in), trainable=False
        )

        # 1) все параметры заморожены
        for p in layer.parameters():
            assert p.requires_grad is False

        # 2) выход не участвует в графе (нет grad_fn)
        x = torch.randn(small_dims["batch"], n_in)
        out = layer(x)
        assert out.requires_grad is False
        assert out.grad_fn is None

        # 3) если вход сам требует grad — граф собирается, backward работает,
        #    а у параметров слоя grad всё равно не появляется
        x_req = torch.randn(small_dims["batch"], n_in, requires_grad=True)
        layer(x_req).sum().backward()
        for p in layer.parameters():
            assert p.grad is None
        assert x_req.grad is not None   

    def test_trainable_false_backward_is_safe(self, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        layer = FuzzyLayer(
            torch.randn(n_out, n_in), torch.ones(n_out, n_in), trainable=False
        )
        x = torch.randn(small_dims["batch"], n_in, requires_grad=True)
        # backward теперь не падает, потому что граф есть через x
        layer(x).sum().backward()
        assert all(p.grad is None for p in layer.parameters()) 

    def test_gradient_descent_reduces_loss(self, small_dims):
        """Мини-overfit: проверяем, что обучение вообще работает."""
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        target = torch.rand(1, n_out)
        x = torch.randn(1, n_in)

        layer = FuzzyLayer.from_dimensions(n_in, n_out)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-1)

        loss0 = ((layer(x) - target) ** 2).mean().item()
        for _ in range(200):
            opt.zero_grad()
            loss = ((layer(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        loss1 = ((layer(x) - target) ** 2).mean().item()
        assert loss1 < loss0 * 0.1


class TestStability:

    def test_no_nan_at_center_bell(self, small_dims):
        """Главный смысл eps в (rx2+eps)^b: grad не должен быть NaN в центре."""
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        centers = torch.randn(n_out, n_in)
        bell = FuzzyBellLayer(
            centers, torch.ones(n_out, n_in),
            torch.full((n_out,), 0.5),
            b_parametrization="softplus",
        )
        out = bell(centers)
        out.sum().backward()
        for name, p in bell.named_parameters():
            assert torch.isfinite(p.grad).all(), f"nan grad in {name}"

    def test_no_nan_at_center_gauss(self, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        centers = torch.randn(n_out, n_in)
        gauss = FuzzyLayer(centers, torch.ones(n_out, n_in))
        out = gauss(centers)
        out.sum().backward()
        for p in gauss.parameters():
            assert torch.isfinite(p.grad).all()

    def test_large_input_no_overflow(self, gauss_layer, small_dims):
        x = torch.randn(small_dims["batch"], small_dims["size_in"]) * 1e3
        out = gauss_layer(x)
        assert torch.isfinite(out).all()
        assert (out >= 0).all()


class TestBParametrization:

    @pytest.mark.parametrize("mode", ["raw", "softplus", "exp"])
    def test_roundtrip_initial_pow(self, mode):
        powers = torch.tensor([0.3, 1.0, 2.5, 4.0])
        bell = FuzzyBellLayer(
            torch.randn(4, 3), torch.ones(4, 3),
            powers, b_parametrization=mode,
        )
        assert torch.allclose(bell.b.detach(), powers, atol=1e-4)

    @pytest.mark.parametrize("mode", ["softplus", "exp"])
    def test_b_is_strictly_positive(self, mode):
        bell = FuzzyBellLayer.from_dimensions(3, 4, b_parametrization=mode)
        # искусственно уводим _b_raw в большой минус
        with torch.no_grad():
            bell._b_raw.fill_(-50.0)
        assert (bell.b > 0).all()
        assert torch.isfinite(bell.b).all()

    def test_negative_initial_pow_rejected_for_softplus(self):
        with pytest.raises(ValueError, match="positive"):
            FuzzyBellLayer(
                torch.randn(3, 2), torch.ones(3, 2),
                torch.tensor([-1.0, 1.0, 2.0]),
                b_parametrization="softplus",
            )

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="b_parametrization"):
            FuzzyBellLayer.from_dimensions(3, 4, b_parametrization="weird")


class TestMasking:

    def test_active_mask_zeroes_output(self, gauss_layer, small_dims):
        gauss_layer.set_active([0, 2], active=False)
        x = torch.randn(small_dims["batch"], small_dims["size_in"])
        out = gauss_layer(x)
        assert (out[:, [0, 2]] == 0).all()
        assert (out[:, 1] != 0).any()

    def test_n_active(self, gauss_layer, small_dims):
        assert gauss_layer.n_active == small_dims["size_out"]
        gauss_layer.set_active([1], active=False)
        assert gauss_layer.n_active == small_dims["size_out"] - 1

    def test_reset_mask(self, gauss_layer, small_dims):
        gauss_layer.set_active([0, 1], active=False)
        gauss_layer.reset_mask()
        assert gauss_layer.n_active == small_dims["size_out"]

    def test_active_indices(self, gauss_layer):
        gauss_layer.set_active([1], active=False)
        assert gauss_layer.active_indices.tolist() == [0, 2]

    def test_mask_blocks_gradient_to_dead_neurons(self, small_dims):
        """Градиент по mul на 0 → градиенты по параметрам мёртвых нейронов тоже 0."""
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        layer = FuzzyLayer.from_dimensions(n_in, n_out)
        layer.set_active([0], active=False)
        x = torch.randn(small_dims["batch"], n_in)
        layer(x).sum().backward()
        assert torch.allclose(layer.centers.grad[0], torch.zeros(n_in))
        # остальные нейроны должны иметь ненулевые градиенты
        assert not torch.allclose(layer.centers.grad[1], torch.zeros(n_in))

    def test_input_mask_applied(self, small_dims):
        """Если признак полностью замаскирован, сдвиг по нему не влияет на выход."""
        n_in, n_out, bs = small_dims.values()
        mask = torch.ones(n_out, n_in)
        mask[:, 0] = 0                                       # никто не видит признак 0
        layer = FuzzyLayer(
            torch.zeros(n_out, n_in), torch.ones(n_out, n_in),
            input_mask=mask,
        )
        x1 = torch.randn(bs, n_in)
        x2 = x1.clone()
        x2[:, 0] += 100.0                                    # сильно "ломаем" признак 0
        assert torch.allclose(layer(x1), layer(x2), atol=1e-5)


class TestPruning:

    def test_prune_reduces_sizes(self, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        layer = FuzzyLayer.from_dimensions(n_in, n_out)
        layer.set_active([1], active=False)
        layer.prune_inactive()
        assert layer.size_out == n_out - 1
        assert layer.centers.shape == (n_out - 1, n_in)
        assert layer.scales.shape == (n_out - 1, n_in)
        assert layer.active_mask.shape == (n_out - 1,)

    def test_prune_preserves_surviving_outputs(self, small_dims):
        n_in, n_out, bs = small_dims.values()
        layer = FuzzyLayer.from_dimensions(n_in, n_out)
        x = torch.randn(bs, n_in)
        layer.set_active([1], active=False)
        kept = [i for i in range(n_out) if i != 1]
        out_before = layer(x)[:, kept].clone()
        layer.prune_inactive()
        out_after = layer(x)
        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_prune_all_active_is_noop(self, gauss_layer):
        before = gauss_layer.centers.data.clone()
        gauss_layer.prune_inactive()
        assert torch.equal(gauss_layer.centers.data, before)

    def test_prune_bell_also_prunes_b(self, small_dims):
        n_in, n_out = small_dims["size_in"], small_dims["size_out"]
        bell = FuzzyBellLayer.from_dimensions(n_in, n_out)
        bell.set_active([0, 2], active=False)
        bell.prune_inactive()
        assert bell._b_raw.shape == (n_out - 2,)
        assert bell.b.shape == (n_out - 2,)

    def test_prune_keeps_param_trainable(self, small_dims):
        layer = FuzzyLayer.from_dimensions(
            small_dims["size_in"], small_dims["size_out"]
        )
        layer.set_active([1], active=False)
        layer.prune_inactive()
        assert layer.centers.requires_grad is True
        x = torch.randn(small_dims["batch"], small_dims["size_in"])
        layer(x).sum().backward()
        assert layer.centers.grad is not None

    def test_prune_optimizer_note(self, small_dims):
        """После prune оптимизатор нужно пересоздать (иначе ошибка размеров)."""
        layer = FuzzyLayer.from_dimensions(
            small_dims["size_in"], small_dims["size_out"]
        )
        opt = torch.optim.SGD(layer.parameters(), lr=1e-2)
        layer.set_active([1], active=False)
        layer.prune_inactive()
        # старый opt держит ссылку на удалённые параметры — теряет связь
        # переинициализируем:
        opt = torch.optim.SGD(layer.parameters(), lr=1e-2)
        x = torch.randn(small_dims["batch"], small_dims["size_in"])
        layer(x).sum().backward()
        opt.step()                                           # не должно падать


class TestFreezing:

    def test_freeze_centers(self, gauss_layer):
        gauss_layer.freeze(centers=True)
        assert gauss_layer.centers.requires_grad is False
        assert gauss_layer.scales.requires_grad is True

    def test_freeze_all_bell(self, bell_layer):
        bell_layer.freeze(centers=True, scales=True, rot=True, powers=True)
        for p in bell_layer.parameters():
            assert p.requires_grad is False

    def test_legacy_setters(self, gauss_layer):
        gauss_layer.set_requires_grad_centroids(False)
        gauss_layer.set_requires_grad_scales(False)
        gauss_layer.set_requires_grad_rot(False)
        for p in gauss_layer.parameters():
            assert p.requires_grad is False


class TestMatrixUtilities:

    def test_A_diagonal_is_scales(self, gauss_layer):
        A = gauss_layer.get_transformation_matrix()
        assert torch.allclose(torch.diagonal(A, dim1=-2, dim2=-1),
                              gauss_layer.scales)

    def test_eigvalsh_real(self, gauss_layer):
        with torch.no_grad():
            gauss_layer.rot.normal_()
        ev = gauss_layer.get_transformation_matrix_eigenvals()
        assert ev.dtype in (torch.float32, torch.float64)

    def test_get_centroids_is_detached(self, gauss_layer):
        c = gauss_layer.get_centroids()
        assert c.requires_grad is False
        # изменение копии не должно трогать параметр
        c += 1.0
        assert not torch.allclose(c, gauss_layer.centers)

class TestSerialization:

    def test_state_dict_roundtrip_gauss(self, small_dims):
        n_in, n_out, bs = small_dims.values()
        layer = FuzzyLayer.from_dimensions(n_in, n_out)
        layer.set_active([0], active=False)
        x = torch.randn(bs, n_in)
        out = layer(x).clone()

        clone = FuzzyLayer.from_dimensions(n_in, n_out)
        clone.load_state_dict(layer.state_dict())
        assert torch.allclose(clone(x), out, atol=1e-6)
        assert clone.n_active == n_out - 1

    def test_state_dict_roundtrip_bell(self, small_dims):
        n_in, n_out, bs = small_dims.values()
        bell = FuzzyBellLayer.from_dimensions(n_in, n_out,
                                              b_parametrization="softplus")
        with torch.no_grad():
            bell._b_raw.normal_()
        x = torch.randn(bs, n_in)
        out = bell(x).clone()

        clone = FuzzyBellLayer.from_dimensions(n_in, n_out,
                                               b_parametrization="softplus")
        clone.load_state_dict(bell.state_dict())
        assert torch.allclose(clone(x), out, atol=1e-6)


class TestDeviceDtype:

    def test_to_double(self, gauss_layer, small_dims):
        layer = gauss_layer.double()
        x = torch.randn(small_dims["batch"], small_dims["size_in"],
                        dtype=torch.float64)
        out = layer(x)
        assert out.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA недоступна")
    def test_cuda(self, gauss_layer, small_dims):
        layer = gauss_layer.cuda()
        # буферы должны переехать вместе с параметрами
        assert layer.active_mask.is_cuda
        assert layer._il_row.is_cuda
        assert layer._il_col.is_cuda
        x = torch.randn(small_dims["batch"], small_dims["size_in"]).cuda()
        out = layer(x)
        assert out.is_cuda


class TestBatchShapes:

    def test_batch_one(self, gauss_layer, small_dims):
        x = torch.randn(1, small_dims["size_in"])
        out = gauss_layer(x)
        assert out.shape == (1, small_dims["size_out"])

    def test_large_batch(self, gauss_layer, small_dims):
        x = torch.randn(1024, small_dims["size_in"])
        out = gauss_layer(x)
        assert out.shape == (1024, small_dims["size_out"])
        assert torch.isfinite(out).all()
