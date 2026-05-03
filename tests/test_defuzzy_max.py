import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import torch
from torchfuzzy import DefuzzyMaxLayer


@pytest.fixture
def simple_Z():
    # size_out=2, size_in=3
    return torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])


@pytest.fixture
def simple_x():
    # batch=2, size_in=3
    return torch.tensor([[0.1, 0.9, 0.2],
                         [0.7, 0.2, 0.1]])


def test_init_shapes(simple_Z):
    layer = DefuzzyMaxLayer(simple_Z)
    assert layer.size_out == 2
    assert layer.size_in == 3
    assert layer.Z.shape == (2, 3)
    assert layer.Z.requires_grad is True


def test_init_dtype_float_and_clone():
    src = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    layer = DefuzzyMaxLayer(src)
    assert layer.Z.dtype == torch.float32
    
    src[0, 0] = 999
    assert layer.Z.data[0, 0].item() == 1.0


def test_init_validation_ndim():
    with pytest.raises(ValueError):
        DefuzzyMaxLayer(torch.rand(3))
    with pytest.raises(ValueError):
        DefuzzyMaxLayer(torch.rand(2, 3, 4))


def test_init_validation_tau():
    with pytest.raises(ValueError):
        DefuzzyMaxLayer(torch.rand(2, 3), tau=0.0)
    with pytest.raises(ValueError):
        DefuzzyMaxLayer(torch.rand(2, 3), tau=-1.0)


def test_forward_input_validation(simple_Z):
    layer = DefuzzyMaxLayer(simple_Z)
    with pytest.raises(ValueError):
        layer(torch.rand(3))                 
    with pytest.raises(ValueError):
        layer(torch.rand(4, 5))              


def test_from_dimensions():
    layer = DefuzzyMaxLayer.from_dimensions(size_in=5, size_out=4)
    assert layer.Z.shape == (4, 5)
    out = layer(torch.rand(3, 5))
    assert out.shape == (3, 4)


def test_from_array_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer = DefuzzyMaxLayer.from_array(arr)
    assert layer.Z.shape == (2, 2)
    assert layer.Z.dtype == torch.float32


def test_from_array_list():
    layer = DefuzzyMaxLayer.from_array([[1, 2, 3], [4, 5, 6]])
    assert layer.Z.shape == (2, 3)


def test_forward_elementwise_values(simple_Z, simple_x):
    layer = DefuzzyMaxLayer(simple_Z, winner_take_all=False)
    out = layer(simple_x)

    # out[b, j] = max_i Z[j, i] * x[b, i]
    expected = torch.empty(2, 2)
    for b in range(2):
        for j in range(2):
            expected[b, j] = (simple_Z[j] * simple_x[b]).max()
    torch.testing.assert_close(out, expected)


def test_forward_elementwise_shape_larger():
    layer = DefuzzyMaxLayer.from_dimensions(size_in=7, size_out=5)
    out = layer(torch.rand(11, 7))
    assert out.shape == (11, 5)


def test_elementwise_gradient_flows(simple_Z, simple_x):
    layer = DefuzzyMaxLayer(simple_Z.clone(), winner_take_all=False)
    x = simple_x.clone().requires_grad_(True)
    out = layer(x)
    out.sum().backward()
    assert layer.Z.grad is not None
    assert x.grad is not None
    assert torch.isfinite(layer.Z.grad).all()
    assert torch.isfinite(x.grad).all()


def test_winner_forward_matches_hard_argmax(simple_Z, simple_x):
    layer = DefuzzyMaxLayer(simple_Z, winner_take_all=True)
    out = layer(simple_x)

    idx = simple_x.argmax(dim=-1)                 # (B,)
    expected = simple_Z.t()[idx]                  # (B, size_out)
    torch.testing.assert_close(out, expected)


def test_winner_forward_independent_of_tau(simple_Z, simple_x):
    o1 = DefuzzyMaxLayer(simple_Z, winner_take_all=True, tau=0.1)(simple_x)
    o2 = DefuzzyMaxLayer(simple_Z, winner_take_all=True, tau=10.0)(simple_x)
    torch.testing.assert_close(o1, o2)


def test_winner_gradient_flows_to_x_via_ste(simple_Z, simple_x):
    """
    Ключевая проверка STE: обычный argmax недифференцируем, а здесь градиент
    по x должен быть ненулевым и соответствовать градиенту soft-варианта.
    """
    layer = DefuzzyMaxLayer(simple_Z.clone(), winner_take_all=True, tau=1.0)
    x = simple_x.clone().requires_grad_(True)
    out = layer(x)
    out.sum().backward()

    assert x.grad is not None
    # grad не должен быть нулевым на всех элементах
    assert x.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_winner_ste_grad_equals_soft_grad(simple_Z, simple_x):
    """
    STE устроен так, что d out / d x = d (soft @ Z.T) / d x.
    Проверим численно: градиенты должны совпадать с soft-версией.
    """
    Zp = simple_Z.clone()

    # STE-ветка
    layer = DefuzzyMaxLayer(Zp.clone(), winner_take_all=True, tau=0.5)
    x1 = simple_x.clone().requires_grad_(True)
    layer(x1).sum().backward()

    # Эквивалентный soft-расчёт вручную
    x2 = simple_x.clone().requires_grad_(True)
    soft = torch.softmax(x2 / 0.5, dim=-1)
    out_soft = soft @ Zp.t()
    out_soft.sum().backward()

    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)


def test_winner_gradient_flows_to_Z(simple_Z, simple_x):
    layer = DefuzzyMaxLayer(simple_Z.clone(), winner_take_all=True)
    out = layer(simple_x)
    out.sum().backward()
    assert layer.Z.grad is not None
    assert layer.Z.grad.abs().sum() > 0


def test_winner_one_hot_weights_sum_to_one_in_forward(simple_x):
    """
    Косвенно: forward-output в winner-режиме == Z[:, argmax(x)] ⇒
    в forward-ветке эффективные веса — one-hot.
    Проверяем через равенство с ручным one-hot умножением.
    """
    Z = torch.randn(4, 6)
    layer = DefuzzyMaxLayer(Z, winner_take_all=True)

    x = torch.randn(8, 6)
    out = layer(x)

    idx = x.argmax(dim=-1)
    hard = torch.zeros_like(x).scatter_(-1, idx.unsqueeze(-1), 1.0)
    expected = hard @ Z.t()
    torch.testing.assert_close(out, expected)


def test_trainable_false_freezes_Z(simple_Z):
    layer = DefuzzyMaxLayer(simple_Z, trainable=False)
    assert layer.Z.requires_grad is False

    # даём градиент входу, чтобы граф вычислений существовал
    x = torch.rand(2, 3, requires_grad=True)
    out = layer(x)
    out.sum().backward()

    # градиент по входу посчитался
    assert x.grad is not None
    # а по замороженному Z — нет
    assert layer.Z.grad is None


def test_extra_repr_contains_key_fields(simple_Z):
    layer = DefuzzyMaxLayer(simple_Z, winner_take_all=True, tau=0.7)
    r = layer.extra_repr()
    assert "size_in=3" in r
    assert "size_out=2" in r
    assert "winner_take_all=True" in r
    assert "tau=0.7" in r


def test_training_loop_runs_end_to_end():
    """Мини-цикл обучения: loss должен уменьшиться."""
    torch.manual_seed(0)
    layer = DefuzzyMaxLayer.from_dimensions(size_in=4, size_out=2,
                                            winner_take_all=True, tau=1.0)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)

    x = torch.rand(32, 4)
    target = torch.rand(32, 2)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
