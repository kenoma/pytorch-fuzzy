import pytest
from fuzzy_layer import FuzzyLayer
import torch

def test_case_1():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([1,1])
    y = model(x)
    assert y.detach().numpy() == pytest.approx([1,0,0,0], abs=0.1)

def test_case_2():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([10,10])
    y = model(x)
    assert y.detach().numpy() == pytest.approx([0,1,0,0], abs=0.1)

def test_case_3():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([1,10])
    y = model(x)
    assert y.detach().numpy() == pytest.approx([0,0,1,0], abs=0.1)

def test_case_4():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([10,1])
    y = model(x)
    assert y.detach().numpy() == pytest.approx([0,0,0,1], abs=0.1)

def test_case_5():
    model = FuzzyLayer.fromcenters([[1,1], [2,2], [1,2], [2,1]])
    x = torch.FloatTensor([1.5,1.5])
    y = model(x)
    assert y.detach().numpy() == pytest.approx([0.5,0.5,0.5,0.5], abs=1e-2)
