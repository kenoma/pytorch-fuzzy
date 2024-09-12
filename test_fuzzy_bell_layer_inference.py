import pytest
from torchfuzzy import FuzzyBellLayer, FuzzyLayer
import torch

def test_case_1():
    model = FuzzyBellLayer.from_centers([[0,0]])
    x = torch.FloatTensor([[0,0]])
    y = model(x)
    assert y.shape == (1,1)
    assert y.detach().numpy()[0] == pytest.approx([1], abs=0.1)

def test_case_2():
    model = FuzzyBellLayer.from_centers([[0, 0]])
    x = torch.FloatTensor([[0, 0]])
    y = model(x)
    assert y.shape == (1, 1)
    assert y.detach().numpy()[0] == pytest.approx([1], abs=0.1)

def test_case_3():
    model = FuzzyBellLayer.from_centers([[0, 0]])
    x = torch.FloatTensor([[1, 1]])
    y = model(x)
    assert y.shape == (1, 1)
    assert y.detach().numpy()[0] == pytest.approx([0.5], abs=0.1)

def test_case_4():
    model = FuzzyBellLayer.from_centers([[0, 0]])
    x = torch.FloatTensor([[100, 100]])
    y = model(x)
    assert y.shape == (1, 1)
    assert y.detach().numpy()[0] == pytest.approx([0.001], abs=0.1)

def test_case_5():
    model = FuzzyBellLayer.from_centers([[0,0], [1,1]])
    x = torch.FloatTensor([[0,0],[0,0],[0,0]])
    y = model(x)
    assert y.shape == (3, 2)
    assert y[0].detach().numpy() == pytest.approx([1, 0.41421354], abs=0.1)
    assert y[1].detach().numpy() == pytest.approx([1, 0.41421354], abs=0.1)
    assert y[2].detach().numpy() == pytest.approx([1, 0.41421354], abs=0.1)


def test_case_6():
    model = FuzzyBellLayer.from_centers([[0,0], [1,1]])
    x = torch.FloatTensor([[0,0],[1, 1],[-1, -1]])
    y = model(x)
    assert y.shape == (3, 2)
    assert y[0].detach().numpy() == pytest.approx([1, 0.41421354], abs=0.1)
    assert y[1].detach().numpy() == pytest.approx([0.41421354, 1], abs=0.1)
    assert y[2].detach().numpy() == pytest.approx([ 0.41421354, 0.26120389], abs=0.1)

