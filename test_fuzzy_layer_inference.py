import pytest
from fuzzy_layer import FuzzyLayer
import torch

def test_case_1():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([[1,1]])
    y = model(x)
    assert y.shape == (1,4)
    assert y.detach().numpy()[0] == pytest.approx([1,0,0,0], abs=0.1)

def test_case_2():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([[10,10]])
    y = model(x)
    assert y.detach().numpy()[0] == pytest.approx([0,1,0,0], abs=0.1)

def test_case_3():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([[1,10]])
    y = model(x)
    assert y.detach().numpy()[0] == pytest.approx([0,0,1,0], abs=0.1)

def test_case_4():
    model = FuzzyLayer.fromcenters([[1,1], [10,10], [1,10], [10,1]])
    x = torch.FloatTensor([[10,1]])
    y = model(x)
    assert y.detach().numpy()[0] == pytest.approx([0,0,0,1], abs=0.1)

def test_case_5():
    model = FuzzyLayer.fromcenters([[1,1], [2,2], [1,2], [2,1]])
    x = torch.FloatTensor([[1.5,1.5]])
    y = model(x)
    assert y.detach().numpy()[0] == pytest.approx([0.5,0.5,0.5,0.5], abs=1e-2)

def test_case_5_inputs_shape():
    model = FuzzyLayer.fromcenters([[1,1], [2,2], [1,2], [2,1]])
    x = torch.FloatTensor([[1,1], [2,2], [1,2], [2,1], [1.5,1.5]])
    y = model(x)
    ny = y.detach().numpy()

    assert ny.shape == (5,4)

def test_case_5_inputs_p0():
    model = FuzzyLayer.fromcenters([[1,1], [20,20], [1,20], [20,1]])
    x = torch.FloatTensor([[1,1], [20,20], [1,20], [20,1]])
    y = model(x)
    ny = y.detach().numpy()

    assert ny[0] == pytest.approx([1,0,0,0], abs=1e-2)

def test_case_5_inputs_p1():
    model = FuzzyLayer.fromcenters([[1,1], [20,20], [1,20], [20,1]])
    x = torch.FloatTensor([[1,1], [20,20], [1,20], [20,1]])
    y = model(x)
    ny = y.detach().numpy()

    assert ny[1] == pytest.approx([0,1,0,0], abs=1e-2)

def test_case_5_inputs_p2():
    model = FuzzyLayer.fromcenters([[1,1], [20,20], [1,20], [20,1]])
    x = torch.FloatTensor([[1,1], [20,20], [1,20], [20,1]])
    y = model(x)
    ny = y.detach().numpy()

    assert ny[2] == pytest.approx([0,0,1,0], abs=1e-2)
    
def test_case_5_inputs_p3():
    model = FuzzyLayer.fromcenters([[1,1], [20,20], [1,20], [20,1]])
    x = torch.FloatTensor([[1,1], [20,20], [1,20], [20,1]])
    y = model(x)
    ny = y.detach().numpy()

    assert ny[3] == pytest.approx([0,0,0,1], abs=1e-2)
    
def test_case_5_inputs_p4():
    model = FuzzyLayer.fromcenters([[1,1], [2,2], [1,2], [2,1]])
    x = torch.FloatTensor([[1,1], [2,2], [1,2], [2,1], [1.5,1.5]])
    y = model(x)
    ny = y.detach().numpy()

    assert ny[4] == pytest.approx([0.5,0.5,0.5,0.5], abs=1e-2)

