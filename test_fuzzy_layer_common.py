import pytest
from fuzzy_layer import FuzzyLayer
import torch

def test_fromcenters_correct_initialization():
    model = FuzzyLayer.fromcenters([[1,2,3]])
    assert model.A.shape == (1,3,4)

def test_fromdimentions_correct_initialization():
    model = FuzzyLayer.fromdimentions(4,7)
    assert model.A.shape == (7, 4, 5)
    
def test_1d_input_1d_output():
    size_in = 1
    size_out = 1
    x = torch.randn((size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert len(y_pred) == size_out

def test_1d_input_2d_output():
    size_in = 1
    size_out = 2
    x = torch.randn((size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert len(y_pred) == size_out

def test_1d_input_7d_output():
    size_in = 1
    size_out = 7
    x = torch.randn((size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert len(y_pred) == size_out

def test_2d_input_1d_output():
    size_in = 2
    size_out = 1
    x = torch.randn((size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert len(y_pred) == size_out

def test_2d_input_2d_output():
    size_in = 2
    size_out = 2
    x = torch.randn((size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert len(y_pred) == size_out

def test_7d_input_7d_output():
    size_in = 7
    size_out = 7
    x = torch.randn((size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert len(y_pred) == size_out

