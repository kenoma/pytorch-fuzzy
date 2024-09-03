import pytest
from torchfuzzy import FuzzyLayer
import torch

def test_fromcenters_correct_initialization():
    model = FuzzyLayer.from_centers([[1,2,3]])
    assert model.scales.shape == (1, 3)
    assert len(model.rots) == 2
    assert model.rots[0].shape == (1,2)
    assert model.rots[1].shape == (1,1)
    assert model.centroids.shape == (1, 3, 1)

def test_fromdimentions_correct_initialization():
    model = FuzzyLayer.from_dimensions(4, 7)
    assert model.scales.shape == (7, 4)
    assert len(model.rots) == 3
    assert model.rots[0].shape == (7, 3)
    assert model.rots[1].shape == (7, 2)
    assert model.rots[2].shape == (7, 1)
    assert model.centroids.shape == (7, 4, 1)
    
def test_1d_input_1d_output():
    batch_size = 1
    size_in = 1
    size_out = 1
    x = torch.randn((batch_size, size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_1d_input_2d_output():
    batch_size = 1
    size_in = 1
    size_out = 2
    x = torch.randn((batch_size, size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_1d_input_7d_output():
    batch_size = 1
    size_in = 1
    size_out = 7
    x = torch.randn((batch_size, size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_2d_input_1d_output():
    batch_size = 1
    size_in = 2
    size_out = 1
    x = torch.randn((batch_size, size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_2d_input_2d_output():
    batch_size = 1
    size_in = 2
    size_out = 2
    x = torch.randn((batch_size, size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_7d_input_7d_output():
    batch_size = 1
    size_in = 7
    size_out = 7
    x = torch.randn((batch_size, size_in))
    initial_scales = torch.randn((size_out, size_in))
    initial_centers = torch.randn((size_out, size_in))

    model = FuzzyLayer(initial_centers, initial_scales)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

