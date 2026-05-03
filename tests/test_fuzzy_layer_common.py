import pytest
from torchfuzzy import FuzzyLayer
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

