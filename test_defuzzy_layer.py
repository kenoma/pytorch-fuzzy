import pytest
from torchfuzzy import DefuzzyLinearLayer
import torch
import numpy as np

def test_from_array_correct_initialization():
    model = DefuzzyLinearLayer.from_array([
        [1,2],
        [1,2],
        [1,2]])
    assert model.Z.shape == (1, 3, 2)

def test_from_dimensions_correct_initialization():
    model = DefuzzyLinearLayer.from_dimensions(4, 7)
    assert model.Z.shape == (1, 7, 4)
    
def test_1d_input_1d_output():
    batch_size = 1
    size_in = 1
    size_out = 1
    x = torch.randn((batch_size, size_in))
    
    model = DefuzzyLinearLayer.from_dimensions(size_in, size_out)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_1d_input_2d_output():
    batch_size = 1
    size_in = 1
    size_out = 2
    x = torch.randn((batch_size, size_in))
    
    model = DefuzzyLinearLayer.from_dimensions(size_in, size_out)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size, size_out)

def test_1d_input_7d_output():
    batch_size = 1
    size_in = 1
    size_out = 7
    x = torch.randn((batch_size, size_in))
    
    model = DefuzzyLinearLayer.from_dimensions(size_in, size_out)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_2d_input_1d_output():
    batch_size = 1
    size_in = 2
    size_out = 1
    x = torch.randn((batch_size, size_in))
    
    model = DefuzzyLinearLayer.from_dimensions(size_in, size_out)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_2d_input_2d_output():
    batch_size = 1
    size_in = 2
    size_out = 2
    x = torch.randn((batch_size, size_in))
    
    model = DefuzzyLinearLayer.from_dimensions(size_in, size_out)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_7d_input_7d_output():
    batch_size = 1
    size_in = 7
    size_out = 7
    x = torch.randn((batch_size, size_in))
    
    model = DefuzzyLinearLayer.from_dimensions(size_in, size_out)
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size,size_out)

def test_inference_1():
    model = DefuzzyLinearLayer.from_array([
        [1, 2],
        [1, 2],
        [1, 2]])
    x = torch.tensor([[0.0, 1.0]],requires_grad=False)
    
    inference = model.forward(x).detach().numpy()

    assert len(inference) == 1
    assert inference[0] == pytest.approx([2, 2, 2], abs=1e-2)


def test_inference_2():
    model = DefuzzyLinearLayer.from_array([
        [1, 2],
        [1, 2],
        [1, 2]])
    x = torch.tensor([[1.0, 0.0]],requires_grad=False)
    
    inference = model.forward(x).detach().numpy()

    assert len(inference) == 1
    assert inference[0] == pytest.approx([1, 1, 1], abs=1e-2)


def test_inference_3():
    model = DefuzzyLinearLayer.from_array([
        [1, 2],
        [1, 2],
        [1, 2]])
    x = torch.tensor([[1.0, 1.0]],requires_grad=False)
    
    inference = model.forward(x).detach().numpy()

    assert len(inference) == 1
    assert inference[0] == pytest.approx([1.5, 1.5, 1.5], abs=1e-2)


def test_inference_4():
    model = DefuzzyLinearLayer.from_array([
        [1, 2],
        [1, 2],
        [1, 2]])
    x = torch.tensor([[111.0, 111.0]],requires_grad=False)
    
    inference = model.forward(x).detach().numpy()

    assert len(inference) == 1
    assert inference[0] == pytest.approx([1.5, 1.5, 1.5], abs=1e-2)