# pytorch-fuzzy

Experiments with fuzzy layers and neural nerworks

## Goals

- Get more fine-grained features from autoencoders
- Semi-supervised learning
- Anomaly detections

## Installation

Package requirements:

`
torch>=1.8
`

Installation via pip:

```python
 pip install torchfuzzy
```

## Fuzzy layer

Membership function for layer `FuzzyLayer` have form $\mu(x, A) = e^{ -|| \[A . \~x\]_{1 \cdots m} ||^2}$ where $m$ is task dimension,  $A$ is [transformation matrix](https://en.wikipedia.org/wiki/Transformation_matrix) in form 

```math
A_{(m+1) \times (m+1)} =
  \left[ {\begin{array}{cccc}
    s_{1} & a_{12} & \cdots & a_{1m} & c_{1}\\
    a_{21} & s_{2} & \cdots & a_{2m} & c_{2}\\
    \vdots & \vdots & \ddots & \vdots & c_{3}\\
    a_{m1} & a_{m2} & \cdots & s_{m} & c_{m}\\
    0 & 0 & \cdots & 0 & 1\\
  \end{array} } \right]

```

with $c_{1\cdots m}$ - centroid, 
$s_{1\cdots m}$ - scaling factor, 
$a_{1\cdots m, 1\cdots m}$ - alignment coefficients and 
$x$ is an extended with $1$ vector 
$x = [x_1, x_2, \cdots, x_m, 1]$.

`FuzzyLayer` stores and tunes set of matricies $A^{n}, n = 1 \dots N$ where $N$ is layer's output dimension.

## How it works

Let's demonstrate how `FuzzyLayer` works on simple 2D case generating dataset with four centroids. 
This dataset consists of 2D point coordinates and centroid belongingness as label.
To each coordinate scaled noise component is added.
Resulting clustered structures are shown on picture below. 

![image](https://user-images.githubusercontent.com/6205671/211149471-9d850748-f40b-4acc-8250-331b5594ffe0.png)


After training procedure completed (full code see [here](experiments_simple_clustering.py)) and correct points labeling is achieved uniform distribution classification performed. On picture below yellow points are not passed through threshold of any centroid belonginess.

![image](https://user-images.githubusercontent.com/6205671/211149065-b72b1e11-a538-479b-813a-df4e06ab115c.png)
![image](https://user-images.githubusercontent.com/6205671/214388927-6e70dcf1-2323-4ac9-8589-144e96a6375d.png)

On this primitive example we can see that `FuzzyLayer` is able to learn clustered structure of underlying manifold.
In such a way `FuzzyLayer` can be used as anomaly detection algorithm if we interpret yellow points as outliers. 
But more interesting application of `FuzzyLayer` is clusterization of another model outputs to get more fine-grained results.

## Usage

### Basic

```python
from torchfuzzy import FuzzyLayer

x = torch.rand((10,2))

fuzzy_layer = FuzzyLayer.from_dimensions(2, 4)

inference = fuzzy_layer.forward(x)

```

### Mamdani-like inference

Full example [see here](experiments_mamdani_mnist.ipynb).

Mamdani fuzzy model can be represented as a ruleset:

```math
    \begin{array}{lcll}
        \text{Rule}_{i-1} & : &\mathbf{IF}\ x\; is\; A_{i-1}\ &\mathbf{THEN}\ y_{i-1},\\
        \text{Rule}_{i}   & : &\mathbf{IF}\ x\; is\; A_{i  }\ &\mathbf{THEN}\ y_{i},\\
        \text{Rule}_{i+1} & : &\mathbf{IF}\ x\; is\; A_{i+1}\ &\mathbf{THEN}\ y_{i+1},\\
    \end{array}
```

where $y_{i}$ is an scalar. Mamdani inference is denoted as:

```math
Output = \frac{\sum \mu(x, A_{i})*y_{i}}{\sum \mu(x, A_{i})}
```

Straightforward implementation with `FuzzyLayer`:

```python
mamdani_fis = nn.Sequential(
    FuzzyLayer.from_dimensions(input_dimention, fuzzy_rules_count, trainable=True),
    nn.Softmax(1),
    nn.Linear(fuzzy_rules_count, output_dimention, bias=False)
    )
```

A more correct implementation is implemented in the `DefuzzyLinearLayer`, the network structure takes the following form

```python
mamdani_fis = nn.Sequential(
            FuzzyLayer.from_dimensions(input_dimention, fuzzy_rules_count, trainable=True),
            DefuzzyLinearLayer.from_dimensions(fuzzy_rules_count, output_dimention)
        )
```

## Publications

[Variational Autoencoders with Fuzzy Inference (Russian)](https://habr.com/ru/articles/803789/)
