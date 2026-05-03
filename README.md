# pytorch-fuzzy

Differentiable fuzzy-logic layers for PyTorch — building blocks for **ANFIS**, **Mamdani FIS**, **Takagi–Sugeno FIS** and hybrid neuro‑fuzzy architectures (classifiers, autoencoders, anomaly detectors).

[![PyPI](https://img.shields.io/pypi/v/torchfuzzy.svg)](https://pypi.org/project/torchfuzzy/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange.svg)]()

## Table of contents

- [General](#general)
- [Installation](#installation)
- [Components](#components)
- [Examples](#examples)
- [Tricks](#tricks)
- [Publications](#publications)
- [Citations](#citations)
- [License](#license)

## General

A **fuzzy inference system (FIS)** generally consists of three stages:

1. **Fuzzification** — input vectors are mapped to rule-activation (firing) strengths through membership functions `μ(x)`.
2. **Inference** — firings are combined according to a rule base.
3. **Defuzzification** — firings are collapsed back into a crisp output (scalar, vector, class logits, …).

`torchfuzzy` provides each of these steps as a standard `torch.nn.Module`, so any FIS — ANFIS, Mamdani, Takagi–Sugeno, TSK, custom — can be assembled as an ordinary `nn.Sequential` and trained end-to-end with autograd.

All fuzzification layers share a common base (`FuzzyLayerBase`) that parametrises each term by a **Mahalanobis-like distance** with a full, learnable symmetric positive-definite metric, factorised via **Cholesky** to remain SPD throughout training:

```math
A_j = L_j \quad\text{(lower-triangular)},\qquad
M_j = A_j^{\!\top} A_j \in \mathbb{S}^{n}_{++},
```

```math
r_j^2(x) \;=\; \bigl\lVert A_j (x - c_j) \bigr\rVert^2
         \;=\; (x - c_j)^{\!\top} M_j (x - c_j).
```

Different subclasses of `FuzzyLayerBase` then turn `r_j^2` into a membership value `μ_j(x)` via different nonlinearities (Gaussian, generalized bell, Cauchy, …).

## Installation

```bash
pip install torchfuzzy
```

Requirements: `torch>=1.8`, `numpy`.

## Components

### `FuzzyLayerBase` *(abstract)*

Parent class. Holds learnable centers `c_j ∈ ℝⁿ`, diagonal `diag(A_j)` (`scales`) and the strictly-lower triangle of `A_j` (`rot`). Supplies:

- `input_mask` — per-term feature gate,
- `active_mask` — per-term on/off switch,
- `prune_inactive()` — physically deletes disabled terms,
- `freeze(centers, scales, rot)` — selectively freeze parameters.

Subclasses only need to implement `_membership(rx2)`.

### `FuzzyLayer` — Gaussian terms

```math
\mu_j(x) \;=\; \exp\!\bigl(-\,r_j^2(x)\bigr)
         \;=\; \exp\!\bigl(-\,(x-c_j)^{\!\top} M_j (x-c_j)\bigr).
```

The classic radial-basis / Gaussian membership. Best default choice.

### `FuzzyCauchyLayer` — Cauchy terms

```math
\mu_j(x) \;=\; \frac{1}{1 + r_j^2(x)}.
```

Heavier tails than Gaussian — useful when you want non-vanishing firing far from the cluster center (e.g. to retain gradient signal during early training).

### `FuzzyBellLayer` — Generalized bell terms

```math
\mu_j(x) \;=\; \frac{1}{1 + \bigl(r_j^2(x)\bigr)^{b_j}},\qquad b_j > 0.
```

The classical ANFIS membership function with an extra learnable shape exponent `b_j`. `b_j` can be parametrised as:

- `"raw"` — `b = b_raw` (unconstrained, may go non‑positive);
- `"softplus"` — `b = softplus(b_raw) + ε` (strictly positive, default);
- `"exp"` — `b = exp(b_raw)` (strictly positive, multiplicative).

### `DefuzzyNWLayer` — Nadaraya–Watson defuzzification

```math
y_k(x) \;=\; \frac{\sum_{j} Z_{k j}\,\varphi_j(x)}{\sum_{j} \varphi_j(x)},
\qquad \varphi_j(x) = \text{firing of rule } j,
```

with learnable consequents $Z ∈ ℝ^{\text{out}\times\text{in}}$. Setting `with_norm=False` removes the denominator and the layer reduces to a bias-free linear map (useful for Takagi–Sugeno-style outputs).

### `DefuzzyMaxLayer` — Max defuzzification

```math
y_k(x) \;=\; \max_{j}\,\bigl(Z_{k j}\,\varphi_j(x)\bigr).
```

Non-smooth, winner-takes-all style. Good fit for classification heads with one rule per class.

---

## Examples

### Minimal fuzzification

```python
import torch
from torchfuzzy import FuzzyLayer

x = torch.rand(10, 2)
layer = FuzzyLayer.from_dimensions(size_in=2, size_out=4)
firings = layer(x)                       # (10, 4)
```

### Mamdani-style FIS

Mamdani rule base:

```math
\text{Rule}_j:\ \mathbf{IF}\ x\ \text{is}\ A_j\ \mathbf{THEN}\ y = Z_j,
```

```math
y(x) \;=\; \frac{\sum_{j} Z_j\,\mu_{A_j}(x)}{\sum_{j} \mu_{A_j}(x)}.
```

```python
import torch.nn as nn
from torchfuzzy import FuzzyLayer, DefuzzyNWLayer

mamdani_fis = nn.Sequential(
    FuzzyLayer.from_dimensions(input_dim, n_rules, trainable=True),
    DefuzzyNWLayer.from_dimensions(n_rules, output_dim, with_norm=True),
)
```

### ANFIS (Takagi–Sugeno of order 0/1) with bell MFs

```python
import torch
import torch.nn as nn
from torchfuzzy import FuzzyBellLayer, DefuzzyNWLayer

class ANFIS(nn.Module):
    def __init__(self, n_in, n_rules, n_out):
        super().__init__()
        self.fuzzify = FuzzyBellLayer.from_dimensions(
            n_in, n_rules, b_parametrization="softplus"
        )
        self.defuzz = DefuzzyNWLayer.from_dimensions(
            n_rules, n_out, with_norm=True   # Nadaraya–Watson normalization
        )

    def forward(self, x):
        firings = self.fuzzify(x)            # (B, n_rules)
        return self.defuzz(firings)          # (B, n_out)

model = ANFIS(n_in=4, n_rules=16, n_out=1)
y = model(torch.randn(32, 4))
```

### Takagi–Sugeno with linear consequents

For a first-order TS system replace the constant consequent by a per-rule linear function of the inputs:

```python
import torch
import torch.nn as nn
from torchfuzzy import FuzzyLayer

class TSK1(nn.Module):
    def __init__(self, n_in, n_rules, n_out):
        super().__init__()
        self.fuzzify    = FuzzyLayer.from_dimensions(n_in, n_rules)
        self.consequent = nn.Linear(n_in, n_rules * n_out)
        self.n_rules, self.n_out = n_rules, n_out

    def forward(self, x):
        w = self.fuzzify(x)                                  # (B, R)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        y = self.consequent(x).view(-1, self.n_rules, self.n_out)
        return (w.unsqueeze(-1) * y).sum(dim=1)              # (B, n_out)
```

### Max-defuzz classifier

```python
from torchfuzzy import FuzzyCauchyLayer, DefuzzyMaxLayer

classifier = nn.Sequential(
    FuzzyCauchyLayer.from_dimensions(n_features, n_rules),
    DefuzzyMaxLayer.from_dimensions(n_rules, n_classes),
)
```

### Fuzzy head on top of a neural encoder

```python
encoder = torchvision.models.resnet18(num_classes=32)
head = nn.Sequential(
    FuzzyLayer.from_dimensions(32, n_rules),
    DefuzzyNWLayer.from_dimensions(n_rules, n_classes),
)
model = nn.Sequential(encoder, head)
```

---

## Tricks

### Masking inactive terms (soft disable)

`active_mask` is a buffer of shape `(size_out,)`. Zeroing out an entry forces the corresponding firing to zero *without* removing the parameters:

```python
layer = FuzzyLayer.from_dimensions(size_in=4, size_out=8)
x = torch.randn(16, 4)

layer.set_active([1, 3, 5], active=False)
assert layer.n_active == 5

out = layer(x)
assert (out[:, [1, 3, 5]] == 0).all()

layer.reset_mask()                          # re-enable all
```

### Pruning dead terms (hard delete)

After soft-disabling rules you can physically shrink the layer. All parameters (`centers`, `scales`, `rot`, and, for `FuzzyBellLayer`, `_b_raw`) are resliced in-place:

```python
with torch.no_grad():
    mean_act = layer(x_val).mean(dim=0)                    # (size_out,)
    dead = (mean_act < 1e-3).nonzero().view(-1).tolist()
    if dead:
        layer.set_active(dead, active=False)
        layer.prune_inactive()

print(layer.size_out, layer.centers.shape)
```

Typical uses:
- removing rules that never fire,
- post-training compression,
- knowledge-distillation / structured pruning schedules.

### Per-term input masking

Each term can be restricted to a subset of input features via `input_mask` of shape `(size_out, size_in)`:

```python
# Rule 0 sees features {0,1}; rule 1 — {2,3}; rule 2 — {0,2}.
mask = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
], dtype=torch.float32)

layer = FuzzyLayer.from_centers(
    torch.randn(3, 4),
    input_mask=mask,
)
```

This is equivalent to forcing the corresponding rows of each `A_j` to zero, and is useful for interpretable rule bases and for feature-selection fuzzy models.

### Freezing / unfreezing parameters

```python
layer = FuzzyBellLayer.from_dimensions(8, 16)

# Freeze centers and bell exponents, keep scales/rot trainable
layer.freeze(centers=True, powers=True)

# Fine-grained control
layer.set_requires_grad_centroids(False)
layer.set_requires_grad_scales(True)
layer.set_requires_grad_rot(True)
```

### Inspecting learned geometry

```python
A      = layer.get_transformation_matrix()   # (size_out, n, n) Cholesky factor
M      = layer.get_metric_matrix()           # (size_out, n, n) SPD metric
eig    = layer.get_transformation_matrix_eigenvals()   # all > 0
centers = layer.get_centroids()
```

---

## Publications

- [Variational Autoencoders with Fuzzy Inference (Russian)](https://habr.com/ru/articles/803789/)
- [Conditional Variational Autoencoders with Fuzzy Inference](https://doi.org/10.1007/978-3-031-77411-9_9)

## Citations

If you use `torchfuzzy` in academic work, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-77411-9_9,
    author    = "Gurov, Yury and Khilkov, Danil",
    editor    = "Kovalev, Sergey and Kotenko, Igor and Sukhanov, Andrey and Li, Yin and Li, Yao",
    title     = "Conditional Variational Autoencoders with Fuzzy Inference",
    booktitle = "Proceedings of the Eighth International Scientific Conference 'Intelligent Information Technologies for Industry' (IITI'24), Volume 2",
    year      = "2024",
    publisher = "Springer Nature Switzerland",
    address   = "Cham",
    pages     = "91--103",
    isbn      = "978-3-031-77411-9"
}
```

Related work worth citing when using specific components:

```bibtex
@article{jang1993anfis,
    title   = {ANFIS: adaptive-network-based fuzzy inference system},
    author  = {Jang, Jyh-Shing Roger},
    journal = {IEEE Transactions on Systems, Man, and Cybernetics},
    volume  = {23}, number = {3}, pages = {665--685}, year = {1993}
}

@article{takagi1985fuzzy,
    title   = {Fuzzy identification of systems and its applications to modeling and control},
    author  = {Takagi, Tomohiro and Sugeno, Michio},
    journal = {IEEE Transactions on Systems, Man, and Cybernetics},
    volume  = {SMC-15}, number = {1}, pages = {116--132}, year = {1985}
}
```

## License

MIT (see `LICENSE`).
