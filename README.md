# pytorch-fuzzy
Experiments with fuzzy layers and neural nerworks

## Goals

 - Get more fine-grained features from autoencoders
 - Semi-supervised learning
 - Anomaly detections

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

![image](https://user-images.githubusercontent.com/6205671/211149392-3563ae02-c13b-4bef-b35c-ce89a1fe46e2.png)


After training procedure completed (full code see [here](experiments_simple_clustering.py)) and correct points labeling is achieved uniform distribution classification performed. On picture below yellow points are not passed through threshold of any centroid belonginess.

![image](https://user-images.githubusercontent.com/6205671/211149065-b72b1e11-a538-479b-813a-df4e06ab115c.png)

On this primitive example we can see that `FuzzyLayer` is able to learn clustered structure of underlying manifold.
In such a way `FuzzyLayer` can be used as anomaly detection algorithm if we interpret yellow points as outliers. 
But more interesting application of `FuzzyLayer` is clusterization of another model outputs to get more fine-grained results.



