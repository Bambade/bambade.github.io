---
layout: post
title: "GBO notes: Machine learning basics (Part 4)"
tags: ["gbo", "ml-basics"]
mathjax: true
---

In this series of notes we will review some basic concepts that are usually covered in an Intro to ML
course. These are based on [this course](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) from Cornell.

In Part 4, we will look at kernels, including kernel SVMs, and Gaussian processes.

## Kernels

How can we use linear classifiers (like perceptron, SVM, logistic regression, etc.) when
the data is not linearly separable? One way might be to increase the data dimensionality
by adding features that are non-linear combinations of other features. But the problem with
this is that the data may become too high-dimensional, which would make the learning
algorithm very slow.

This is where the kernel trick comes in. Consider learning a linear classifier with gradient
descent. Since the updates at each step are just linear combinations of the inputs, we can
show by induction that $\mathbf{w}$ would also be a linear combination of all $\mathbf{x}_i$'s.
As such, the loss and label computation can be written entirely in terms of inner products
of the inputs.

A kernel function is defined as the product of the high-dimensional transformation of two
vectors:

$$ k(\mathbf{x}_i,\mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j). $$

Usually, all pairwise kernels are pre-computed and stored in a kernel matrix $K$. If we
have this matrix, we can do gradient descent in high-dimensions by simple matrix lookup.

So what exactly is a kernel function. A kernel function *must* correspond to real inner-products
after some transformation $\phi$. This is the case if and only if $K$ is positive
semi-definite (easy to prove).

Some popular kernels:

1. Linear: $K(\mathbf{x},\mathbf{z}) = \mathbf{x}^T\mathbf{z}$
2. Polynomial: $K(\mathbf{x},\mathbf{z}) = (1+\mathbf{x}^T\mathbf{z})^d$
3. Gaussian: $K(\mathbf{x},\mathbf{z}) = e^{-\frac{\Vert\mathbf{x}-\mathbf{z}\Vert^2}{\sigma^2}}$
4. Exponential: $K(\mathbf{x},\mathbf{z}) = e^{-\frac{\Vert\mathbf{x}-\mathbf{z}\Vert}{2\sigma^2}}$
5. Laplacian: $K(\mathbf{x},\mathbf{z}) = e^{-\frac{\Vert\mathbf{x}-\mathbf{z}\Vert}{\sigma}}$
6. Sigmoid: $K(\mathbf{x},\mathbf{z}) = \mathrm{tanh}(a\mathbf{x}^T\mathbf{z}+c)$

In general, we can combine these simple kernels using a set of rules (such as addition, scalar
multiplication, function transform, etc.) to get well-defined kernels.

**How to kernelize an algorithm?**

1. Prove that the solution is spanned by the input vectors.
2. Rewrite updates/algorithm to only use dot products of inputs.
3. Define some kernel function $k$ and substitute dot products with $k(\mathbf{x}_i,\mathbf{x}_j)$.

Let us see how to apply this idea to linear regression and SVMs.

### Kernelized linear regression

We want to express $\mathbf{w}$ as $\mathbf{X}\mathbf{\alpha}$. Starting from the solution
obtained earlier, we get

$$
\begin{aligned}
\mathbf{w} = \mathbf{X\alpha} &= (\mathbf{X}\mathbf{X}^T)^{-1} \mathbf{X}\mathbf{y}^T \\
    (\mathbf{X}\mathbf{X}^T)\mathbf{X\alpha} &= (\mathbf{X}\mathbf{X}^T)(\mathbf{X}\mathbf{X}^T)^{-1} \mathbf{X}\mathbf{y}^T \\
    \mathbf{X}^T(\mathbf{X}\mathbf{X}^T)\mathbf{X\alpha} &= \mathbf{X}^T\mathbf{X}\mathbf{y}^T \\
    \mathbf{K}^2 \mathbf{\alpha} &= \mathbf{Ky} \\
    \mathbf{\alpha} = \mathbf{K}^{-1}\mathbf{y},
\end{aligned}
$$

where we have defined $\mathbf{K} = \mathbf{X}^T\mathbf{X}$.

### Kernel SVMs

Based on primal-dual optimization methods, the original SVM formulation can be written
as dual form where the constraints become the parameters. Recall that in the primal problem,
the parameter was the weight vector $\mathbf{w}$ and the constraints were over the data
points. We can write the dual formulation as

$$ \begin{array}{ll} 
& \min _{\alpha_{1}, \cdots, \alpha_{n}} \frac{1}{2} \sum_{i, j} \alpha_{i} \alpha_{j} y_{i} y_{j} K_{i j}-\sum_{i=1}^{n} \alpha_{i} \\
\text { s.t. } & 0 \leq \alpha_{i} \leq C \\
& \sum_{i=1}^{n} \alpha_{i} y_{i}=0
\end{array} $$

In some ways, kernel SVM can be thought of as a "soft" version of the k-NN algorithm, where instead
of considering only the $k$ nearest points, it considers all points but assigns more weight
to the closer points.

## Gaussian processes

A Gaussian random variable is characterized by the following distribution

$$ P(x;\mu,\Sigma) = \frac{1}{(2\pi)^{\frac{d}{2}}|\Sigma|} e^{-\frac{1}{2}((x-\mu)^T \Sigma (x-\mu))}. $$

It often occurs in the real world because of the Central Limit Theorem, which states that
sum of independent random variables tends to be close to a normal distribution.

Recall that in linear regression, we first made the assumption that the data lies approximately
on a straight line with some normally-distributed error, and then used either MLE or MAP
to solve for $\mathbf{w}$. Then at test time, we computed $y_i = \mathbf{w}^T \mathbf{x}_i$
to get the result.

But this is a frequentist approach, i.e., we are committing ourselves to a single $\mathbf{w}$.
If we think in a more Bayesian way, we can try to avoid every computing such a $\mathbf{w}$.
Instead, let us model the probability distribution for all the labels as follows:

$$
P(Y|\mathbf{X},D) = \int_{\mathbf{w}} P(Y|\mathbf{X},\mathbf{w}) P(\mathbf{w}|D) \partial \mathbf{w}, 
$$

Here, both the terms within the integral are Gaussian (if we assume $\mathbf{w}$ has a Gaussian prior),
and so the resulting distribution will also be Gaussian. So at test time, given input $\mathbf{x}$,
we can write

$$ P(y_{\ast}|D,\mathbf{x}) \sim \mathcal{N}(\mu,\Sigma), $$

where 

$$ \mu = K_{\ast}^T (K+\sigma^2 I)^{-1}y, $$

and

$$ \Sigma = K_{\ast\ast} K_{\ast}^T (K+\sigma^2 I)^{-1}K_{\ast}, $$

where $K$ is a kernel function.