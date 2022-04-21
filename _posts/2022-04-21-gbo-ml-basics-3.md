---
layout: post
title: "GBO notes: Machine learning basics (Part 3)"
tags: ["gbo", "wfst"]
mathjax: true
---

In this series of notes we will review some basic concepts that are usually covered in an Intro to ML
course. These are based on [this course](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) from Cornell.

In Part 3, we will look at SVMs, empirical risk minimization, model selection, and the bias-variance
tradeoff.

## Support Vector Machine (SVM)

SVMs are an extension of the perceptron model discussed earlier. They are again a linear
classifier, so we define the hypothesis class as $h(\mathbf{x}) = \mathrm{sign}(\mathbf{w}^T\mathbf{x} + b)$,
where we can again combine the bias term by adding a constant dimension to the input.

The difference between perceptron and SVM is that while perceptron returns *any* hyperplane
separating the classes, SVM returns the hyperplane with the *maximum margin*. Let us first define
the margin: it is the minimum distance from hyperplane $\mathcal{H}$ to any point in $D$, i.e.,

$$ \gamma(\mathbf{w}, b) = \min_{\mathbf{x}\in D} \frac{|\mathbf{w}^T\mathbf{x} + b|}{\Vert \mathbf{w} \Vert^2}. $$

Such a hyperplane can be computing by maximizing $\gamma(\mathbf{w},b)$ under the constraint
that all points must lie on the correct side. By playing around with constraints a bit, we can show
that this is equivalent to solving

$$ \min_{\mathbf{w},b} \mathbf{w}^T\mathbf{w} \quad \mathrm{s.t.} \quad \forall i, y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq 1. $$

The points for which the constraint is satisfied with equality are called the "support vectors".
The hyperplane only changes if we change the support vectors. Later, we will look at the dual
formulation of SVMs, which makes use of support vectors.

If the data is not linearly separable, we can add slack variables to allow points to be
misclassified a little. We can also write a loss formulation for SVMs that encompasses the
constraints into the loss function (in the form of a hinge loss):

$$ \min _{\mathbf{w}, b} \underbrace{\mathbf{w}^{T} \mathbf{w}}_{l_{2}-\text { regularizer }}+C \sum_{i=1}^{n} \underbrace{\max \left[1-y_{i}\left(\mathbf{w}^{T} \mathbf{x}+b\right), 0\right]}_{\text {hinge-loss }}. $$

We can see that if the constraint is satisfied, the hinge loss term is 0, otherwise it is equal
to the "slack" times the constant $C$. With this unconstrained loss function, we can optimize
SVM parameters using gradient descent.

## Empirical Risk Minimization (ERM)

It means minimizing some continuous loss function $l$ with a regularizer $r$.

**Binary classification losses:**

1. Hinge loss: used in SVMs
2. Log-loss: used in logistic regression

**Regression losses:**

1. Mean squared error: sensitive to outliers
2. Absolute error: not differentiable at 0
3. Log-cosh: best of both worlds

**Regularizers:** L1, L2, Lp norms.

## Model selection and bias-variance tradeoff

Overfitting and underfitting are equivalent to high variance and high bias, respectively.
Usually we need to select a regularization coefficient which avoids both high bias and
high variance.

*What to do when we have high variance (overfitting)?*

- Add more training data
- Reduce model complexity

*What to do when we have high bias (underfitting)?*

- Add more features
- Increase model complexity (e.g., non-linear models)



