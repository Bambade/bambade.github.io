---
layout: post
title: "GBO notes: Machine learning basics (Part 1)"
tags: ["gbo", "ml-basics"]
mathjax: true
---

In this series of notes we will review some basic concepts that are usually covered in an Intro to ML
course. These are based on [this course](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) from Cornell.

In Part 1, we will look at the basic problem of supervised learning, simple classifiers such
as k-NN and perceptron, and MLE and MAP estimators.

## Machine learning setup

In supervised learning, we are given a dataset $D = \{(\mathbf{x}_1,y_1),\ldots,(\mathbf{x}_n,y_n)\} \in \mathbb{R}^d\times \mathcal{C}$,
and the goal is to learn a function $h$ such that $h(\mathbf{x}) = y$ with high probability.
Here $\mathbf{x}_i,y_i)$ are sampled from some unknown distribution.

We usually have to make some assumption about the type of function $h$ so that learning is
possible. This is called the *no free lunch theorem*.

Once we have selected the hypothesis class, we find the best function $h$ within the class
by solving some kind of optimization problem. The objective of this optimization is to 
minimize some notion of *loss* on the training data.

To ensure that we are generalizing and not just memorizing the training data, we keep a 
held-out validation set on which we periodically measure the loss while training. If the test
data is large enough, it is an unbiased estimator of the model performance, because of the
weak law of large numbers.

## k-nearest neighbors

It is a non-parametric classification method. The idea is that given a test input $\mathbf{x}$,
return the most common label among its $k$ nearest neighbors ($k$ most similar training inputs).
Small values of $k$ will return noisy result, and very large values will increase computation.
Usually, the $L_p$-norm is used as the distance metric to compute the $k$-nearest neighbors.

**Theorem:** For large $n$, the 1-NN error is no more than twice the error of a Bayes optimal
classifier.

The proof uses the fact that for large $n$, the nearest neighbor is the same point as the test
point. Unfortunately, in high-dimensional setting, this is never true, since two points in
high-dimensional space are far apart.

However, if we consider a **hyperplane** in the high-dimensional space, the distance of points
to the hyperplane does not change regardless of the dimensionality! This means that in high
dimensions, distances between points are large but distances to hyperplanes are tiny. Classifiers
such as SVMs and perceptrons use this idea.

## Perceptron

It is used for binary classification and assumes that the data is linearly separable, i.e.,
separable by a hyperplane. The hypothesis class is defined as $h(\mathbf{x}_i) = \textrm{sign}(\mathbf{w}^T \mathbf{x}_i + b)$.
We can absorb the bias into the weight by adding a constant dimension to the input.

To learn $\mathbf{w}$, we keep updating it as $\mathbf{w} = \mathbf{w} + y\mathbf{x}$ until
it learns to classify everything correctly.

**Theorem:** Suppose there is a hyperplane separating the points in a unit-normalized hypersphere,
defined by the normal $\mathbf{w}*$. Let $\gamma$ be its margin, or the distance to the nearest
point. Then the perceptron algorithm makes at most $\frac{1}{\gamma^2}$ mistakes.

## Estimating probabilities from data

**Maximum likelihood estimate (MLE):** We first assume some distribution from which the data
was sampled from, and then compute the parameters of the distribution which maximizes the
probability of generating the data, i.e.,

$$ \theta^{MLE} = \textrm{arg}\max_{\theta} P(X;\theta). $$

Example: for estimating the probability of head in a coin toss given some observations,
we can assume that the probability comes from a Bernoulli distribution with parameter $\theta$,
and then applying MLE will give us $\theta = \frac{\mathrm{number of heads}}{\mathrm{number of tosses}}$.

However, if we have small number of observations, this method of estimating $\theta$ can
give a very biased estimate. Instead, we can do it the Bayesian way. Assume that $\theta$ comes
from some prior distribution $P(\theta)$. To make computation easier, we can choose a conjugate
prior (for example, beta prior for bernoulli distribution and dirichlet prior for multivariate
distribution).

**Maximum a Posteriori (MAP):** The MAP estimate is given as:

$$ \theta^{MAP} = \mathrm{arg}\max_{\theta} P(\theta|D) = \mathrm{arg}\max_{\theta} P(D|\theta)P(\theta). $$

Quick summary:

* MLE prediction: $P(y|\mathbf{x};\theta)$ Learning: $\theta = \textrm{arg}\max_{\theta}P(D;\theta)$. Here $\theta$
is purely a model parameter.
* MAP prediction: $P(y|\mathbf{x},\theta)$ Learning: $\theta = \textrm{arg}\max_{\theta}P(\theta|D) = \mathrm{arg}\max_{\theta} P(D|\theta)P(\theta)$.
Here $\theta$ is a random variable.

