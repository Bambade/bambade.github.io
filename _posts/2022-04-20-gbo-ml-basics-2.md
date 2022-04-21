---
layout: post
title: "GBO notes: Machine learning basics (Part 2)"
tags: ["gbo", "wfst"]
mathjax: true
---

In this series of notes we will review some basic concepts that are usually covered in an Intro to ML
course. These are based on [this course](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) from Cornell.

In Part 2, we will look at Naive Bayes, logistic regression, gradient descent, and linear
regression.

## Bayes classifier and Naive Bayes

If we have an estimate $P(y|\mathbf{x})$, we can build a classifier that simply returns the $y$ which
maximizes this probability. Such a classifier is called a Bayes Optimal classifier.

So how can we estimate $P(y|\mathbf{x})$? Using the MLE predictor, we can set it to be equal to
the fraction of times $y$ occured as label when the input was $\mathbf{x}$. However, for high
dimensional inputs, it is highly unlikely that $\mathbf{x}$ repeats, leading to the sparsity
problem.

To solve this problem, we can use Bayes rule and the Naive Bayes assumption. First, recall that
$\textrm{arg}\max_y P(y|\mathbf{x}) = \textrm{arg}\max_y P(\mathbf{x}|y)P(y)$, so we can
estimate those other two terms instead. Computing $P(y)$ is easy in case of tasks like binary
classification since we can simply count the occurrence of the classes. But computing
$P(\mathbf{x}|y)$ is not so easy. To make it simpler, we make the **Naive Bayes** assumption:

$$ P(\mathbf{x}|y) = \prod_{j=1}^d P(x_j|y), $$

i.e., we assume that the feature dimensions are independent of each other.

Suppose we have a real-valued feature vector. We will assume that for every class, each feature
is generated from a Gaussian distribution. This is equivalent to saying that

$$ P(\mathbf{x}|y=c) = \mathcal{N}(\mu_c, \Sigma_c), $$

where $\Sigma_c$ is a diagonal covariance matrix. The parameters can then be estimated using MLE.

## Logistic regression

Logistic regression is the *discriminative* counterpart of Naive Bayes. It defines $P(y|\mathbf{x})$
to take the form

$$ P(y|\mathbf{x}) = \frac{1}{1 + e^{-y(\mathbf{w}^T\mathbf{x}+b)}}. $$

Note that this form is the same as what is obtained using Naive Bayes by taking a Gaussian form
for the likelihood function. The model is trained by maximizing the conditional likelihood, or 
equivalently, minimizing the negative log-likelihood, which gives

$$ \mathbf{w}^{MLE} = \mathrm{arg}\min_{\mathbf{w}} \sum_{i=1}^n \log (1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i}). $$

There are no closed form solutions to this optimization problem, so we have to use gradient descent.

In addition to the MLE estimate, we can also compute the MAP estimate by treating $\mathbf{w}$ as a
random variable. If we assume that $\mathbf{w} \sim \mathcal{N}(\mathbf{0},\sigma^2 I)$, then

$$ \mathbf{w}^{MAP} = \mathrm{arg}\min_{\mathbf{w}} \sum_{i=1}^n \log (1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i}) + \frac{\mathbf{w}^T\mathbf{w}}{2\sigma^2}, $$

which again needs to be solved by gradient descent.

## Gradient descent and Newton's method

Hill climbing methods use Taylor's apprximation of the function. Suppose we want to minimize
a function $f(\mathbf{w})$. For some small $\mathbf{s}$, we can make first and second
order approximations as

$$ f(\mathbf{w} + \mathbf{s}) = f(\mathbf{w}) + g(\mathbf{w})^T \mathbf{s}, $$

$$ f(\mathbf{w} + \mathbf{s}) = f(\mathbf{w}) + g(\mathbf{w})^T \mathbf{s} + \frac{1}{2}\mathbf{s}^T H(\mathbf{w})\mathbf{s}, $$

where $g$ and $H$ are the gradient and Hessian of $f$. Gradient descent uses the first order
approximation and Newton's method uses second order approximation.

In gradient descent, we choose $\mathbf{s}$ to be the gradient times some scaling factor (called
the learning rate):

$$ \mathbf{s} = -\alpha g(\mathbf{w}). $$

Newton's method sets $\mathbf{s}$ as

$$ \mathbf{s} = -[H(\mathbf{w})]^{-1}g(\mathbf{w}). $$

## Linear regression

So far we have looked mainly at classification tasks. We will now see a regression task, i.e.,
where $y_i \in \mathbb{R}$. Linear regression assumes that the model is a Gaussian,

$$ y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i, $$

where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.

**Estimating with MLE:**

$$
\begin{aligned}
\mathbf{w} &=\underset{\mathbf{w}}{\operatorname{argmax}} P\left(y_{1}, \mathbf{x}_{1}, \ldots, y_{n}, \mathbf{x}_{n} \mid \mathbf{w}\right) \\
&=\underset{\mathbf{w}}{\operatorname{argmax}} \prod_{i=1}^{n} P\left(y_{i}, \mathbf{x}_{i} \mid \mathbf{w}\right) \\
&=\underset{\mathbf{w}}{\operatorname{argmax}} \prod_{i=1}^{n} P\left(y_{i} \mid \mathbf{x}_{i}, \mathbf{w}\right) P\left(\mathbf{x}_{i} \mid \mathbf{w}\right) \\
&=\underset{\mathbf{w}}{\operatorname{argmax}} \prod_{i=1}^{n} P\left(y_{i} \mid \mathbf{x}_{i}, \mathbf{w}\right) P\left(\mathbf{x}_{i}\right) \\
&=\underset{\mathbf{w}}{\operatorname{argmax}} \prod_{i=1}^{n} P\left(y_{i} \mid \mathbf{x}_{i}, \mathbf{w}\right) \\
&=\underset{\mathbf{w}}{\operatorname{argmax}} \sum_{i=1}^{n} \log \left[P\left(y_{i} \mid \mathbf{x}_{i}, \mathbf{w}\right)\right] \\
&=\underset{\mathbf{w}}{\operatorname{argmax}} \sum_{i=1}^{n}\left[\log \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)+\log \left(e^{-\frac{\left(\mathbf{x}_{i}^{\top} \mathbf{w}-y_{i}\right)^{2}}{2 \sigma^{2}}}\right)\right] \\
&=\underset{\mathbf{w}}{\operatorname{argmax}}-\frac{1}{2 \sigma^{2}} \sum_{i=1}^{n}\left(\mathbf{x}_{i}^{\top} \mathbf{w}-y_{i}\right)^{2} \\
&=\underset{\mathbf{w}}{\operatorname{argmin}} \frac{1}{n}\sum_{i=1}^{n}\left(\mathbf{x}_{i}^{\top} \mathbf{w}-y_{i}\right)^{2}
\end{aligned}
$$

Therefore, computing the MLE for linear regression is equivalent to minimizing the mean squared error.
This can be done using gradient descent (see above), but it also has a closed form solution
given as

$$ \mathbf{w} = (\mathbf{X}\mathbf{X}^T)^{-1} \mathbf{X}\mathbf{y}^T. $$

**Estimating with MAP:**

We assume that $\mathbf{w}$ comes from a distribution given by $P(\mathbf{w}) = \frac{1}{\sqrt{2\pi\tau^2}}e^{-\frac{\mathbf{w}^T\mathbf{w}}{2\tau^2}}$.
Solving gives us

$$\mathbf{w} = \underset{\mathbf{w}}{\operatorname{argmin}} \frac{1}{n}\sum_{i=1}^{n}\left(\mathbf{x}_{i}^{\top} \mathbf{w}-y_{i}\right)^{2} + \frac{\sigma^2}{n\tau^2}\Vert \mathbf{w} \Vert^2,$$

which is also known as ridge regression. Conceptually, it adds an L2-regularization to the mean
squared loss.
