---
layout: post
title: "GBO notes: Machine learning basics (Part 5)"
tags: ["gbo", "ml-basics"]
mathjax: true
---

In this series of notes we will review some basic concepts that are usually covered in an Intro to ML
course. These are based on [this course](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) from Cornell.

In this final part, we will look at k-dimensional trees, decision trees, bagging, and boosting.

## k-Dimensional trees

In the k-NN algorithm, at test time, we have to compute the distance between the test point
and every other point in the training set. This can take a long time if $n$ is large. To
avoid computing all these distances, we can use a clever data structure called a k-D tree.

The idea is that we partition the feature space into boxes. For each feature, we split the
dataset in half based on the feature, and repeat this process for top $k$ features. Usually,
features with the most variance are chosen for partitioning. Now, we only need to search
within the box, if we can show that the nearest neighbor in the box is closer than the
distance to the walls.

A limitation of this approach is that in high dimensions, all the points are far apart, and
so it is likely that every point will be in its own box. Also, all the splits are axis-aligned.

An improvement to the simple k-d tree is the Ball-tree, in which instead of boxes, we split
based on hyperspheres.

## Decision trees

Although k-d trees improve upon the time complexity of k-NN algorithm for classification,
they still require storing a lot of information about the training data and what boxes they
lie in. For classification, we don't really need all this storage overhead. Decision
trees help to build a non-linear classifier which does not need to store the training data.

This is done by recursively splitting the dataset (i.e., building a tree) based on thresholding
some chosen feature. We stop the splitting when the set at the leaves are pure (i.e., contain
all points with the same label). Once we have constructed the tree, we do not need to store
the training points anymore. They are also super fast at test time and non-parametric since
they only require feature comparisons.

Although it is NP-hard to find a decision tree with the minimum size, we can find good
approximations using some impurity functions like Gini impurity.

### CART

An important model related to decision trees are CART (Classification and Regression Trees).
It can be used to perform regression task, i.e., $y\in \mathbb{R}$. For constructing the
tree, we try to minimize a loss function which computes the average squared distance from the
average label of a leaf. At test time, we return the leaf average as the prediction.

### Parametric and non-parametric models

Models such as perceptron, logistic regression, etc. are *parametric*, meaning that they
have a fixed number of parameters independent of the size of training data. Algorithms like 
k-NN are *non-parametric* since the apparent number of parameters scales with the size 
of the training data (we need to store all training data to make predictions).

Kernel SVMs are an interesting case since their category depends on the type of kernel used.
Linear kernel SVMs are parametric since they are functionally similar to perceptrons.
RBF kernel SVMs need to store the support vectors, and so are non-parametric. Polynomial
kernels are technically parametric, but in practice, it is much more convenient to just
store the support vectors, and so they behave like non-parametric models.

Decision trees also behave both ways. If trained to full depth ($\mathcal{O}(\log n)$), they
are non-parametric. But in practice, we limit the depth by a maximum value and so they
behave like a parametric model.

## Bagging

Bagging is an ensemble technique used to reduce variance (overfitting). It stands for
"bootstrap aggregating".

The idea comes from weak law of large numbers. If we have $m$ datasets and train a classifier
on each of them, then taking their average would give us result close to the expectation,
and the variance would go to 0. Of course, we don't really have $m$ datasets. So we actually
just sample from $D$ with replacement to simulate the effect of having $m$ datasets.

**Random forest:** It is the most popular bagging-based method --- it is nothing but decision
trees with bagging, with one small modification: at each split, we randomly sample $k$ (less than
$d$) features, and only use these for the split.

## Boosting

While bagging is used to reduce variance, boosting is a method to reduce bias, i.e., solve
underfitting. Suppose we have a bunch of "weak learners" (i.e., high training error). The
question is how can we combine these to form a strong learner?

The idea is to create an ensemble classifier as $H_T(\mathbf{x})=\sum_{t=1}^T \alpha_t h_t(\mathbf{x})$
in iterative fashion. This is like gradient descent, but instead of adding the gradient at the
point, we add functions to our ensemble. At any time step $t$, we search for the weak learner
that minimizes the total loss

$$ l(H) = \frac{1}{n}\sum_{i=1}^n l(H(\mathbf{x}_i),y_i), $$

i.e,

$$ h_{t+1} = \operatorname{argmin}_{h\in H} l(H_t + \alpha h). $$

To find such an $h$, we use gradient descent over the function space.

**Gradient descent in functional space:** Again, using Taylor approximation, we get

$$ l(H + \alpha h) = l(H) + \alpha <\Delta l(H),h>. $$

So the solution is

$$ h_{t+1} = \operatorname{argmin}_{h\in H} l(H_t + \alpha h) = \operatorname{argmin}_{h\in H} <\Delta l(H),h>, $$

where we have fixed $\alpha$ to be some small constant (like 0.1). The inner product of the
functions is given as

$$ <\Delta l(H),h> = \sum_{i=1}^n \frac{\partial l}{\partial [H(\mathbf{x}_i)]}h(\mathbf{x}_i). $$

This means that if we have some algorithm to compute $\frac{\partial l}{\partial [H(\mathbf{x}_i)]}$,
we are done. One such algorithm is known as Gradient Boosted Regression Trees (GBRT), which is
one of the components of the XGBoost algorithm.

**GBRT:** The weak learners are fixed-depth decision trees (or CART). In this case, the hypothesis
that minimizes the loss turns out to be the one that is closes to $(y_i - H(\mathbf{x}_i))$ in
squared loss.

AdaBoost is another powerful boosting algorithm over binary learners, but we will skip it here.
