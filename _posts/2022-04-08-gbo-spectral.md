---
layout: post
title: "GBO notes: Spectral clustering"
tags: ["gbo", "i-vectors"]
mathjax: true
---

In this note, I will review a popular clustering algorithm called spectral clustering. We will
discuss its connection to the min-cut problem in graph partitioning, and 
then look at 2 methods to extend it to multi-class clustering. This post is based heavily on
[this tutorial](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf).

## Similarity graph and the Laplacian matrix

Suppose we have $n$ observations $x_1,\ldots,x_n$ and pairwise similarity scores $s_{ij}\geq 0$.
We can represent them as a *similarity graph* $G = (V,E)$, where $V = \{x_1,\ldots,x_n\}$ and
$E$ contains edges between nodes for which $s_{ij}$ is greater than some threshold, weighted
by $s_{ij}$.

This is a unidirected graph, and we can define it in terms of its adjacency matrix $W = \{w_{ij}\}_{i,j=1,\ldots,n}$. We can also define a *degree matrix* $D$ which is a diagonal matrix such that

$$ d_i = \sum_{j=1}^n w_{ij}. $$

Given two subsets of vertices $A$ and $B$, we define $W(A,B) = \sum_{i\in A, j\in B} w_{ij}$.

The **unnormalized graph Laplacian** is defined as $L = D - W$. Let use prove some properties
of the Laplacian matrix.

**Property 1:** *$L$ is symmetric and positive semi-definite.*

**Proof:** Since both $D$ and $W$ are symmetric, $L$ must be symmetric. Consider some vector
$f \in \mathbb{R}^n$. We have

$$\begin{aligned}
f^T Lf &= f^T Df - f^T Wf \\
    &= \sum_{i=1}^n f_i^2 d_i - \sum_{i,j=1}^n f_i f_j w_{ij} \\
    &= \frac{1}{2} \left( \sum_{i=1}^n f_i^2 d_i - 2 \sum_{i,j=1}^n f_i f_j w_{ij} + \sum_{i=1}^n f_i^2 d_i \right) \\
    &= \frac{1}{2} \sum_{i,j=1}^n w_{ij} (f_i - f_j)^2 \geq 0,
\end{aligned}$$

which means $L$ is positive semi-definite.

> The above also means that if $f$ is a vector that assigns values to all the nodes in the graph $G$,
> then the sum of weighted squared distances between all neighbors can be obtained by computing $f^T Lf$.

**Property 2:** *The smallest eigenvalue is 0, and corresponding eigenvector is constant $\mathbb{1}$.*

**Proof:** We have

$$\begin{aligned}
L \mathbb{1} &= D\mathbb{1} - W\mathbb{1} \\
    &= (d_1,\ldots,d_n)^T - (\sum_{j=1}^n w_{1j},\ldots,\sum_{j=1}^n w_{nj})^T \\
    &= \mathbb{0} = 0\cdot \mathbb{1}.
\end{aligned}$$

Hence, $0$ is an eigenvalue of $L$ with $\mathbb{1}$ as the corresponding eigenvector.

**Property 3:** *$L$ has orthogonal eigenvectors.*

**Proof:** Let $v_1$ and $v_2$ be eigenvectors of $L$ corresponding to eigenvalues $\lambda_1$
and $\lambda_2$. We have

$$\begin{aligned}
\lambda_1 v_1^T v_2 &= (\lambda_1 v_1)^T v_2 \\
    &= (L v_1)^T v_2 \\
    &= v_1^T L^T v_2 \\
    &= v_1^T (L v_2) \\
    &= v_1^T (\lambda_2 v_2) \\
    &= \lambda_2 v_1^T v_2 \\
\implies v_1^T v_2 &= 0,
\end{aligned}$$

where $L^T =L$ since $L$ is symmetric.

**Property 4:** *$L$ has $n$ non-negative, real eigenvalues.*

**Proof:** Since $L$ is PSD, all its eigenvalues must be real and non-negative. The smallest
eigenvalue is 0, as shown in Property 2.

Some interesting results arise from the relationships between the graph and the spectrum of
the Laplacian. But before that, let us define a few terms.

The **geometric multiplicity** of an eigenvalue $\lambda$ of matrix $A$ is the number of non-zero
eigenvectors corresponding to $\lambda$, i.e., it is the dimension of the null-space of
$A - \lambda I$.

The **algebraic multiplicity** of an eigenvalue $\lambda$ of matrix $A$ is the number of times it
occurs as a root of the characteristic equation $\text{det}(A-\lambda I) = 0$.

As such, the G.M. cannot be greater than the A.M (skip proof).

## Spectrum of the Laplacian

**Result 1:** The algebraic multiplicity of the eigenvalue $0$ of $L$ is equal to the number of connected 
components of the graph $G$.

**Proof:** First, suppose we have $k$ connected components $S_1,\ldots,S_k$. Define $k$ vectors $u_1,\ldots,u_k \in \{0,1\}^n$
such that $u_i$ contains 1 in the indices corresponding to the nodes in component $S_i$. Since all
components are disjoint, so $u_i \cdot u_j = 0$ for all $i,j$, i.e., the vectors are orthogonal. Also,
it is easy to verify that $Lu_i = 0 \forall i$. Hence, there are at least $k$ orthogonal eigenvectors
corresponding to eigenvalue 0. This means that the geometric multiplicity of 0 is at least $k$, and 
consequently, the algebraic multiplicity of 0 is at least $k$.

Now consider some eigenvector $v$ corresponding to eigenvalue 0. We have

$$\begin{aligned}
Lv &= 0 \\
\implies v^T Lv &= 0 \\
\implies \sum_{i,j} w_{i,j}(v_i - v_j)^2 = 0,
\implies \sum_{(i,j)\in E} (v_i - v_j)^2 = 0,
\end{aligned}$$

since for the pairs $(i,j)$ which are not edges, the weight is 0. This means that the eigenvector
$v$ is such that the indices corresponding to connected components have the same value. Hence,
we can have scalars $\alpha_1,\ldots,\alpha_k$ such that

$$ v = \sum_{i=1}^k \alpha_i u_i, $$

where $u_i$'s are as defined earlier. This means that all eigenvectors correposnding to 0 lie
in the basis spanned by $u_i$'s, which means that there are at most $k$ orthogonal eigenvectors,
and so the A.M. of 0 is at most $k$.

**Result 2:** The **Fiedler vector** of $L$ corresponds to the sparsest cut of the graph.

The Fiedler vector is the eigenvector associated with the second smallest eigenvalue of $L$.

## Spectral clustering using k-means

In these methods, we first compute the Laplacian $L$, obtain its first $k$ eigenvectors as
the matrix $U \in \mathbb{R}^{n\times k}$, and then cluster the $n$ rows of the matrix $U$
using k-means clustering. The idea is that we used the spectrum of $L$ to convert the original
data points into $k$-dimensional representations that are well separable.

## Spectral clustering using convex optimization

Another method that was proposed in [this paper](https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf) 
presents a more mathematically robust approach to multi-class spectral clustering. The idea is
to represent the graph partitioning problem as a discrete optimization problem. Suppose
we have estimated the number of clusters as $K$. Then, we can write the clustering problem as

$$
\begin{aligned}
\mathrm{maximize} \quad \epsilon(X) &= \frac{1}{K}\sum_{k=1}^K \frac{X_k^T \mathbf{A} X_k}{X_k^T \mathbf{D} X_k} \\
\mathrm{subject~to}\,\,\,\,\,\, \quad X &\in \{0,1\}^{N\times K}, \\
    X \boldsymbol{1}_K &= \boldsymbol{1}_N.
\end{aligned}
$$

Here, the objective function tries to maximize the average "link-ratio" in the graph, and the
constraint just enforces that each item can be assigned to exactly 1 cluster. Unfortunately, these
discrete constraints make the problem NP-hard. So instead of solving the discrete version of the problem,
we remove the constraints and make it continuous. We also rewrite it as follows.

Let $Z = f(X) = X(X^T DX)^{-\frac{1}{2}}$ (note that $f$ is invertible). We can verify that $Z^T DZ = I_K$. Using this, we can simplify the 
objective as

$$
\begin{aligned}
\mathrm{maximize}\,\,\,\,\, \quad \epsilon(Z) &= \frac{1}{K}\text{tr}(Z^T\mathbf{A}Z) \\
\mathrm{subject~to} \quad Z^T\mathbf{D}Z &= I_K.
\end{aligned}
$$

Let $P=D^{-1}A$. Since $P$ is diagonalizable, we can write its eigen-decomposition as $P = VSV^{-1}$,
where $V$ is the matrix of eigenvectors and $S$ is the diagonal matrix of eigenvalues. Let $Z^{\ast}$ contain
the first $K$ eigenvectors of $P$, and $\Lambda^{\ast}$ is the $K\times K$ sub-matrix from $S$. Then,
the solution set for the above problem is given as

$$ \{Z^* R : R^T R = I_K, PZ^* = Z^* \Lambda^*\}, $$

i.e., it is the subspace spanned by the first $K$ eigenvectors of $P$ through orthonormal matrices.
The matrix $Z^{\ast}$ is a continuous solution to our clustering problem. We now need to discretize it
to obtain $X^{\ast}$ which obeys the discrete constraints. Suppose $\tilde{X}^{\ast} = f^{-1}(Z^{\ast})$. Then our
new goal is to solve the following optimization problem:

$$
\begin{aligned}
\mathrm{minimize} \quad \phi(X,R) &= \left\lVert X - \tilde{X}^{\ast}R \right\rVert^2 \\
\mathrm{subject~to}\, \quad\quad\quad X &\in \{0,1\}^{N\times K}, \\
    X \boldsymbol{1}_K &= \boldsymbol{1}_N, \\
    R^TR &= I_K.
\end{aligned}
$$

This means that we are trying to find a discrete $X$ which is closest to any one of the solutions in the 
solution set described above. This problem is again difficult to solve jointly in $X$ and $R$, so we
optimize it alternatively. Given a fixed $R$, the optimum $X^{\ast}$ is obtained by applying
non-maximal suppression on the rows of $\tilde{X}^{\ast}R$. Then, given a fixed $X$,
the optimum $R^{\ast}$ is given by

$$ R^{\ast} = \tilde{U}U^T, $$

where $X^T\tilde{X}^{\ast} = U\Sigma \tilde{U}^T$ is a singular value decomposition. We alternate between
the two steps until convergence and finally return $X^{\ast}$ as the clustering assignment matrix.
