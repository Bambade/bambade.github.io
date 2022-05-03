---
layout: post
title: "GBO notes: Approximation algorithms"
tags: ["gbo", "algorithms"]
mathjax: true
---

This note is a brief introduction to approximation algorithms. Basically, the "Intro to
Algorithms" courses are concerned with problems which are solvable in poly-time (i.e., problems
in the class P). But there are a ton of important problems that are NP-hard, and cannot
be solved in poly-time. We want to design algorithms for such problems which can give us
some guarantees about the quality of the solutions --- by obtaining bounds on how much
worse the solution is compared to the optimal solution. Let us see some examples of these
problems and simple algorithms for solving them.

There are basically four types of algorithms to solve such problems:

* Greedy
* Local search
* Dynamic programming
* Randomized algorithms

Other than these, a more optimization-based approach is to express the problems as linear
programs and then solve them. This is called LP rounding. The basic idea is that we first
express the NP-hard problem as an integer linear program (ILP) which is still NP-hard. We
then let go of some of the integrality constraints, which makes the problem solvable. Once
we have the LP solutions, we then round them to get corresponding integer solutions to the
original problem.

## Examples of some NP-hard problems and algorithms

### Vertex cover

**Problem:** Given a graph $G = (V,E)$, find the minimum set of vertices that covers every edge.

**Algorithm:**

- Choose an arbitrary edge and add both its end-points to the set.
- Remove all edges incident on the newly added vertices.
- Repeat until no edges are remaining.

### Steiner tree

**Problem:** Given a graph $G = (V,E)$, costs $c$ for each edge, and a set of terminals $T$, find
the minimum-weighted tree that spans $T$.

Note that if $T = V$ this is the minimum spanning tree problem, and if $T$ is just a pair of nodes,
this is the shortest path problem. The set of non-terminal nodes on the tree are called Steiner nodes.

**Algorithm:** First, we apply metric completion to make sure all nodes are pairwise connected. This
is done by computing $c'(u,v)$ as the total length of the shortest path between $u$ and $v$.

For the metric $c'$, we compute the minimum spanning tree on the terminals $T$. This can be done using
a greedy approach (known as Kruskal's algorithm).

Now, we reduce this solution to a solution for the original problem by expanding the edges in the 
solution and combining the edges that are on multiple paths (which will create Steiner nodes).

> A graph is Eulerian if there is a closed tour that uses every edge exactly once.

### Metric TSP

**Problem:** Given a graph $G = (V,E)$, costs $c$ for each edge, find the minimum cost Hamiltonian
cycle (cycle visiting all nodes once).

**Algorithm:** First compute a minimum spanning tree $T$ on $G$. Then, double $T$ (double all edges)
and find a Eulerian tour $C$ on the new tree. Finally, shortcut $C$ to get the required Hamiltonian
cycle.

$$ c(H) \leq c(C) = c(2T) = 2c(T) \leq 2c(F) \leq 2(1-\frac{1}{n})c(H*), $$

where $F$ is the spanning tree obtained by removing the largest-cost edge from $H*$.

We are losing a factor of 2 here because we doubled all the edges (to make $G$ Eulerian). But we actually
only need to worry about the nodes with an odd degree.

**Christofides algorithm:** Find a min-cost perfect matching of the odd-degree nodes of $G$ and connect these.
Find a Eulerian tour of $T' = T + M$, where $T$ is the MST and $M$ is the perfect matching.

### Set cover

**Problem:** Given a universe $U$ of $n$ elements and $m$ sets $S_1,\ldots,S_m$, find the minimum
number of sets that covers $U$.

**Algorithm (greedy):** Greedily select a set $S_i$ which has the most elements outside the current
selection, and add it to the selection. This is a $\mathcal{O}(\log n)$ approximation. We can prove this
by analyzing the number of uncovered elements at any step.

### Weighted set cover

**Problem:** Given a universe $U$ of $n$ elements and $m$ sets $S_1,\ldots,S_m$ with associated costs
$c_1,\ldots,c_m$, find the minimum weight of sets, $\sum_{i\in I}c_i$, that covers $U$.

**Algorithm:** We can again use a greedy algorithm, but now we try to maximize the "bang for the buck",
i.e., maximize $\frac{|X \cap S_i|}{c_i}$.

### Max k-cover

**Problem:** Given a universe $U$ of $n$ elements, $m$ sets $S_1,\ldots,S_m$, and an integer $k$, 
find $k$ sets that maximize the number of elements covered in $U$.

**Algorithm:** We can again use the greedy approach as in set cover, but stop after $k$ iterations.
This gives us a $(1-\frac{1}{e})$-approximation.

### K-center

**Problem:** Given a set of $n$ points on a metric space, find a subset $F$ of size $k$ (called "centers"), such that
the maximum distance of any point to the nearest center is minimized.

**Algorithm (greedy):** Start with an arbitrary center. Greedily select a center that is farthest from
the previously selected centers, and add it to $F$. This is a 2-approximation.

---

So far we have mostly seen greedy algorithms for problems. Let us now look at some "local search" methods.
The idea in these is to start with a feasible solution and then make local improving changes
until we get to a local optimum.

### Max-cut

**Problem:** Given a graph $G=(V,E)$, find a partition of the vertices that maximizes the number
of edges going across the partition.

> The max-cut problem is NP-hard, but the min-cut problem is in P.

**Algorithm:** Start with an arbitrary cut $S, \bar{S}$. While there exists some $v \in V$ s.t. $v$
has more edges to same side of cut than to other side, move $v$ to the other side. This algorithm is
a 2-approximation.

### Weighted max-cut

**Problem:** Given a graph $G=(V,E)$ with weights $w(e)$ on edges, find a partition of the vertices that 
maximizes the total weight of edges going across the partition.

We cannot use the same idea as before for this problem since it may take exponential time.

**Algorithm:** Instead, we bound the improvement with a multiplicative factor, i.e., we only
change sides for a node if the weight improvement is at least $1+\frac{\epsilon}{n}$ times the
current weight. This ensures that the algorithm ends in a polynomial time, and the solution is
a $2+\epsilon$-approximation.

---

Let us see an example of dynamic programming to solve some NP-hard problems.

### Knapsack

**Problem:** Given $n$ items, their corresponding profits $p_i$ and sizes $s_i$, find the maximum
profit that can be obtained such that the total size is less than $k$.

**Naive greedy method:** Use the "bang-for-the-buck", i.e., keep adding items such that the ratio
$\frac{p_i}{s_i}$ is maximum. Once we have achieved the limit, either return the items in the
knapsack or the first item that couldn't be added, whichever is more. This is a 2-approximation.

**D.P. algorithm:** Instead of asking what is the maximum profit that we can get for size $k$, we
ask what is the minimum size necessary to achieve some profit. We know that the maximum profit is upper
bounded by $nM$, where $M$ is the maximum profit by some item. We can set this up as a recurrence
similar to how DP problems are solved.
