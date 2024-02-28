---
aliases:
  - t-SNE
cssclasses: 
ReviewedDate: "[[Daily_Notes/08-12-23]]"
tags:
  - my/summary
  - state/process
child: 
url: https://arxiv.org/pdf/1802.03426.pdf
---
`=this.url` -> 选了一篇分析原文的文章，这样比较 easy to read


Dataset cardinality $n$

Datapoint $\mathbf{x}\in \mathbb{R}^{h}$, $h$ for high, 
Low dimension map of points $\mathbf{y}\in \mathbb{R}^{l}$, $l$ for low

Building a *joint probability distribution* or *similarity matrix* over all pairs of data points $\left\{ (\mathbf{x}_{i}, \mathbf{x}_{j}) \right\}_{1\le i\ne j \le n}$ 

Represented as $\mathbf{P}=(p_{ij})_{1\le i,j \le n}$
- $p_{ii} =0,\forall 1\le i \le n$
- $p_{ij}$ 
	$$
	p_{ij}=\frac{\exp(- \left\| \mathbf{x}_{i} - \mathbf{x}_{j} \right\|^{2} / 2\tau_{i}^{2} )}{\sum_{k\ne j} \exp(- \left\| \mathbf{x}_{k} - \mathbf{x}_{j} \right\|^{2} / 2\tau^{2}_{i}  ) }
	$$
	- $\tau_{i}$s are tuning params based on 
		- perplexity measure
		- binary search


Building a *joint probability distribution* or *similarity matrix* over all pairs of data points $\left\{ (\mathbf{y}_{i}, \mathbf{y}_{j}) \right\}_{1\le i\ne j \le n}$ 

Represented as $\mathbf{Q}=(q_{ij})_{1\le i,j \le n}$
- $q_{ii} =0,\forall 1\le i \le n$
- $q_{ij}$ 
	$$
	q_{ij}=\frac{(1+\left\| \mathbf{y}_{i} - \mathbf{y}_{j} \right\|^{2} )^{-1}}{\sum_{k\ne j} (1+ \left\| \mathbf{y}_{k} - \mathbf{y}_{j} \right\|^{2} )^{-1} }
	$$

Find $\left\{ \mathbf{y}_{i} \right\}_{1\le i \le n}$ that minimizes the KL divergence between $\mathbf{P}$ and $\mathbf{Q}$

$$
\arg\min_{\mathbf{y}_{1},\dots,\mathbf{y}_{n}} \text{KL}(\mathbf{P}\|Q)=\arg\min_{\mathbf{y}_{1},\dots,\mathbf{y}_{n}}\sum_{i\ne j}p_{ij}\log \frac{p_{ij}}{q_{ij}}
$$

## Reference

