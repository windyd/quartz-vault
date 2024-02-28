---
aliases:
  - UMAP
  - Uniform Manifold Approximation and Projection
cssclasses: 
ReviewedDate: "[[Daily_Notes/04-12-23]]"
tags:
  - my/summary
  - state/process
child: 
pdf: "[[UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction.pdf]]"
---

> At a high level, UMAP uses *local manifold approximations* and patches together their *local fuzzy simplicial set representations* to construct a *topological representation* of the high dimensional data. Given some low dimensional representation of the data, a similar process can be used to construct an equivalent topological representation. UMAP then optimizes the layout of the data representation in the low dimensional space, to minimize the *cross-entropy* between the two topological representations.

[[UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction.pdf#page=4&selection=7,0,13,58|UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction, page 4]]

[[UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction.pdf#page=5&selection=29,0,129,44|Lemma 1]] -> 
## Reference

- shallow layer of understanding
	[Understanding UMAP (pair-code.github.io)](https://pair-code.github.io/understanding-umap/)


- manifold theory and topological data analysis
	
	> Much of the theory is most easily explained in the language of topology and category theory. Readers may consult [39], [49] and [40] for background.
	
	[[UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction.pdf#page=3&selection=36,31,40,35|UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction, page 3]]
	
	[39] Saunders Mac Lane. Categories for the working mathematician, vol- ume 5. Springer Science & Business Media, 2013.
	[40] J Peter May. Simplicial objects in algebraic topology, volume 11. Uni- versity of Chicago Press, 1992.
	[49] Emily Riehl. Category theory in context. Courier Dover Publications, 2017.

- siplicial sets
	
	> For more details on simplicial sets we refer the reader to [25], [40], [48], or [22].
	
	[[UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction.pdf#page=5&selection=215,38,216,46|UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction, page 5]]

- Category theory

	[Functor - Wikipedia](https://en.wikipedia.org/wiki/Functor)
