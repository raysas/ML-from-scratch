# ML-from-scratch

This repository contains implementations of various machine learning algorithms from scratch. The purpose of this repository is to go through the algorithms and practice implementing them in Python. The algorithms are implemented in an object-oriented manner, and the code is well-documented (eventually).  

## Unsupervised Learning

This consists of algorithms that do not require labeled data. The main task here is _clustering_.  
Clustering is about grouping similar data points together. Similarity is a complex function that can be defined as the opposite of distance. The conditions for a good similarity function:  
1. _Reflexivity_: `similarity(x, x) = 1`  
2. _Symmetry_: `similarity(x, y) = similarity(y, x)`  

### K-Means Clustering

K-means clustering is a simple and widely-used clustering algorithm. It is an iterative algorithm that partitions the data into `k` clusters. The algorithm works as follows: 
```text 
1. Initialize `k` cluster centroids randomly.  
2. Assign each data point to the nearest cluster centroid.
3. Update the cluster centroids by taking the mean of all data points assigned to that cluster.
4. Repeat steps 2 and 3 until convergence.
```

### Hierarchical Clustering  

Hierarchical clustering is a clustering algorithm that builds a tree of clusters. The tree is called a _dendrogram_, it can be built via either _agglomerative_ or _divisive_ methods. We will be implementing the agglomerative method. The algorithm works as follows:  
```text
1. Start with `n` clusters, each containing one data point.
2. Find the two closest clusters and merge them into a single cluster.
3. Repeat step 2 until there is only one cluster left.
```

When merging clusters, we need to define a _linkage criterion_ to determine the distance between clusters. The most common linkage criteria are:  
1. _Single linkage_: The distance between two clusters is the distance between the two closest points in the clusters.  
2. _Complete linkage_: The distance between two clusters is the distance between the two farthest points in the clusters.  
3. _Average linkage_: The distance between two clusters is the average distance between all pairs of points in the clusters.  
4. _Centroid linkage_: The distance between two clusters is the distance between the centroids of the clusters.  

### Spectral Clustering

Spectral clustering is a clustering algorithm that has advantages over traditional clustering algorithms like K-means, especially when the clusters are non-convex or have complex shapes.  
It is based on the _spectral graph theory_, which studies the properties of graphs in relation to the eigenvalues and eigenvectors of the graph's adjacency matrix. This approach eventually leads to a _low-dimensional embedding_ of the data, constructed from the eigenvectors of the graph Laplacian matrix which is the solution to the _normalized cut_ problem, a variant of the _max flow/min cut_ NP-hard problem.
Then the data points are clustered in this low-dimensional space using a clustering algorithm like K-means.

_Some linear/graph theory background because it's cool_: 
A Graph $G=(V, E)$ is a set of vertices $V$ and edges $E$ connecting the vertices. 
There are 3 types of matrices when it comes to representing them, the most common being:  
- _adjacency matrix_ $A$ which is a square matrix of size $|V| \times |V|$ where $A_{ij} = w_{ij}$ if there is an edge between vertices $i$ and $j$, and $0$ otherwise. The _degree matrix_ $D$, on the side, is a diagonal matrix of size $|V| \times |V|$ where $D_{ii} = \sum_{j} A_{ij}$, i.e. the sum of the weights of the edges incident to vertex $i$, and this can be calculated as $D = \sum_{i} A$.   
- _incidence matrix_ $B$ which is a matrix of size $|V| \times |E|$ where $B_{ij} = 1$ if vertex $i$ is incident to edge $j$, $-1$ if it is the other end of the edge, and $0$ otherwise (each column has 2 non-zero entries, one +1 and one -1).  
- _Laplacian matrix_ $L$ which is a matrix of size $|V| \times |V|$ where $L = D - A$. The Laplacian matrix has some interesting properties, for example, the sum of the elements in each row/column is zero, and the smallest eigenvalue is zero with the corresponding eigenvector being the all-ones vector. It is also symmetric and positive semi-definite.  

With this we can define the _normalized Laplacian matrix_ $L_{norm} = D^{-1/2} L D^{-1/2}$ which is used in spectral clustering. The Laplacian matrix is normalized by the degree matrix to make it scale-invariant. According to Fiedler's theorem, the second smallest eigenvalue of the normalized Laplacian matrix corresponds to the optimal cut of the graph.



The algorithm works as follows:  
```text
```