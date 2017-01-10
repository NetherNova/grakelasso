# Repository for Semantic-guided Machine Learning Algorithms

##  Knowledge Graph Constraints for Multi-Label Graph Classification
This is an implementation of the constraint-based subgraph pattern mining algorithm.

For details, check our [DamNet'16 paper](http://damnet.reading.ac.uk/program.html)

The [gspan](gspan) module contains a modified version of the original gspan algorithm.

The following feature selection metrics are supported:
* Information Gain
* Top-k frequent
* GMLC (Kong et al. 2012)

You can add your own Must-Link and Cannot-Link constraints implementation in [constraints.py](gspan/constraints.py)

For multi-process evaluation, check [multi_process_eval.py](gspan/multi_process_eval.py)

## Semantic Graph Kernel Lasso (GraKeLasso) (0.9)
This is a simple Python API for training and evaluating graph-regularized linear regression models

It was built to test the ideas of graph kernel regularization - see our [ISWC'15 paper](http://iswc2015.semanticweb.org/sites/iswc2015.semanticweb.org/files/93670191.pdf)

To get started, have a look at [testgrake.py](tests/testgrake.py)

You need to provide data and the Laplacian matrix of the semantic graph as in [data](data/laplacian.csv)

### Version 0.9
Most important features are:

* Loading data and regularization matrices
* Standard Lasso Coordinate-descent implementation
* Modified Coordinate-descent for graph-regularization implementation
* n-fold cross-validation

### Requirements
* Python (>= 2.7)
* NumPy (>= 1.9)
* SciKit-Learn (>=0.15.2)
* Pandas (>=0.15.1)