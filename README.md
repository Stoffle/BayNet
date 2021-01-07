# BayNet

BayNet is a Python library for generating, sampling data from, comparing, and visualising Bayesian Networks.

## Installation
```bash
pip install BayNet
```

## Usage
### Generate a 10-node Forest Fire DAG, and parameters, then sample data from it:
```python
from baynet import DAG
dag = DAG.generate("forest fire", 10, seed=1) # Creates a DAG
dag.generate_discrete_parameters(alpha=5.0, min_levels=3, max_levels=5, seed=1) # Generates parameters
data = dag.sample(1_000) # Samples data, returning a pandas DataFrame
```
### Generate a 5-node Barabasi-Albert (preferential attachment) graph and plot it:
```python
from baynet import DAG
DAG.generate("barabasi albert", 5, seed=1).plot() # Saves 'DAG.png' in working directory
```
![Example DAG.png](https://raw.githubusercontent.com/Stoffle/BayNet/master/example_DAG.png)


### Generate two 5-node Erdos-Renyi DAGs and compare them:
```python
from baynet import DAG, metrics
dag_1 = DAG.generate("erdos_renyi", 5,seed=1)
dag_2 = DAG.generate("erdos_renyi", 5)
print(metrics.shd(dag_1, dag_2)) # prints DAG SHD, in this case 3
print(metrics.shd(dag_1, dag_2, skeleton=True)) # prints skeleton SHD, in this case 3
dag_1.compare(dag_2).plot() # saves 'comparison.png' in working directory
```
![Example comparison.png](https://raw.githubusercontent.com/Stoffle/BayNet/master/example_comparison.png)

Taking dag_1 to be the ground truth and dag_2 to be a structure learning result:
- Dashed red arcs represent false negatives
- Blue arcs are represent positives
- Green arcs represent incorrectly directed arcs



