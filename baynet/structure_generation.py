"""Structure generation algorithms."""

import random
from typing import List, Optional, Union

import igraph
import networkx as nx

import numpy as np

from .structure import DAG

__all__ = [
    "forest_fire",
    "barabasi_albert",
    "erdos_renyi",
    "watts_strogatz",
    "ide_cozman",
    "waxman",
]


def forest_fire(
    n_nodes: int,
    *,
    fw_prob: float = 0.2,
    bw_factor: float = 0.4,
    ambs: int = 1,
    seed: Optional[int] = None,
) -> DAG:
    """Create a DAG using the Forest Fire algorithm."""
    if seed is not None:
        random.seed(seed)
    return DAG(
        igraph.Graph.Forest_Fire(n_nodes, fw_prob, bw_factor=bw_factor, ambs=ambs, directed=True)
    )


def barabasi_albert(
    n_nodes: int,
    *,
    outgoing_per_node: Union[int, List[int]] = 1,
    power: float = 0.5,
    seed: Optional[int] = None,
) -> DAG:
    """Create a DAG using the Barabasi-Albert (Preferential Attachment) algorithm."""
    if seed is not None:
        random.seed(seed)
    return DAG(igraph.Graph.Barabasi(n_nodes, m=outgoing_per_node, power=power, directed=True))


def erdos_renyi(n_nodes: int, *, edge_prob: float = 0.5, seed: Optional[int] = None) -> DAG:
    """Create a DAG using the Erdos-Renyi algorithm."""
    if seed is not None:
        random.seed(seed)
    return _make_dag(igraph.Graph.Erdos_Renyi(n_nodes, edge_prob, directed=True))


def watts_strogatz(
    n_nodes: int, *, nei: int = 2, rw_prob: float = 0.25, seed: Optional[int] = None
) -> "DAG":
    """Create a DAG using the Watts-Strogatz (Small World) algorithm."""
    if seed is not None:
        random.seed(seed)
    return _make_dag(igraph.Graph.Watts_Strogatz(dim=1, size=n_nodes, nei=nei, p=rw_prob))


def ide_cozman(
    n_nodes: int,
    *,
    burn_in: int = 10_000,
    max_degree: int = 3,
    max_indegree: int = 3,
    max_outdegree: int = 3,
    seed: Optional[int] = None,
) -> DAG:
    """
    Sample a DAG uniformly at random.

    Uses Ide's and Cozman's Generating Multi-connected DAGs algorithm.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Create a simple tree
    dag = DAG()
    dag.add_vertices(n_nodes)
    dag.name_nodes()
    nodes = sorted(dag.nodes)
    dag.add_edges(list(zip(nodes[1:], nodes[:-1])))
    for _ in range(burn_in):
        dag_copy = dag.copy()
        i, j = np.random.choice(n_nodes, 2, replace=False)
        if (dag.get_node_name(i), dag.get_node_name(j)) in dag.edges:
            dag_copy.delete_edges([(i, j)])
            if dag_copy.as_undirected().is_connected():
                dag = dag_copy
        else:
            node_i = dag.get_node(i)
            node_j = dag.get_node(j)
            if (
                node_j.indegree() == max_indegree
                or node_i.outdegree() == max_outdegree
                or node_j.degree() == max_degree
                or node_i.degree() == max_degree
            ):
                continue
            try:
                dag_copy.add_edge(i, j)
                dag = dag_copy
            except AssertionError:  # added a cycle
                continue
    return dag


def waxman(
    n_nodes: int,
    *,
    alpha: float = 0.4,
    beta: float = 0.2,
    seed: Optional[int] = None,
) -> DAG:
    """Create a Waxman random graph, converted to DAG."""
    if seed is not None:
        random.seed(seed)
    return _make_dag(DAG.from_other(nx.waxman_graph(n=n_nodes, alpha=alpha, beta=beta)))


def _make_dag(graph: igraph.Graph) -> DAG:
    """Make an arbitrary graph (un/directed, a/cyclic) into a DAG."""
    dag = DAG()
    dag.graph = graph
    amat = np.tril(dag.get_numpy_adjacency(skeleton=True), -1)
    return DAG.from_amat(amat)
