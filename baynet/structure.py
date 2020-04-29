"""Graph object."""
from __future__ import annotations
from itertools import combinations
from typing import List, Union, Tuple, Set, Any, Optional, Type, Dict
from pathlib import Path
from copy import deepcopy

import igraph
import numpy as np
import pandas as pd
from baynet.utils import dag_io, visualisation

from .parameters import ConditionalProbabilityDistribution, ConditionalProbabilityTable


def _nodes_sorted(nodes: Union[List[int], List[str], List[object]]) -> List[str]:
    return sorted([str(node) for node in nodes])


def _nodes_with_parents(modelstring: str) -> List[str]:
    return modelstring.strip().strip("[]]").split("][")


def _nodes_from_modelstring(modelstring: str) -> List[str]:
    nodes = [node.split("|")[0] for node in _nodes_with_parents(modelstring)]
    return _nodes_sorted(nodes)


def _edges_from_modelstring(modelstring: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for node_and_parents in _nodes_with_parents(modelstring):
        try:
            node, parents = node_and_parents.split("|")
        except ValueError:
            continue
        for parent in parents.split(":"):
            edges.append((parent, node))
    return edges


class DAG(igraph.Graph):
    """Directed Acyclic Graph object, built around igraph.Graph, adapted for bayesian networks."""

    # pylint: disable=unsubscriptable-object, not-an-iterable, arguments-differ
    def __init__(self, buf: Optional[bytes] = None) -> None:
        """Create a graph object."""
        super().__init__(directed=True, vertex_attrs={"CPD": None})
        if buf is not None:
            dag_io.buf_to_dag(buf, dag=self)

    def __reduce__(self) -> Tuple:
        """Return representation for Pickle."""
        return self.__class__, (self.save(),)

    @classmethod
    def from_modelstring(cls, modelstring: str) -> "DAG":
        """Instantiate a Graph object from a modelstring."""
        dag = cls()
        dag.add_vertices(_nodes_from_modelstring(modelstring))
        dag.add_edges(_edges_from_modelstring(modelstring))
        return dag

    @classmethod
    def from_amat(cls, amat: Union[np.ndarray, List[List[int]]], colnames: List[str]) -> "DAG":
        """Instantiate a Graph object from an adjacency matrix."""
        if isinstance(amat, np.ndarray):
            amat = amat.tolist()
        if not len(colnames) == len(amat):
            raise ValueError("Dimensions of amat and colnames do not match")
        if not isinstance(colnames, list):
            raise ValueError(
                f"Graph.from_amat() expected `colnames` of type list, but got {type(colnames)}"
            )
        dag = cls()
        dag.add_vertices(colnames)
        dag.add_edges(
            [
                (str(colnames[parent_idx]), str(colnames[target_idx]))
                for parent_idx, row in enumerate(amat)
                for target_idx, val in enumerate(row)
                if val
            ]
        )
        return dag

    @classmethod
    def from_other(cls, other_graph: Any) -> "DAG":
        """Attempt to create a Graph from an existing graph object (nx.DiGraph etc.)."""
        graph = cls()
        graph.add_vertices(_nodes_sorted(other_graph.nodes))
        graph.add_edges(other_graph.edges)
        return graph

    @property
    def dtype(self) -> Optional[str]:
        """Return data type of parameterised network."""
        if all(isinstance(vertex["CPD"], ConditionalProbabilityTable) for vertex in self.vs):
            return "discrete"
        elif all(
            isinstance(vertex["CPD"], ConditionalProbabilityDistribution) for vertex in self.vs
        ):
            return "continuous"
        elif all(
            type(vertex["CPD"]) in [ConditionalProbabilityTable, ConditionalProbabilityDistribution]
            for vertex in self.vs
        ):
            return "mixed"
        return None

    @property
    def nodes(self) -> Set[str]:
        """Return a set of the names of all nodes in the network."""
        return {v["name"] for v in self.vs}

    @property
    def edges(self) -> Set[Tuple[str, str]]:
        """Return all edges in the Graph."""
        if self.is_directed():
            return self.directed_edges
        return self.skeleton_edges

    @property
    def skeleton_edges(self) -> Set[Tuple[str, str]]:
        """Return all edges in the skeleton of the Graph."""
        return self.reversed_edges | self.directed_edges

    @property
    def directed_edges(self) -> Set[Tuple[str, str]]:
        """Return forward edges in the Graph."""
        return {(self.vs[e.source]["name"], self.vs[e.target]["name"]) for e in self.es}

    @property
    def reversed_edges(self) -> Set[Tuple[str, str]]:
        """Return reversed edges in the Graph."""
        return {(self.vs[e.target]["name"], self.vs[e.source]["name"]) for e in self.es}

    def get_node_name(self, node: int) -> str:
        """Convert node index to node name."""
        return self.vs[node]["name"]

    def get_node_index(self, node: str) -> int:
        """Convert node name to node index."""
        return self.vs["name"].index(node)

    def get_node(self, name: str) -> igraph.Vertex:
        """Get Vertex object by node name."""
        try:
            return self.vs[self.get_node_index(name)]
        except ValueError:
            raise KeyError(f"Node `{name}` does not appear in DAG.")

    def add_edge(self, source: str, target: str) -> None:
        """
        Add a single edge, using node names (as strings).

        Overrides: igraph.Graph.add_edge
        """
        if (source, target) in self.edges:
            raise ValueError(f"Edge {source}->{target} already exists in Graph")
        super().add_edge(source, target)
        assert self.is_dag()

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        """Add multiple edges from a list of tuples, each containing (from, to) as strings."""
        for source, target in edges:
            if (source, target) in self.edges:
                raise ValueError(f"Edge {source}->{target} already exists in Graph")
        if len(edges) != len(set(edges)):
            raise ValueError("Edges list contains duplicates")
        super().add_edges(edges)
        assert self.is_dag()

    def get_numpy_adjacency(self, skeleton: bool = False) -> np.ndarray:
        """Obtain adjacency matrix as a numpy (boolean) array."""
        if skeleton:
            amat = self.get_numpy_adjacency()
            return amat | amat.T
        return np.array(list(self.get_adjacency()), dtype=bool)

    def get_modelstring(self) -> str:
        """Obtain modelstring representation of stored graph."""
        modelstring = ""
        for node in _nodes_sorted(list(self.nodes)):
            parents = _nodes_sorted(
                [v["name"] for v in self.get_ancestors(node, only_parents=True)]
            )
            modelstring += f"[{node}"
            modelstring += f"|{':'.join(parents)}" if parents else ""
            modelstring += "]"
        return modelstring

    def get_ancestors(
        self, node: Union[str, int, igraph.Vertex], only_parents: bool = False
    ) -> igraph.VertexSeq:
        """Return an igraph.VertexSeq of ancestors for given node (string or node index)."""
        if isinstance(node, str):
            # Convert name to index
            node = self.get_node_index(node)
        elif isinstance(node, igraph.Vertex):
            node = node.index
        order = 1 if only_parents else len(self.vs)
        ancestors = list(self.neighborhood(vertices=node, order=order, mode="IN"))
        ancestors.remove(node)
        if len(ancestors) <= 1:
            return igraph.VertexSeq(self, ancestors)
        return self.vs[sorted(ancestors)]

    def get_descendants(
        self, node: Union[str, int, igraph.Vertex], only_children: bool = False
    ) -> igraph.VertexSeq:
        """Return an igraph.VertexSeq of descendants for given node (string or node index)."""
        if isinstance(node, str):
            # Convert name to index
            node = self.get_node_index(node)
        elif isinstance(node, igraph.Vertex):
            node = node.index
        order = 1 if only_children else len(self.vs)
        ancestors = list(self.neighborhood(vertices=node, order=order, mode="OUT"))
        ancestors.remove(node)
        if len(ancestors) <= 1:
            return igraph.VertexSeq(self, ancestors)
        return self.vs[sorted(ancestors)]

    def are_neighbours(self, node_a: igraph.Vertex, node_b: igraph.Vertex) -> bool:
        """Check if two nodes are neighbours in the Graph."""
        return node_a.index in self.neighborhood(vertices=node_b)

    def get_v_structures(self, include_shielded: bool = False) -> Set[Tuple[str, str, str]]:
        """Return a list of the Graph's v-structures in tuple form; (a,b,c) = a->b<-c."""
        v_structures: List[Tuple[str, str, str]] = []
        for node in self.nodes:
            all_parents = self.get_ancestors(node, only_parents=True)
            all_pairs = combinations(all_parents, 2)
            all_pairs = [sorted(pair) for pair in all_pairs]
            if include_shielded:
                node_v_structures = [(a["name"], node, b["name"]) for a, b in all_pairs]
            else:
                node_v_structures = [
                    (a["name"], node, b["name"])
                    for a, b in all_pairs
                    if not self.are_neighbours(a, b)
                ]
            v_structures += node_v_structures
        return set(v_structures)

    def generate_continuous_parameters(
        self,
        possible_weights: Optional[List[float]] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Populate continuous conditional distributions for each node."""
        for vertex in self.vs:
            vertex["CPD"] = ConditionalProbabilityDistribution(vertex, mean=mean, std=std)
            vertex["CPD"].sample_parameters(weights=possible_weights, seed=seed)

    def generate_levels(
        self,
        min_levels: Optional[int] = None,
        max_levels: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Set number of levels in each node, for generating discrete data."""
        if seed is not None:
            np.random.seed(seed)
        if min_levels is None:
            min_levels = 2
        if max_levels is None:
            max_levels = 3
        assert max_levels >= min_levels >= 2
        for vertex in self.vs:
            n_levels = np.random.randint(min_levels, max_levels + 1)
            vertex["levels"] = list(map(str, range(n_levels)))

    def generate_discrete_parameters(
        self,
        alpha: Optional[float] = None,
        min_levels: Optional[int] = None,
        max_levels: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Populate discrete conditional parameter tables for each node."""
        try:
            self.vs["levels"]
        except KeyError:
            self.generate_levels(min_levels, max_levels, seed)
        for vertex in self.vs:
            vertex["CPD"] = ConditionalProbabilityTable(vertex)
            vertex["CPD"].sample_parameters(alpha=alpha, seed=seed)

    def estimate_parameters(
        self,
        data: pd.DataFrame,
        method: str = "mle",
        infer_levels: bool = False,
        method_args: Optional[Dict[str, Union[int, float]]] = None,
    ) -> None:
        """Estimate conditional probabilities based on supplied data."""
        try:
            if all(vertex["levels"] is None for vertex in self.vs):
                raise KeyError
        except KeyError:
            if not infer_levels:
                raise ValueError(
                    "`estimate_parameters()` requires levels be defined or `infer_levels=True`"
                )
            for vertex in self.vs:
                vertex["levels"] = sorted(data[vertex["name"]].unique().astype(str))
        for vertex in self.vs:
            vertex["CPD"] = ConditionalProbabilityTable.estimate(
                vertex, data=data, method=method, method_args=method_args
            )

    def sample(self, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Sample n_samples rows of data from the graph."""
        if seed is not None:
            np.random.seed(seed)
        sorted_nodes = self.topological_sorting(mode="out")
        dtype: Type
        if all(isinstance(vertex["CPD"], ConditionalProbabilityTable) for vertex in self.vs):
            dtype = int
        elif all(
            isinstance(vertex["CPD"], ConditionalProbabilityDistribution) for vertex in self.vs
        ):
            dtype = float
        else:
            raise RuntimeError("DAG requires parameters before sampling is possible.")
        data = pd.DataFrame(
            np.zeros((n_samples, len(self.nodes))).astype(dtype), columns=self.vs["name"],
        )
        for node_idx in sorted_nodes:
            data.iloc[:, node_idx] = self.vs[node_idx]["CPD"].sample(data)
        data = pd.DataFrame(data, columns=[vertex["name"] for vertex in self.vs])
        return data

    def save(self, buf_path: Optional[Path] = None) -> bytes:
        """Save DAG as protobuf, or string if no path is specified."""
        dag_proto = dag_io.dag_to_buf(self)
        if buf_path is not None:
            with buf_path.open("wb") as stream:
                stream.write(dag_proto)
        return dag_proto

    @classmethod
    def load(cls, buf: Union[Path, bytes]) -> "DAG":
        """Load DAG from yaml file or string."""
        if isinstance(buf, Path):
            with buf.open("rb") as stream:
                buf_str = stream.read()
        else:
            buf_str = buf
        return dag_io.buf_to_dag(buf_str)

    def remove_node(self, node: str) -> None:
        """Remove a node (inplace), marginalising it out of any children's CPTs."""
        assert node in self.nodes
        assert isinstance(self.get_node(node)["CPD"], ConditionalProbabilityTable)
        for vertex in self.get_descendants(node, only_children=True):
            assert isinstance(vertex["CPD"], ConditionalProbabilityTable)
            vertex["CPD"].marginalise(node)
        self.delete_vertices([node])

    def remove_nodes(self, nodes: Union[List[str], igraph.VertexSeq]) -> None:
        """Remove multiple nodes (inplace), marginalising it out of any children's CPTs."""
        if isinstance(nodes, igraph.VertexSeq):
            nodes = [node["name"] for node in nodes]
        for node in nodes:
            self.remove_node(node)

    def mutilate(self, node: str, evidence_level: str) -> "DAG":
        """Return a copy with node's value fixed at evidence_level, and parents killed off."""
        assert node in self.nodes
        mutilated_dag = self.copy()
        mutilated_dag.remove_nodes(mutilated_dag.get_ancestors(node, only_parents=True))
        mutilated_dag.get_node(node)["CPD"].intervene(evidence_level)
        return mutilated_dag

    def copy(self) -> "DAG":
        """Return a copy."""
        self_copy = super().copy()
        for vertex in self_copy.vs:
            vertex["CPD"] = deepcopy(vertex["CPD"])
        return self_copy

    def plot(self, path: Path = Path().resolve() / 'DAG.png') -> None:
        """Save a plot of the DAG to specified file path."""
        dag = self.copy()
        dag.vs['label'] = dag.vs['name']
        dag.vs['fontsize'] = 30
        dag.vs['fontname'] = "Helvetica"
        dag.es['color'] = "black"
        dag.es['penwidth'] = 2
        dag.es['style'] = "solid"
        visualisation.draw_graph(dag, save_path=path)

    def compare(self, other_graph: DAG) -> visualisation.GraphComparison:
        """Produce comparison to another DAG for plotting."""
        return visualisation.GraphComparison(self, other_graph, list(self.vs['name']))
