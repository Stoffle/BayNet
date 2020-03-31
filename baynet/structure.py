"""Graph object."""
from __future__ import annotations
from itertools import combinations
from typing import List, Union, Tuple, Set, Any, Dict, Optional
from pathlib import Path

import igraph
import numpy as np
from yaml import safe_dump, safe_load

from . import parameters
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
    def __init__(self, *args: None, **kwargs: Any) -> None:
        """Create a graph object."""
        # Grab *args and **kwargs because pickle/igraph do weird things here
        super().__init__(directed=True, vertex_attrs={'CPD': None})
        if 'name' in kwargs.keys():
            self.name = kwargs['name']
        else:
            self.name = "unnamed"

    @property
    def __dict__(self) -> Dict:
        """Return dict of attributes needed for pickling."""
        if self.vs['CPD'] == [None for _ in self.vs]:
            return {
                'name': self.name,
                'vs': [{'name': v['name']} for v in self.vs],
                'edges': list(self.edges),
            }
        return {
            'name': self.name,
            'vs': [
                {'name': v['name'], 'CPD': v['CPD'].to_dict(), 'type': type(v['CPD']).__name__}
                for v in self.vs
            ],
            'edges': list(self.edges),
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set new instance's state from a dict."""
        for vertex in state['vs']:
            if 'CPD' in vertex.keys():
                cpd = getattr(parameters, vertex['type']).from_dict(**vertex['CPD'])
                self.add_vertex(name=vertex['name'], CPD=cpd)
            else:
                self.add_vertex(name=vertex['name'])
        self.add_edges([(node_from, node_to) for node_from, node_to in state.get('edges', [])])
        self.name = state['name']

    @classmethod
    def from_modelstring(cls, modelstring: str, **kwargs: Dict[str, Any]) -> 'DAG':
        """Instantiate a Graph object from a modelstring."""
        dag = cls(**kwargs)
        dag.add_vertices(_nodes_from_modelstring(modelstring))
        dag.add_edges(_edges_from_modelstring(modelstring))
        return dag

    @classmethod
    def from_amat(
        cls, amat: Union[np.ndarray, List[List[int]]], colnames: List[str], **kwargs: Dict[str, Any]
    ) -> 'DAG':
        """Instantiate a Graph object from an adjacency matrix."""
        if isinstance(amat, np.ndarray):
            amat = amat.tolist()
        if not len(colnames) == len(amat):
            raise ValueError("Dimensions of amat and colnames do not match")
        if not isinstance(colnames, list):
            raise ValueError(
                f"Graph.from_amat() expected `colnames` of type list, but got {type(colnames)}"
            )
        dag = cls.Adjacency(amat, **kwargs)
        dag.vs['name'] = colnames
        return dag

    @classmethod
    def from_other(cls, other_graph: Any, **kwargs: Dict[str, Any]) -> 'DAG':
        """Attempt to create a Graph from an existing graph object (nx.DiGraph etc.)."""
        graph = cls(**kwargs)
        graph.add_vertices(_nodes_sorted(other_graph.nodes))
        graph.add_edges(other_graph.edges)
        return graph

    @property
    def nodes(self) -> Set[str]:
        """Return a set of the names of all nodes in the network."""
        return {v['name'] for v in self.vs}

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
        return {(self.vs[e.source]['name'], self.vs[e.target]['name']) for e in self.es}

    @property
    def reversed_edges(self) -> Set[Tuple[str, str]]:
        """Return reversed edges in the Graph."""
        return {(self.vs[e.target]['name'], self.vs[e.source]['name']) for e in self.es}

    def get_node_name(self, node: int) -> str:
        """Convert node index to node name."""
        return self.vs[node]['name']

    def get_node_index(self, node: str) -> int:
        """Convert node name to node index."""
        return self.vs['name'].index(node)

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
                [v['name'] for v in self.get_ancestors(node, only_parents=True)]
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
                node_v_structures = [(a['name'], node, b['name']) for a, b in all_pairs]
            else:
                node_v_structures = [
                    (a['name'], node, b['name'])
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
            vertex['CPD'] = ConditionalProbabilityDistribution(vertex, mean=mean, std=std)
            vertex['CPD'].sample_parameters(weights=possible_weights, seed=seed)

    def generate_levels(
        self,
        cardinality_min: Optional[int] = None,
        cardinality_max: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Set number of levels in each node, for generating discrete data."""
        if seed is not None:
            np.random.seed(seed)
        if cardinality_min is None:
            cardinality_min = 2
        if cardinality_max is None:
            cardinality_max = 3
        assert cardinality_max >= cardinality_min >= 2
        for vertex in self.vs:
            n_levels = np.random.randint(cardinality_min, cardinality_max + 1)
            vertex['levels'] = list(map(str, range(n_levels)))

    def generate_discrete_parameters(
        self,
        alpha: Optional[float] = None,
        cardinality_min: Optional[int] = None,
        cardinality_max: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Populate discrete conditional parameter tables for each node."""
        try:
            self.vs['levels']
        except KeyError:
            self.generate_levels(cardinality_min, cardinality_max, seed)
        for vertex in self.vs:
            vertex['CPD'] = ConditionalProbabilityTable(vertex)
            vertex['CPD'].sample_parameters(alpha=alpha, seed=seed)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample n_samples rows of data from the graph."""
        if seed is not None:
            np.random.seed(seed)
        sorted_nodes = self.topological_sorting(mode="out")
        data = np.zeros((n_samples, len(self.nodes)))
        for node_idx in sorted_nodes:
            data[:, node_idx] = self.vs[node_idx]['CPD'].sample(data)
        return data

    def save(self, yaml_path: Optional[Path] = None) -> Optional[str]:
        """Save DAG as yaml file, or string if no path is specified."""
        if yaml_path is None:
            return safe_dump(self.__dict__)
        with yaml_path.open('w') as stream:
            return safe_dump(self.__dict__, stream=stream)

    @classmethod
    def load(cls, yaml: Union[Path, str]) -> DAG:
        """Load DAG from yaml file or string."""
        if isinstance(yaml, Path):
            with yaml.open('r') as stream:
                yaml_str = stream.read()
        else:
            yaml_str = yaml
        state = safe_load(yaml_str)
        dag = cls()
        dag.__setstate__(state)
        return dag
