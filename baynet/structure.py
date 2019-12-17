"""Graph object."""
from __future__ import annotations
from itertools import combinations
from typing import List, Union, Tuple, Set
import igraph
import numpy as np


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


class Graph(igraph.Graph):
    """Graph object, built around igraph.Graph, adapted for bayesian networks."""

    # pylint: disable=unsubscriptable-object, not-an-iterable, arguments-differ
    def __init__(self, *args, **kwargs) -> None:
        """Create a graph object."""
        if 'directed' not in kwargs:
            kwargs['directed'] = True
        elif not kwargs['directed']:
            raise ValueError("Graph() can only be used with directed=True")
        kwargs['vertex_attrs'] = {'CPD': None, 'levels': None}
        super().__init__(*args, **kwargs)

    @classmethod
    def from_modelstring(cls, modelstring: str) -> Graph:
        """Instantiate a Graph object from a modelstring."""
        dag = cls()
        dag.add_vertices(_nodes_from_modelstring(modelstring))
        dag.add_edges(_edges_from_modelstring(modelstring))
        return dag

    @classmethod
    def from_amat(cls, amat: Union[np.ndarray, List[List[int]]], colnames: List[str]) -> Graph:
        """Instantiate a Graph object from an adjacency matrix."""
        if isinstance(amat, np.ndarray):
            amat = amat.tolist()
        if not len(colnames) == len(amat):
            raise ValueError("Dimensions of amat and colnames do not match")
        if not isinstance(colnames, list):
            raise ValueError(
                f"Graph.from_amat() expected `colnames` of type list, but got {type(colnames)}"
            )
        dag = cls.Adjacency(amat)
        dag.vs['name'] = colnames
        # dag.add_vertices(_nodes_sorted(colnames))
        return dag

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
            return self.as_undirected().get_numpy_adjacency()
        return np.array(list(self.get_adjacency()), dtype=bool)

    def get_ancestors(self, node: Union[str, int], only_parents: bool = False) -> igraph.VertexSeq:
        """Return an igraph.VertexSeq of ancestors for given node (string or node index)."""
        if isinstance(node, str):
            # Convert name to index
            node = self.get_node_index(node)
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
        # from functools import partial
        # self.v_structures = []
        # self.motifs_randesu(callback=partial(motif_callback, include_shielded))
        # return self.v_structures


# def motif_callback(
#   include_shielded: bool,
#   graph: Graph, vertices: List[int],
#   isomorphy_class: int
#   ):
#     if (include_shielded and isomorphy_class == 7) or (isomorphy_class == 2):
#         graph.v_structures.append(tuple(graph.get_node_name(v_idx) for v_idx in vertices))
