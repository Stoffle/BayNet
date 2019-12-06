from typing import List, Union, Tuple, Set
import warnings
import igraph
import numpy as np
from numba import njit

def _nodes_sorted(nodes: List[Union[int, str]]) -> List[Union[int, str]]:
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
            edges.append(
                (parent, node)
            )
    return edges

class Graph(igraph.Graph):
    # pylint: disable=unsubscriptable-object, not-an-iterable
    def __init__(self, *args, **kwargs):
        if 'directed' not in kwargs:
            kwargs['directed'] = True
        elif kwargs['directed'] == False:
            raise ValueError("Graph() can only be used with directed=True")
        kwargs['vertex_attrs'] = {'CPD': None, 'levels': None}
        super().__init__(*args, **kwargs)
        self.metrics = {}

    @classmethod
    def from_modelstring(cls, modelstring: str):
        """Instantiate a Graph object from a modelstring"""
        dag = cls()
        dag.add_vertices(_nodes_from_modelstring(modelstring))
        dag.add_edges(_edges_from_modelstring(modelstring))
        return dag

    @property
    def nodes(self) -> Set[str]:
        """Returns a set of the names of all nodes in the network"""
        return {v['name'] for v in self.vs}

    @property
    def edges(self) -> Set[Tuple[str, str]]:
        """Returns all edges in the Graph"""
        if self.is_directed():
            return self.directed_edges
        else:
            return self.skeleton_edges

    @property
    def skeleton_edges(self) -> Set[Tuple[str, str]]:
        """Returns all edges in the skeleton of the Graph"""
        return self.reversed_edges | self.directed_edges

    @property
    def directed_edges(self) -> Set[Tuple[str, str]]:
        """Returns forward edges in the Graph"""
        return {(self.vs[e.source]['name'], self.vs[e.target]['name']) for e in self.es}

    @property
    def reversed_edges(self) -> Set[Tuple[str, str]]:
        """Returns reversed edges in the Graph"""
        return {(self.vs[e.target]['name'], self.vs[e.source]['name']) for e in self.es}

    def get_node_name(self, node: int) -> str:
        """Converts node index to node name"""
        return self.vs[node]['name']
    
    def get_node_index(self, node: str) -> int:
        """Converts node name to node index"""
        return self.vs['name'].index(node)

    def add_edge(self, source: str, target: str):
        """
        Add a single edge, using node names (as strings)
        Overrides: igraph.Graph.add_edge
        """
        if (source, target) in self.edges:
            raise ValueError(f"Edge {source}->{target} already exists in Graph")
        super().add_edge(source, target)

    def add_edges(self, edges: List[Tuple[str, str]]):
        """Add multiple edges from a list of tuples, each containing (from, to) as strings"""
        for source, target in edges:
            if (source, target) in self.edges:
                raise ValueError(f"Edge {source}->{target} already exists in Graph")
            if len(edges) != len(set(edges)):
                raise ValueError("Edges list contains duplicates")
        super().add_edges(edges)

    def get_numpy_adjacency(self, skeleton: bool = False) -> np.ndarray:
        """Obtain adjacency matrix as a numpy (boolean) array"""
        if skeleton:
            return self.as_undirected().get_numpy_adjacency()
        else:
            return np.array(list(self.get_adjacency()), dtype=bool)

    def get_ancestors(self, node: Union[str, int]) -> igraph.VertexSeq:
        """Return an igraph.VertexSeq of ancestors for given node (string or node index)"""
        if isinstance(node, str):
            # Convert name to index
            node = self.get_node_index(node)
        ancestors = list(self.neighborhood(vertices=node, order=len(self.vs), mode="IN"))
        ancestors.remove(node)
        if len(ancestors) <= 1:
            return igraph.VertexSeq(self, ancestors)
        else:
            return self.vs[sorted(ancestors)]





