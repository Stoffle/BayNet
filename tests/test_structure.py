from time import time
import pytest
import numpy as np
from igraph import VertexSeq
from baynet.structure import Graph, _nodes_sorted, _nodes_from_modelstring, _edges_from_modelstring
from .utils import TEST_MODELSTRING, test_dag, partial_dag


def test_nodes_sorted():
    nodes = ["a", "B", "aa", 1, 2]
    assert _nodes_sorted(nodes) == ["1", "2", "B", "a", "aa"]


def test_nodes_from_modelstring():
    assert _nodes_from_modelstring(TEST_MODELSTRING) == ["A", "B", "C", "D"]


def test_edges_from_modelstring():
    assert _edges_from_modelstring(TEST_MODELSTRING) == [("C", "B"), ("D", "B"), ("D", "C")]


def test_Graph_undirected():
    with pytest.raises(ValueError):
        Graph(directed=False)


def test_Graph_from_modelstring():
    dag = test_dag()
    assert dag.nodes == {"A", "B", "C", "D"}
    assert dag.edges == {("C", "B"), ("D", "B"), ("D", "C")}


def test_Graph_from_amat():
    unconnected_amat = np.zeros((4, 4))
    unconnected_graph = Graph.from_amat(unconnected_amat, list("ABCD"))
    unconnected_graph_list = Graph.from_amat(unconnected_amat.tolist(), list("ABCD"))
    fully_connected_amat = np.tril(np.ones((4, 4)), -1)
    fully_connected_graph = Graph.from_amat(fully_connected_amat, list("ABCD"))

    with pytest.raises(ValueError):
        Graph.from_amat(unconnected_amat, ["A", "B", "C"])
    with pytest.raises(ValueError):
        Graph.from_amat(unconnected_amat, "ABCD")

    assert np.all(unconnected_graph.get_numpy_adjacency() == unconnected_amat)
    assert np.all(unconnected_graph_list.get_numpy_adjacency() == unconnected_amat)
    assert np.all(fully_connected_graph.get_numpy_adjacency() == fully_connected_amat)

    assert fully_connected_graph.nodes == unconnected_graph.nodes == {"A", "B", "C", "D"}
    assert unconnected_graph.edges == set()
    assert fully_connected_graph.edges == {
        ('C', 'A'),
        ('B', 'A'),
        ('D', 'B'),
        ('D', 'C'),
        ('D', 'A'),
        ('C', 'B'),
    }


def test_Graph_edge_properties():
    dag = test_dag()
    forward = {("C", "B"), ("D", "B"), ("D", "C")}
    backward = {("B", "C"), ("B", "D"), ("C", "D")}
    assert dag.edges == dag.directed_edges == forward
    assert dag.reversed_edges == backward
    assert dag.as_undirected().edges == dag.skeleton_edges == forward | backward


def test_Graph_add_edge():
    dag = test_dag()
    dag.add_edge("B", "A")
    assert dag.edges == {("C", "B"), ("D", "B"), ("D", "C"), ("B", "A")}


def test_Graph_adding_duplicates():
    dag = test_dag()
    with pytest.raises(ValueError):
        dag.add_edge("C", "B")
    with pytest.raises(ValueError):
        dag.add_edges([("C", "B")])
    with pytest.raises(ValueError):
        dag.add_edges([("D", "A"), ("D", "A")])


def test_Graph_get_numpy_adjacency():
    dag = test_dag()
    amat = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
        ],
        dtype=bool,
    )
    assert np.all(dag.get_numpy_adjacency() == amat)
    assert np.all(dag.get_numpy_adjacency(skeleton=True) == amat | amat.T)


def test_Graph_get_ancestors():
    dag = test_dag()
    assert dag.get_ancestors("A")['name'] == dag.get_ancestors(0)['name'] == []
    assert dag.get_ancestors("B")['name'] == dag.get_ancestors(1)['name'] == ['C', 'D']
    assert dag.get_ancestors("C")['name'] == dag.get_ancestors(2)['name'] == ['D']
    assert dag.get_ancestors("D")['name'] == dag.get_ancestors(3)['name'] == []


def test_Graph_get_node_name_or_index():
    dag = test_dag()
    for name, index in zip("ABCD", range(4)):
        assert dag.get_node_name(index) == name
        assert dag.get_node_index(name) == index


def test_Graph_are_neighbours():
    dag = test_dag()
    a, b, c, d = dag.vs
    assert not dag.are_neighbours(a, b)
    assert not dag.are_neighbours(a, c)
    assert not dag.are_neighbours(a, d)
    assert dag.are_neighbours(b, c)
    assert dag.are_neighbours(b, d)
    assert dag.are_neighbours(c, d)


def test_Graph_get_v_structures():
    dag = test_dag()
    part_dag = partial_dag()
    reversed_dag = test_dag(True)
    assert partial_dag().get_v_structures() == {("C", "B", "D")}
    assert dag.get_v_structures() == set()
    assert dag.get_v_structures(True) == {("C", "B", "D")}
    assert reversed_dag.get_v_structures(True) == {("B", "D", "C")}
