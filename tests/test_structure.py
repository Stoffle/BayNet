from time import time
import pytest
import numpy as np
from igraph import VertexSeq
from baynet.structure import Graph, _nodes_sorted, _nodes_from_modelstring, _edges_from_modelstring

TEST_MODELSTRING = "[A][B|C:D][C|D][D]"

def test_nodes_sorted():
    nodes = ["a", "B", "aa", 1, 2]
    assert _nodes_sorted(nodes) == ["1", "2", "B", "a", "aa"]

def test_nodes_from_modelstring():
    assert _nodes_from_modelstring(TEST_MODELSTRING) == ["A", "B", "C", "D"]

def test_edges_from_modelstring():
    assert _edges_from_modelstring(TEST_MODELSTRING) == [("C", "B"), ("D", "B"), ("D", "C")]

def test_Graph_from_modelstring():
    dag = Graph.from_modelstring(TEST_MODELSTRING)
    assert dag.nodes == {"A", "B", "C", "D"}
    assert dag.edges == {("C", "B"), ("D", "B"), ("D", "C")}

def test_Graph_edge_properties():
    dag = Graph.from_modelstring(TEST_MODELSTRING)
    forward = {("C", "B"), ("D", "B"), ("D", "C")}
    backward = {("B", "C"), ("B", "D"), ("C", "D")}
    assert dag.edges == dag.directed_edges == forward
    assert dag.reversed_edges == backward
    assert dag.as_undirected().edges == dag.skeleton_edges == forward | backward

def test_Graph_adding_duplicates():
    dag = Graph.from_modelstring(TEST_MODELSTRING)
    with pytest.raises(ValueError):
        dag.add_edge("C", "B")
    with pytest.raises(ValueError):
        dag.add_edges([("C", "B")])
    with pytest.raises(ValueError):
        dag.add_edges([
            ("D", "A"),
            ("D", "A")
            ])

def test_Graph_get_numpy_adjacency():
    dag = Graph.from_modelstring(TEST_MODELSTRING)
    assert np.all(dag.get_numpy_adjacency() == np.array([
        [False, False, False, False],
        [False, False, False, False],
        [False,  True, False, False],
        [False,  True,  True, False]
        ], dtype=bool))

def test_Graph_get_ancestors():
    dag = Graph.from_modelstring(TEST_MODELSTRING)
    assert dag.get_ancestors("A")['name'] == dag.get_ancestors(0)['name'] == []
    assert dag.get_ancestors("B")['name'] == dag.get_ancestors(1)['name'] == ['C', 'D']
    assert dag.get_ancestors("C")['name'] == dag.get_ancestors(2)['name'] == ['D']
    assert dag.get_ancestors("D")['name'] == dag.get_ancestors(3)['name'] == []

def test_Graph_get_node_name_or_index():
    dag = Graph.from_modelstring(TEST_MODELSTRING)
    for name, index in zip("ABCD", range(4)):
        assert dag.get_node_name(index) == name
        assert dag.get_node_index(name) == index


