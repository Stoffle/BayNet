from time import time
import pickle
from pathlib import Path

import pytest
import networkx as nx
import numpy as np
from igraph import VertexSeq
import yaml

from baynet.structure import DAG, _nodes_sorted, _nodes_from_modelstring, _edges_from_modelstring
from .utils import TEST_MODELSTRING, REVERSED_MODELSTRING, test_dag, partial_dag, temp_out


def test_nodes_sorted():
    nodes = ["a", "B", "aa", 1, 2]
    assert _nodes_sorted(nodes) == ["1", "2", "B", "a", "aa"]


def test_nodes_from_modelstring():
    assert _nodes_from_modelstring(TEST_MODELSTRING) == ["A", "B", "C", "D"]


def test_edges_from_modelstring():
    assert _edges_from_modelstring(TEST_MODELSTRING) == [("C", "B"), ("D", "B"), ("D", "C")]


def test_DAG_from_modelstring():
    dag = test_dag()
    assert dag.nodes == {"A", "B", "C", "D"}
    assert dag.edges == dag.directed_edges == {("C", "B"), ("D", "B"), ("D", "C")}


def test_DAG_from_amat():
    unconnected_amat = np.zeros((4, 4))
    unconnected_graph = DAG.from_amat(unconnected_amat, list("ABCD"))
    unconnected_graph_list = DAG.from_amat(unconnected_amat.tolist(), list("ABCD"))
    fully_connected_amat = np.tril(np.ones((4, 4)), -1)
    fully_connected_graph = DAG.from_amat(fully_connected_amat, list("ABCD"))

    with pytest.raises(ValueError):
        DAG.from_amat(unconnected_amat, ["A", "B", "C"])
    with pytest.raises(ValueError):
        DAG.from_amat(unconnected_amat, "ABCD")

    assert np.all(unconnected_graph.get_numpy_adjacency() == unconnected_amat)
    assert np.all(unconnected_graph_list.get_numpy_adjacency() == unconnected_amat)
    assert np.all(fully_connected_graph.get_numpy_adjacency() == fully_connected_amat)

    assert fully_connected_graph.nodes == unconnected_graph.nodes == {"A", "B", "C", "D"}
    assert unconnected_graph.edges == set()
    assert (
        fully_connected_graph.edges
        == fully_connected_graph.directed_edges
        == {('C', 'A'), ('B', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'A'), ('C', 'B'),}
    )


def test_DAG_from_other():
    test_graph = nx.DiGraph()
    test_graph.add_nodes_from(list("ABCD"))
    edges = [("C", "B"), ("D", "B"), ("D", "C")]
    test_graph.add_edges_from(edges)
    graph = DAG.from_other(test_graph)
    assert graph.edges == graph.directed_edges == set(edges)
    assert graph.nodes == set(list("ABCD"))


def test_DAG_edge_properties():
    dag = test_dag()
    forward = {("C", "B"), ("D", "B"), ("D", "C")}
    backward = {("B", "C"), ("B", "D"), ("C", "D")}
    assert dag.edges == dag.directed_edges == forward
    assert dag.reversed_edges == backward
    assert dag.as_undirected().edges == dag.skeleton_edges == forward | backward


def test_DAG_add_edge():
    dag = test_dag()
    dag.add_edge("B", "A")
    assert dag.edges == {("C", "B"), ("D", "B"), ("D", "C"), ("B", "A")}


def test_DAG_adding_duplicates():
    dag = test_dag()
    with pytest.raises(ValueError):
        dag.add_edge("C", "B")
    with pytest.raises(ValueError):
        dag.add_edges([("C", "B")])
    with pytest.raises(ValueError):
        dag.add_edges([("D", "A"), ("D", "A")])


def test_DAG_get_numpy_adjacency():
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


def test_DAG_get_modelstring():
    assert test_dag().get_modelstring() == TEST_MODELSTRING
    assert test_dag(reverse=True).get_modelstring() == REVERSED_MODELSTRING


def test_DAG_get_ancestors():
    dag = test_dag()
    assert (
        dag.get_ancestors("A")['name']
        == dag.get_ancestors(dag.vs[0])['name']
        == dag.get_ancestors(0)['name']
        == []
    )
    assert (
        dag.get_ancestors("B")['name']
        == dag.get_ancestors(dag.vs[1])['name']
        == dag.get_ancestors(1)['name']
        == ['C', 'D']
    )
    assert (
        dag.get_ancestors("C")['name']
        == dag.get_ancestors(dag.vs[2])['name']
        == dag.get_ancestors(2)['name']
        == ['D']
    )
    assert (
        dag.get_ancestors("D")['name']
        == dag.get_ancestors(dag.vs[3])['name']
        == dag.get_ancestors(3)['name']
        == []
    )


def test_DAG_get_node_name_or_index():
    dag = test_dag()
    for name, index in zip("ABCD", range(4)):
        assert dag.get_node_name(index) == name
        assert dag.get_node_index(name) == index


def test_DAG_are_neighbours():
    dag = test_dag()
    a, b, c, d = dag.vs
    assert not dag.are_neighbours(a, b)
    assert not dag.are_neighbours(a, c)
    assert not dag.are_neighbours(a, d)
    assert dag.are_neighbours(b, c)
    assert dag.are_neighbours(b, d)
    assert dag.are_neighbours(c, d)


def test_DAG_get_v_structures():
    dag = test_dag()
    reversed_dag = test_dag(True)
    assert partial_dag().get_v_structures() == {("C", "B", "D")}
    assert dag.get_v_structures() == set()
    assert dag.get_v_structures(True) == {("C", "B", "D")}
    assert reversed_dag.get_v_structures(True) == {("B", "D", "C")}


def test_DAG_pickling():
    dag = test_dag()
    p = pickle.dumps(dag)
    unpickled_dag = pickle.loads(p)

    assert dag.nodes == unpickled_dag.nodes
    assert dag.edges == unpickled_dag.edges == unpickled_dag.directed_edges


def test_DAG_yaml_continuous_file(temp_out):
    dag_path = temp_out / 'cont.yml'
    dag = test_dag()
    dag.generate_continuous_parameters()
    dag.save(dag_path)
    dag2 = DAG.load(dag_path)
    assert dag.nodes == dag2.nodes
    assert dag.edges == dag2.edges
    assert dag.__dict__['vs'] == dag2.__dict__['vs']


def test_DAG_yaml_continuous_str():
    dag = test_dag()
    dag.generate_continuous_parameters()
    dag_string = dag.save()
    dag2 = DAG.load(dag_string)
    assert dag.nodes == dag2.nodes
    assert dag.edges == dag2.edges
    assert dag.__dict__['vs'] == dag2.__dict__['vs']


def test_DAG_yaml_discrete_file(temp_out):
    dag_path = temp_out / 'cont.yml'
    dag = test_dag()
    dag.generate_discrete_parameters(seed=0)
    dag.save(dag_path)
    dag2 = DAG.load(dag_path)
    assert dag.nodes == dag2.nodes
    assert dag.edges == dag2.edges
    assert dag.__dict__['vs'] == dag2.__dict__['vs']


def test_DAG_yaml_discrete_str():
    dag = test_dag()
    dag.generate_discrete_parameters(seed=0)
    dag_string = dag.save()
    dag2 = DAG.load(dag_string)
    assert dag.nodes == dag2.nodes
    assert dag.edges == dag2.edges
    assert dag.__dict__['vs'] == dag2.__dict__['vs']


def test_DAG_generate_parameters():
    dag = test_dag()
    dag.generate_continuous_parameters(possible_weights=[1], std=0.0)
    for v in dag.vs:
        assert np.allclose(v['CPD'].array, 1)

    for levels in [["0", "1"], ["0", "1", "2"]]:
        dag.vs['levels'] = [levels for v in dag.vs]
        dag.generate_discrete_parameters()
        assert dag.vs[0]['CPD'].array.shape == (len(levels),)
        assert dag.vs[1]['CPD'].array.shape == (len(levels), len(levels), len(levels))
        assert dag.vs[2]['CPD'].array.shape == (len(levels), len(levels))
        assert dag.vs[3]['CPD'].array.shape == (len(levels),)


def test_DAG_sample_continuous():
    dag = test_dag()
    dag.generate_continuous_parameters(std=0.0)
    assert np.allclose(dag.sample(10), 0)

    dag.generate_continuous_parameters(std=1.0)
    assert not np.allclose(dag.sample(10, seed=1), 0)


def test_DAG_sample_discrete():
    dag = test_dag()
    dag.generate_discrete_parameters()
    assert not np.allclose(dag.sample(10, seed=1), 0)


def test_Graph():
    from baynet import Graph

    g = Graph()
