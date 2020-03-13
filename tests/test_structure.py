from time import time
import pickle
from pathlib import Path

import pytest
import networkx as nx
import numpy as np
from igraph import VertexSeq

from baynet.structure import Graph, _nodes_sorted, _nodes_from_modelstring, _edges_from_modelstring
from .utils import TEST_MODELSTRING, REVERSED_MODELSTRING, test_dag, partial_dag


def test_nodes_sorted():
    nodes = ["a", "B", "aa", 1, 2]
    assert _nodes_sorted(nodes) == ["1", "2", "B", "a", "aa"]


def test_nodes_from_modelstring():
    assert _nodes_from_modelstring(TEST_MODELSTRING) == ["A", "B", "C", "D"]


def test_edges_from_modelstring():
    assert _edges_from_modelstring(TEST_MODELSTRING) == [("C", "B"), ("D", "B"), ("D", "C")]


def test_Graph_undirected():
    # Graph.__init__() no longer accepts keywords, check that remains the case
    with pytest.raises(TypeError):
        Graph(directed=False)


def test_Graph_from_modelstring():
    dag = test_dag()
    assert dag.nodes == {"A", "B", "C", "D"}
    assert dag.edges == dag.directed_edges == {("C", "B"), ("D", "B"), ("D", "C")}


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
    assert (
        fully_connected_graph.edges
        == fully_connected_graph.directed_edges
        == {('C', 'A'), ('B', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'A'), ('C', 'B'),}
    )


def test_Graph_from_other():
    test_graph = nx.DiGraph()
    test_graph.add_nodes_from(list("ABCD"))
    edges = [("C", "B"), ("D", "B"), ("D", "C")]
    test_graph.add_edges_from(edges)
    graph = Graph.from_other(test_graph)
    assert graph.edges == graph.directed_edges == set(edges)
    assert graph.nodes == set(list("ABCD"))


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


def test_Graph_get_modelstring():
    assert test_dag().get_modelstring() == TEST_MODELSTRING
    assert test_dag(reversed=True).get_modelstring() == REVERSED_MODELSTRING


def test_Graph_get_ancestors():
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


def test_Graph_pickling():
    dag = test_dag()
    state = dag.__dict__
    dag_from_state = Graph()
    dag_from_state.__setstate__(state)
    p = pickle.dumps(dag)
    unpickled_dag = pickle.loads(p)

    assert dag.nodes == dag_from_state.nodes
    assert dag.edges == dag_from_state.edges == dag_from_state.directed_edges
    assert dag.nodes == unpickled_dag.nodes
    assert dag.edges == unpickled_dag.edges == unpickled_dag.directed_edges


def test_Graph_generate_parameters():
    dag = test_dag()
    dag.generate_parameters(data_type='cont', possible_weights=[1], noise_scale=0.0)
    for v in dag.vs:
        assert np.allclose(v['CPD']._array, 1)

    with pytest.raises(NotImplementedError):
        dag.generate_parameters(data_type='disc')


def test_Graph_sample():
    dag = test_dag()
    dag.generate_parameters(data_type='cont', noise_scale=0.0)
    assert np.allclose(dag.sample(10), 0)

    dag.generate_parameters(data_type='cont', noise_scale=1.0)
    assert not np.allclose(dag.sample(10), 0)


def test_Graph_to_bif():
    dag = test_dag()
    assert (
        dag.to_bif()
        == """network unknown {
}
variable A {
  type continuous;
  }
variable B {
  type continuous;
  }
variable C {
  type continuous;
  }
variable D {
  type continuous;
  }
"""
    )

    dag = test_dag()
    dag.generate_parameters(data_type='cont', possible_weights=[2], noise_scale=0.0)
    dag.name = 'test_dag'
    assert (
        dag.to_bif()
        == """network test_dag {
}
variable A {
  type continuous;
  }
variable B {
  type continuous;
  }
variable C {
  type continuous;
  }
variable D {
  type continuous;
  }
probability ( B | C, D ) {
  table 2, 2 ;
  }
probability ( C | D ) {
  table 2 ;
  }
"""
    )

    test_path = Path(__file__).parent.resolve() / 'test_graph.bif'
    dag.to_bif(filepath = test_path)
    import time
    assert test_path.read_text() == dag.to_bif()
    test_path.unlink()

    with pytest.raises(NotImplementedError):
        dag = test_dag()
        dag.generate_parameters(data_type='discrete')
        assert (
            dag.to_bif()
            == """
    
        """
        )
