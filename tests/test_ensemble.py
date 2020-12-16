from tests.test_structure import test_Graph
from baynet.ensemble import GraphEnsemble
from baynet import DAG
import pytest
import numpy as np


def test_GraphEnsemble_init(test_dag):
    dags = [test_dag] * 9
    dags.append(DAG.from_modelstring("[A][E|D][D|C][C|B][B|A]"))

    ensemble = GraphEnsemble(dags)
    graph = ensemble.generate_graph()
    assert set(graph.vs['name']) == set("ABCDE")
    assert np.allclose(graph.es['penwidth'], [1.825] * 3 + [0.425] * 4)


def test_plot(temp_out, test_dag):
    ensemble = GraphEnsemble()
    ensemble.add_dag(test_dag)
    img_path = temp_out / 'ensemble.png'
    ensemble.plot(img_path)
    assert img_path.exists()
