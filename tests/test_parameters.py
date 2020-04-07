from time import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from baynet.structure import DAG
from baynet.parameters import (
    ConditionalProbabilityTable,
    _sample_cpt,
    ConditionalProbabilityDistribution,
)


def test_CPT_init(test_dag):
    dag = test_dag
    dag.vs['levels'] = [["0", "1"] for v in dag.vs]
    cpt = ConditionalProbabilityTable(dag.vs[1])
    assert cpt.array.shape == (2, 2, 2)
    assert np.allclose(cpt.array, 0)

    dag.add_vertex("E")
    with pytest.raises(ValueError):
        ConditionalProbabilityTable(dag.vs[dag.get_node_index("E")])

    dag.add_edge("E", "A")
    with pytest.raises(ValueError):
        ConditionalProbabilityTable(dag.vs[dag.get_node_index("A")])


def test_CPT_rescale(test_dag):
    dag = test_dag
    for n_levels in [1, 2, 3, 4]:
        dag.vs['levels'] = [list(map(str, range(n_levels))) for v in dag.vs]
        cpt = ConditionalProbabilityTable(dag.vs[1])
        cpt.rescale_probabilities()
        # Check cumsum is working properly
        for i in range(cpt.n_levels):
            assert np.allclose(cpt.cumsum_array[:, :, i], (i + 1) / cpt.n_levels)
    cpt.array = np.random.uniform(size=(3, 3, 3))
    cpt.rescale_probabilities()
    for i in range(3):
        for j in range(3):
            # Check last value in each CPT 'row' is 1 (double checking cumsum with random init)
            assert np.isclose(np.sum(cpt.cumsum_array[i, j, -1]), 1)
            # and each value is larger than the previous
            assert (
                cpt.cumsum_array[i, j, 0] <= cpt.cumsum_array[i, j, 1] <= cpt.cumsum_array[i, j, 2]
            )


def test_CPT_sample_exceptions(test_dag):
    dag = test_dag
    dag.vs['levels'] = [["0", "1"] for v in dag.vs]
    cpt = ConditionalProbabilityTable(dag.vs[1])
    with pytest.raises(ValueError):
        cpt.sample(None)


def test_CPT_sample_parameters(test_dag):
    dag = test_dag
    dag.vs['levels'] = [["0", "1"] for v in dag.vs]
    cpt = ConditionalProbabilityTable(dag.vs[1])
    cpt_shape = cpt.array.shape
    cpt.sample_parameters(seed=0)
    assert cpt.array.shape == cpt_shape


def test_CPT_marginalise(test_dag):
    dag = test_dag
    dag.generate_discrete_parameters(min_levels=2, max_levels=2, seed=1)
    cpt = dag.vs[1]['CPD']
    cpt.marginalise('C')
    assert cpt.parents == ['D']
    assert cpt.array.shape == (2, 2)
    assert cpt.parent_levels == [['0', '1']]
    assert cpt.n_parent_levels == [2]


def test_sample_cpt(test_dag):
    dag = test_dag
    dag.vs['levels'] = [["0", "1"] for v in dag.vs]
    cpt = ConditionalProbabilityTable(dag.vs[1])
    cpt.array[0, 0, :] = [0.5, 0.5]
    cpt.array[0, 1, :] = [1.0, 0.0]
    cpt.array[1, 0, :] = [0.0, 1.0]
    cpt.array[1, 1, :] = [0.5, 0.5]
    cpt.rescale_probabilities()
    parent_values = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]])
    parent_values_tuples = list(map(tuple, parent_values))

    random_vector = np.repeat([[0.1, 0.9]], 4, axis=0).flatten()

    expected_output = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    assert np.all(
        _sample_cpt(cpt.cumsum_array, parent_values_tuples, random_vector) == expected_output
    )
    np.random.seed(0)  # TODO: replace with mocking np.random.normal
    data = pd.DataFrame(np.zeros((8, 4)), columns=list("ABCD"), dtype=int)
    data.iloc[:, [2, 3]] = parent_values
    assert np.all(cpt.sample(data).astype(int) == [1, 1, 0, 0, 1, 1, 0, 1])


def test_cpd_init(test_dag):
    dag = test_dag
    cpd = ConditionalProbabilityDistribution(dag.vs[1])
    assert cpd.array.shape == (2,)
    assert np.allclose(cpd.array, 0)


def test_cpd_sample_params(test_dag):
    dag = test_dag
    cpd = ConditionalProbabilityDistribution(dag.vs[1])
    cpd.sample_parameters(weights=[1], seed=0)
    assert np.allclose(cpd.array, 1)


def test_cpd_sample(test_dag):
    dag = test_dag
    cpd = ConditionalProbabilityDistribution(dag.vs[1], std=0)
    cpd.sample_parameters(weights=[1])
    assert np.allclose(cpd.sample(pd.DataFrame(np.ones((10, 4)), columns=list("ABCD"))), 2)
    with pytest.raises(TypeError):
        cpd.sample()

    cpd_no_parents = ConditionalProbabilityDistribution(dag.vs[0], std=0)
    cpd_no_parents.sample_parameters(weights=[1])
    assert np.allclose(
        cpd_no_parents.sample(pd.DataFrame(np.ones((10, 4)), columns=list("ABCD"))), 0
    )
