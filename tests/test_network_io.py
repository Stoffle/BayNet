import pytest

from baynet.network_io import BIFWriter
from baynet.parameters import ConditionalProbabilityDistribution, ConditionalProbabilityTable
from .utils import test_dag

def test_BIFWriter():
    return


def test_BIFWriter_node_to_str():
    writer = BIFWriter()

    dag = test_dag()
    dag.generate_discrete_parameters(
        cardinality_min=2,
        cardinality_max=2,
        seed=1
    )
    assert writer._node_to_bif_string(dag.vs[0]) == "variable A {\n  type discrete [ 2] { ['0', '1'] };\n}\n" 

    dag.generate_continuous_parameters()
    assert writer._node_to_bif_string(dag.vs[0]) == "variable A {\n  type continuous;\n  property mean 0.0\n  property std 1.0\n}\n" 

    with pytest.raises(NotImplementedError):
        dag.vs['CPD'] = None
        writer._node_to_bif_string(dag.vs[0])

def test_BIFWriter_params_to_str():
    writer = BIFWriter()

    dag = test_dag()
    dag.generate_discrete_parameters(
        cardinality_min=2,
        cardinality_max=2,
        seed=1
    )
    assert writer._parameters_to_bif_string(dag.vs[0]) == "probability ( A ) {\n  table 0.6648847938292544, 0.3351152061707456 ;\n}\n"

    dag.generate_continuous_parameters(possible_weights=[2])
    assert writer._parameters_to_bif_string(dag.vs[0]) == "probability ( A ) {\n}\n"
    assert writer._parameters_to_bif_string(dag.vs[1]) == """probability ( B | C, D ) {
  table 2, 2;
}
"""

    with pytest.raises(NotImplementedError):
        dag.vs['CPD'] = None
        writer._node_to_bif_string(dag.vs[0])


