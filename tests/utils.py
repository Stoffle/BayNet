import numpy as np
from baynet.structure import DAG

TEST_MODELSTRING = "[A][B|C:D][C|D][D]"
REVERSED_MODELSTRING = "[A][B][C|B][D|B:C]"


def test_dag(reverse: bool = False) -> DAG:
    if not reverse:
        return DAG.from_modelstring(TEST_MODELSTRING, name='test_dag')
    else:
        return DAG.from_modelstring(REVERSED_MODELSTRING, name='test_dag')


def partial_dag() -> DAG:
    return DAG.from_modelstring("[A][B|C:D][C][D]", name='partial_dag')


def empty_dag() -> DAG:
    return DAG.from_amat(np.zeros((4, 4)), list("ABCD"), name='empty_dag')

