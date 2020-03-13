import numpy as np
from baynet.structure import DAG

TEST_MODELSTRING = "[A][B|C:D][C|D][D]"
REVERSED_MODELSTRING = "[A][B][C|B][D|B:C]"


def test_dag(reversed: bool = False) -> DAG:
    if not reversed:
        return DAG.from_modelstring(TEST_MODELSTRING)
    else:
        return DAG.from_modelstring(REVERSED_MODELSTRING)


def partial_dag() -> DAG:
    return DAG.from_modelstring("[A][B|C:D][C][D]")


def empty_dag() -> DAG:
    return DAG.from_amat(np.zeros((4, 4)), list("ABCD"))
