import tempfile
from pathlib import Path
import numpy as np
import pytest
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


@pytest.fixture(scope="function")
def temp_out():
    """
    Create temporary directory for storing test outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()
