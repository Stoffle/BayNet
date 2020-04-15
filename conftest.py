import pytest, tempfile
from pathlib import Path
import numpy as np
from baynet.structure import DAG

TEST_MODELSTRING = "[A][B|C:D][C|D][D]"
REVERSED_MODELSTRING = "[A][B][C|B][D|B:C]"


@pytest.fixture(scope="function")
def test_dag() -> DAG:
    return DAG.from_modelstring(TEST_MODELSTRING)


@pytest.fixture(scope="function")
def reversed_dag() -> DAG:
    return DAG.from_modelstring(REVERSED_MODELSTRING)


@pytest.fixture(scope="function")
def partial_dag() -> DAG:
    return DAG.from_modelstring("[A][B|C:D][C][D]")


@pytest.fixture(scope="function")
def empty_dag() -> DAG:
    return DAG.from_amat(np.zeros((4, 4)), list("ABCD"))


@pytest.fixture(scope="session")
def test_modelstring() -> str:
    return TEST_MODELSTRING


@pytest.fixture(scope="session")
def reversed_modelstring() -> str:
    return REVERSED_MODELSTRING


@pytest.fixture(scope="function")
def temp_out() -> Path:
    """
    Create temporary directory for storing test outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()
