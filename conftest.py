import pytest, tempfile
from pathlib import Path


@pytest.fixture(scope="function")
def temp_out():
    """
    Create temporary directory for storing test outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()
