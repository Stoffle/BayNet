from baynet.structure import Graph

TEST_MODELSTRING = "[A][B|C:D][C|D][D]"
REVERSED_MODELSTRING = "[A][C|B][D|B:C][B]"

def test_dag(reversed: bool = False) -> Graph:
    if not reversed:
        return Graph.from_modelstring(TEST_MODELSTRING)
    else:
        return Graph.from_modelstring(REVERSED_MODELSTRING)

def partial_dag() -> Graph:
    return Graph.from_modelstring("[A][B|C:D][C][D]")

