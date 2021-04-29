from typing import List, Union

import pandas as pd

import pytest
from baynet import DAG


def _test_bn(level_1: Union[str, int], level_2: Union[str, int]) -> DAG:
    dag = DAG.from_modelstring("[A|B:C:D][B|C][C][D]")
    data = pd.DataFrame(
        {
            "A": [level_1] * 7 + [level_2] * 3,
            "B": [level_1] * 2 + [level_2] * 8,
            "C": [level_1] * 1 + [level_2] * 9,
            "D": [level_1] * 4 + [level_2] * 6,
        }
    )
    dag.estimate_parameters(data, infer_levels=True)
    return dag


@pytest.mark.parametrize("level_1,level_2", [(0, 1), ("level_1", "level_2")])
def test_odds_ratio_config(level_1: Union[str, int], level_2: Union[str, int]):
    """
    Test both string and int input data.
    """
    bn = _test_bn(level_1=level_1, level_2=level_2)
    config = {
        "target_node": "A",
        "target_reference": level_1,
        "target_subjects": level_2,
        "interventions":
            [
                {
                    "intervention_node": "B",
                    "intervention_reference": level_1,
                    "intervention_subjects": [level_2, level_2]
                }
            ]
    }
    ors = bn.adjusted_odds_ratio(config)
    assert all([isinstance(odds_ratio, float) for key, odds_ratio in ors.items()])


@pytest.mark.parametrize("level_1,level_2", [(0, 1), ("level_1", "level_2")])
def test_odds_ratio_all(level_1: Union[str, int], level_2: Union[str, int]):
    """
    Test both string and int input data.
    """
    bn = _test_bn(level_1=level_1, level_2=level_2)
    ors = bn.adjusted_odds_ratio_all("A")
    assert all([isinstance(odds_ratio, float) for key, odds_ratio in ors.items()])

