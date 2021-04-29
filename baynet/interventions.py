from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Tuple, List, TYPE_CHECKING, Optional
from typing_extensions import Literal

import numpy as np
import yaml

if TYPE_CHECKING:
    from .structure import DAG
    import pandas as pd

__all__ = ["odds_ratio_config", "odds_ratio_all"]


def propagate_marginal(bn: DAG, target: str) -> np.ndarray:
    marginals: Dict[str, np.ndarray] = dict()
    for node_idx in bn.topological_sorting():
        node = bn.vs[node_idx]["name"]
        node_cpd = bn.get_node(node)["CPD"].array.copy()
        parents = bn.get_node(node)["CPD"].parents
        if parents:
            for parent in parents[::-1]:
                node_cpd = marginals[parent].dot(node_cpd)
        marginals[node] = node_cpd
        if node == target:
            break
    return marginals[target]


def marginal_ratio(
    bn: DAG, target: str, target_reference: Union[str, int], target_subject: Union[str, int]
) -> float:
    reference_idx = bn.get_node(target)["CPD"].levels.index(str(target_reference))
    subject_idx = bn.get_node(target)["CPD"].levels.index(str(target_subject))
    target_marginal = propagate_marginal(bn=bn, target=target)
    return target_marginal[subject_idx] / target_marginal[reference_idx]


def odds_ratio(
    bn: DAG,
    target: str,
    target_reference: Union[str, int],
    target_subject: Union[str, int],
    intervention: str,
    intervention_reference: Union[str, int],
    intervention_subject: Union[str, int],
) -> float:
    reference_bn = bn.mutilate(node=intervention, evidence_level=str(intervention_reference))
    intervention_bn = bn.mutilate(node=intervention, evidence_level=str(intervention_subject))
    reference_ratio = marginal_ratio(
        bn=reference_bn,
        target=target,
        target_reference=target_reference,
        target_subject=target_subject,
    )
    subject_ratio = marginal_ratio(
        bn=intervention_bn,
        target=target,
        target_reference=target_reference,
        target_subject=target_subject,
    )
    return subject_ratio / reference_ratio


def odds_ratio_config(bn: DAG, config) -> Dict[Tuple[str, int], float]:
    results: Dict[Tuple[str, int], float] = {}
    if not isinstance(config["target_subjects"], list):
        config["target_subjects"] = [config["target_subjects"]]
    for target_subject in config["target_subjects"]:
        for intervention in config["interventions"]:
            if not isinstance(intervention["intervention_subjects"], list):
                intervention["intervention_subjects"] = [intervention["intervention_subjects"]]
            for intervention_subject in intervention["intervention_subjects"]:
                key = (
                    config["target_node"],
                    config["target_reference"],
                    target_subject,
                    intervention["intervention_node"],
                    intervention["intervention_reference"],
                    intervention_subject,
                )
                results[key] = odds_ratio(bn, *key)
    return results


def odds_ratio_all(
    bn: DAG, target: str, target_reference: Optional[str]
) -> Dict[Tuple[str, int], float]:
    def _levels(node: str):
        return sorted(bn.get_node(node)["CPD"].levels)

    def _intervention(node: str):
        levels = _levels(node)
        return {
            "intervention_node": node,
            "intervention_reference": levels[0],
            "intervention_subjects": levels[1:],
        }

    target_levels = _levels(target)
    if not target_reference:
        target_reference = target_levels[0]

    config = {
        "target_node": target,
        "target_reference": target_reference,
        "target_subjects": list(set(target_levels) - set(target_reference)),
        "interventions": [_intervention(n) for n in list(bn.nodes - set(target))],
    }
    return odds_ratio_config(bn=bn, config=config)


def value_aggregator(
    values: List[float],
    aggregation: Literal['mean', 'median'],
    bounds: Optional[Literal['minmax', 'quartiles']],
):
    agg_dict = {
        "mean": lambda x: {"mean": np.mean(x)},
        "median": lambda x: {"median": np.median(x)},
    }
    bound_dict = {
        "minmax": lambda x: {"min": np.min(x), "max": np.max(x)},
        "quartiles": lambda x: {"25%": np.quantile(x, 0.25), "75%": np.quantile(x, 0.75)},
    }
    return {
        **agg_dict.get(aggregation, lambda x: {})(values),
        **bound_dict.get(bounds, lambda x: {})(values),
    }


def odds_ratio_aggregator(
    bn: DAG,
    *,
    config: Optional[Union[dict, Path]] = None,
    target: Optional[str] = None,
    target_reference: Optional[Union[str, int]] = None,
    cpdag: bool = False,
    data: pd.DataFrame = None,
    aggregation: Optional[Literal['mean', 'median']] = "median",
    bounds: Optional[Literal['minmax', 'quartiles']] = "minmax",
):
    if cpdag and data is None:
        raise ValueError(
            "Data must be provided to populate the parameters of the markov equivalence set."
        )
    results = None
    if config and not target:
        if isinstance(config, Path):
            try:
                with open(config, "r") as f:
                    config = yaml.load(f, yaml.FullLoader)
            except FileNotFoundError:
                raise FileNotFoundError(f"Config file not found at: {config}")
        if cpdag:
            ld = [odds_ratio_config(bni, config) for bni in bn.get_equivalence_class(data=data)]
            results = {k: [dic[k] for dic in ld] for k in ld[0]}
        else:
            results = odds_ratio_config(bn, config)
    if target and not config:
        if cpdag:
            ld = [
                odds_ratio_all(bni, target, target_reference)
                for bni in bn.get_equivalence_class(data=data)
            ]
            results = {k: [dic[k] for dic in ld] for k in ld[0]}
        else:
            results = odds_ratio_all(bn, target, target_reference)
    if not results:
        raise ValueError("Either target or config must be set. Both cannot be set.")
    if not cpdag:
        return results
    else:
        return {k: value_aggregator(v, aggregation, bounds) for k, v in results.items()}