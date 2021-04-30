"""
Functions which perform interventions on a given Bayesian network.

Only odds ratios currently supported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Tuple, List, TYPE_CHECKING, Optional, cast
from typing_extensions import Literal

import numpy as np
import yaml

if TYPE_CHECKING:
    from .structure import DAG
    import pandas as pd

__all__ = ["odds_ratio_aggregator"]


def collapse_posterior(bayesian_network: DAG, target: str) -> np.ndarray:
    """Collapse the posterior distributions for a particular target."""
    marginals: Dict[str, np.ndarray] = dict()
    for node_idx in bayesian_network.topological_sorting():
        node = bayesian_network.vs[node_idx]["name"]
        node_cpd = bayesian_network.get_node(node)["CPD"].array.copy()
        parents = bayesian_network.get_node(node)["CPD"].parents
        if parents:
            for parent in parents[::-1]:
                node_cpd = marginals[parent].dot(node_cpd)
        marginals[node] = node_cpd
        if node == target:
            break
    return marginals[target]


def posterior_ratio(
    bayesian_network: DAG,
    target: str,
    target_reference: Union[str, int],
    target_subject: Union[str, int],
) -> float:
    """Calculate the ratio of collapsed posterior of target level / reference level."""
    reference_idx = bayesian_network.get_node(target)["CPD"].levels.index(str(target_reference))
    subject_idx = bayesian_network.get_node(target)["CPD"].levels.index(str(target_subject))
    target_marginal = collapse_posterior(bayesian_network=bayesian_network, target=target)
    return target_marginal[subject_idx] / target_marginal[reference_idx]


def odds_ratio(
    bayesian_network: DAG,
    target: str,
    target_reference: Union[str, int],
    target_subject: Union[str, int],
    intervention: str,
    intervention_reference: Union[str, int],
    intervention_subject: Union[str, int],
) -> float:
    """Calculate the adjusted odds ratio given specified input target, intervention and levels."""
    reference_bn = bayesian_network.mutilate(
        node=intervention, evidence_level=str(intervention_reference)
    )
    intervention_bn = bayesian_network.mutilate(
        node=intervention, evidence_level=str(intervention_subject)
    )
    reference_ratio = posterior_ratio(
        bayesian_network=reference_bn,
        target=target,
        target_reference=target_reference,
        target_subject=target_subject,
    )
    subject_ratio = posterior_ratio(
        bayesian_network=intervention_bn,
        target=target,
        target_reference=target_reference,
        target_subject=target_subject,
    )
    return subject_ratio / reference_ratio


def odds_ratio_config(bayesian_network: DAG, config: dict) -> Dict[tuple, float]:
    """
    Calculate the odds ratio given a configuration.

    Configuration specifies target / interventions and their levels.
    """
    results: Dict[tuple, float] = {}
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
                results[key] = odds_ratio(bayesian_network, *key)
    return results


def odds_ratio_all(
    bayesian_network: DAG, target: str, target_reference: Optional[Union[str, int]]
) -> Dict[tuple, float]:
    """Calculate ALL odds ratios given a target, and optionally a target reference."""

    def _levels(node: str) -> List[Union[str, int]]:
        return sorted(bayesian_network.get_node(node)["CPD"].levels)

    def _intervention(node: str) -> Dict[str, Union[str, int, list]]:
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
        "target_subjects": list(set(target_levels) - {target_reference}),
        "interventions": [_intervention(n) for n in list(bayesian_network.nodes - {target})],
    }
    return odds_ratio_config(bayesian_network=bayesian_network, config=config)


def value_aggregator(
    values: List[float],
    aggregation: Literal['mean', 'median'],
    bounds: Optional[Literal['minmax', 'quartiles']],
) -> Dict[str, float]:
    """Aggregate set of odds ratios given aggregation type and bound type."""
    agg_dict = {
        "mean": lambda x: {"mean": np.mean(x)},
        "median": lambda x: {"median": np.median(x)},
    }
    bound_dict = {
        "minmax": lambda x: {"min": np.min(x), "max": np.max(x)},
        "quartiles": lambda x: {"25%": np.quantile(x, 0.25), "75%": np.quantile(x, 0.75)},
    }
    return {
        **agg_dict.get(cast(str, aggregation), lambda x: {})(values),
        **bound_dict.get(cast(str, bounds), lambda x: {})(values),
    }


def odds_ratio_aggregator(
    bayesian_network: DAG,
    *,
    config: Optional[Union[dict, Path]] = None,
    target: Optional[str] = None,
    target_reference: Optional[Union[str, int]] = None,
    cpdag: bool = False,
    data: pd.DataFrame = None,
    aggregation: Literal['mean', 'median'] = "median",
    bounds: Optional[Literal['minmax', 'quartiles']] = "minmax",
) -> Union[Dict[tuple, Dict[str, float]], Dict[tuple, float]]:
    """Calculate odds ratio given config or target input."""
    if cpdag and data is None:
        raise ValueError(
            "Data must be provided to populate the parameters of the markov equivalence set."
        )
    results = None
    cpdag_results = None
    if config and not target:
        if isinstance(config, Path):
            try:
                with open(config, "r") as file:
                    config = yaml.load(file, yaml.FullLoader)
            except FileNotFoundError:
                raise FileNotFoundError(f"Config file not found at: {config}")
        if cpdag:
            temp = [
                odds_ratio_config(bni, cast(dict, config))
                for bni in bayesian_network.get_equivalence_class(data=data)
            ]
            cpdag_results = {k: [dic[k] for dic in temp] for k in temp[0]}
        else:
            results = odds_ratio_config(bayesian_network, cast(dict, config))
    if target and not config:
        if cpdag:
            temp = [
                odds_ratio_all(bni, target, target_reference)
                for bni in bayesian_network.get_equivalence_class(data=data)
            ]
            cpdag_results = {k: [dic[k] for dic in temp] for k in temp[0]}
        else:
            results = odds_ratio_all(bayesian_network, target, target_reference)
    if not results and not cpdag_results:
        raise ValueError("Either target or config must be set. Both cannot be set.")
    if not cpdag:
        return results
    return {k: value_aggregator(v, aggregation, bounds) for k, v in cpdag_results.items()}  # type: ignore
