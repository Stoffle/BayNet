from __future__ import annotations

from typing import Dict, Union, Tuple, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .structure import DAG

__all__ = [
    "odds_ratio_config",
    "odds_ratio_all"
]


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


def marginal_ratio(bn: DAG, target: str, target_reference: Union[str, int], target_subject: Union[str, int]) -> float:
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
        intervention_subject: Union[str, int]
) -> float:
    reference_bn = bn.mutilate(node=intervention, evidence_level=str(intervention_reference))
    intervention_bn = bn.mutilate(node=intervention, evidence_level=str(intervention_subject))
    reference_ratio = marginal_ratio(bn=reference_bn,
                                     target=target,
                                     target_reference=target_reference,
                                     target_subject=target_subject)
    subject_ratio = marginal_ratio(bn=intervention_bn,
                                   target=target,
                                   target_reference=target_reference,
                                   target_subject=target_subject)
    return subject_ratio / reference_ratio


def odds_ratio_config(bn: DAG, config) -> Dict[Tuple[str, int]]:
    results: Dict[Tuple[str, int]] = {}
    if not isinstance(config["target_subjects"], list):
        config["target_subjects"] = [config["target_subjects"]]
    for target_subject in config["target_subjects"]:
        for intervention in config["interventions"]:
            if not isinstance(intervention["intervention_subjects"], list):
                intervention["intervention_subjects"] = [intervention["intervention_subjects"]]
            for intervention_subject in intervention["intervention_subjects"]:
                key = (config["target_node"],
                       config["target_reference"],
                       target_subject,
                       intervention["intervention_node"],
                       intervention["intervention_reference"],
                       intervention_subject)
                results[key] = odds_ratio(bn, *key)
    return results


def odds_ratio_all(self, target, target_reference, target_subjects):
    return None
