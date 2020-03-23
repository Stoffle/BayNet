from string import Template
from typing import List

from igraph import Vertex

from .structure import DAG
from .parameters import ConditionalProbabilityDistribution, ConditionalProbabilityTable


class BIFWriter():
    """Class for writing .bif files."""

    # pylint: disable=unsubscriptable-object, not-an-iterable, arguments-differ
    _network_template = Template("network $name {\n}\n")
    _continuous_variable_template = Template(
        """variable $name {\n  type continuous;\n$properties}\n"""
    )
    _continuous_probability_template = Template(
        """probability ( $name$parents ) {$weights\n}\n"""
    )
    _continuous_weights_template = Template(
        """\n  table $weights;"""
    )
    _discrete_variable_template = Template(
        """variable $name {\n  type discrete [ $n_levels] { $levels };\n}\n"""
    )
    _discrete_probability_template = Template(
        """probability ( $name$parents ) {\n  $cpt ;\n}\n"""
    )
    _discrete_source_template = Template(
        """table $values"""
    )
    _discrete_cpt_row_template = Template(
        """($parent_configuration) $probabilities;"""
    )

    def __init__(self) -> None:
        return

    def network_to_bif_string(self, network: DAG) -> str:
        bif_string = self._network_template.safe_substitute(name=DAG.name)

        for vertex in network.vs:
            bif_string += self._continuous_variable_template.safe_substitute(
                name=vertex['name'], properties=""
            )

        for vertex in network.vs:
            if vertex['CPD'] is not None and vertex['CPD'].array.size > 0:
                bif_string += self._continuous_probability_template.safe_substitute(
                    node=vertex['name'],
                    parents=self._parents_to_string(vertex['CPD'].parent_names),
                    values=', '.join(list(vertex['CPD'].array.astype(str))),
                )

    def _node_to_bif_string(self, node: Vertex):
        cpd = node['CPD']
        if isinstance(cpd, ConditionalProbabilityTable):
            return self._discrete_variable_template.safe_substitute(
                name = node['name'],
                n_levels = cpd.n_levels,
                levels = cpd.levels,
            )
        elif isinstance(cpd, ConditionalProbabilityDistribution):
            return self._continuous_variable_template.safe_substitute(
                name = node['name'],
                properties = f"  property mean {cpd.mean}\n  property std {cpd.std}\n"
            )
        else:
            raise NotImplementedError()

    def _parameters_to_bif_string(self, node: Vertex):
        cpd = node['CPD']
        if isinstance(cpd, ConditionalProbabilityTable):
            return self._discrete_probability_template.safe_substitute(
                name = node['name'],
                parents = self._parents_to_string(cpd.parent_names),
                cpt = self._cpt_to_string(cpd)
            )
        elif isinstance(cpd, ConditionalProbabilityDistribution):
            return self._continuous_probability_template.safe_substitute(
                name = node['name'],
                parents = self._parents_to_string(cpd.parent_names),
                weights = self._cpd_weights_to_string(cpd)
            )
        else:
            raise NotImplementedError()

    def _cpt_to_string(self, cpt: ConditionalProbabilityTable) -> str:
        if len(cpt.array.shape) == 1:
            return self._discrete_source_template.safe_substitute(
                values = ', '.join(list(cpt.array.astype(str)))
            )
        cpt_str = ""
        for config, config_indices in zip(cpt.parent_configurations, cpt.parent_configurations_idx):
            cpt_str += self._discrete_cpt_row_template.safe_substitute(
                parent_configuration = config,
                probabilities = ', '.join(cpt.array[config_indices].astype(str))
            )
        return cpt_str

    def _cpd_weights_to_string(self, cpd: ConditionalProbabilityDistribution) -> str:
        if cpd.array.size == 0:
            return ""
        else:
            return self._continuous_weights_template.safe_substitute(
                weights = ', '.join(list(cpd.array.astype(str)))
            )
    
    def _parents_to_string(self, parent_names: List[str]) -> str:
        if not parent_names:
            return ""
        return f" | {', '.join(parent_name for parent_name in parent_names)}"

