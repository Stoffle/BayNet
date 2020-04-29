"""Functions for loading/saving DAGs."""
from typing import no_type_check, Optional
from pathlib import Path
import numpy as np
import pyparsing as pp
import baynet
from baynet.utils import DAG_pb2
from baynet.parameters import (
    ConditionalProbabilityDistribution,
    ConditionalProbabilityTable,
)


@no_type_check
def dag_to_buf(dag: "baynet.DAG") -> bytes:
    """Dump DAG object to bytes using protobuf."""
    # pylint: disable=no-member
    dag_buf = DAG_pb2.DAG()
    for vertex in dag.vs:
        node = DAG_pb2.Node()
        node.name = vertex["name"]
        node.parents.extend([str(v["name"]) for v in vertex.neighbors(mode="in")])
        if vertex["CPD"] is not None:
            if isinstance(vertex["CPD"], ConditionalProbabilityTable):
                node.variable_type = DAG_pb2.NodeType.DISCRETE
                node.levels.extend(vertex["levels"])
            elif isinstance(vertex["CPD"], ConditionalProbabilityDistribution):
                node.variable_type = DAG_pb2.NodeType.CONTINUOUS
            node.cpd_array.shape.extend(vertex["CPD"].array.shape)
            node.cpd_array.flat_array = vertex["CPD"].array.tobytes()
        dag_buf.nodes.append(node)
    return dag_buf.SerializeToString()


@no_type_check
def buf_to_dag(dag_buf: bytes, dag: Optional["baynet.DAG"] = None) -> "baynet.DAG":
    """Convert protobuf generated bytes into DAG object."""
    # pylint: disable=no-member
    dag_from_buf = DAG_pb2.DAG.FromString(dag_buf)
    if dag is None:
        dag = baynet.DAG()
    dag.add_vertices([node.name for node in dag_from_buf.nodes])
    for buf_node in dag_from_buf.nodes:
        edges = [(source, buf_node.name) for source in buf_node.parents]
        dag.add_edges(edges)
        node = dag.get_node(buf_node.name)
        if buf_node.variable_type == DAG_pb2.NodeType.DISCRETE:
            node["levels"] = list(buf_node.levels)
            cpd = ConditionalProbabilityTable()
            cpd.name = buf_node.name
            cpd.levels = list(buf_node.levels)
            cpd.array = buf_to_array(buf_node.cpd_array)
            cpd.rescale_probabilities()
        elif buf_node.variable_type == DAG_pb2.NodeType.CONTINUOUS:
            cpd = ConditionalProbabilityDistribution()
            cpd.name = buf_node.name
            cpd.array = buf_to_array(buf_node.cpd_array)
        cpd.parents = list(buf_node.parents)
        node["CPD"] = cpd
    return dag


@no_type_check
def buf_to_array(array_buf: DAG_pb2.Array) -> np.ndarray:
    """Convert protobuf array object into numpy ndarray."""
    # pylint: disable=no-member
    arr = np.frombuffer(array_buf.flat_array).copy()
    if arr.size > 0:
        arr = arr.reshape(list(array_buf.shape))
    return arr


def dag_from_bif(bif_path: Path) -> 'baynet.DAG':
    """Create a DAG object from a .bif file."""
    lcurly, rcurly, lsquare, rsquare, lbracket, rbracket, vbar, semicolon = map(
        pp.Suppress, "{}[]()|;"
    )
    var_literal = pp.Suppress(pp.CaselessLiteral("variable"))
    probability_literal = pp.Suppress(pp.CaselessLiteral("probability"))
    type_discrete_literal = pp.Suppress(pp.CaselessLiteral("type discrete"))
    float_ = pp.Word(pp.nums + ".").setParseAction(lambda s, l, t: [float(t[0])])
    var_name = pp.Word(pp.alphanums + "_")
    int_ = pp.Word(pp.nums).setParseAction(lambda s, l, t: [int(t[0])])

    source_cpt_row = pp.Suppress(pp.CaselessLiteral("table")) + pp.delimitedList(float_) + semicolon
    source_cpt = (
        probability_literal
        + lbracket
        + var_name("variable")
        + rbracket
        + lcurly
        + source_cpt_row("probabilities")
        + rcurly
    )

    child_cpt_row = pp.Group(
        lbracket
        + pp.delimitedList(var_name)("parent_values")
        + rbracket
        + pp.delimitedList(float_)("probabilities")
        + semicolon
    )
    child_cpt = (
        probability_literal
        + lbracket
        + var_name("variable")
        + vbar
        + pp.delimitedList(var_name)("parents")
        + rbracket
        + lcurly
        + pp.OneOrMore(child_cpt_row)("cpt")
        + rcurly
    )

    network = pp.Literal("network unknown") + lcurly + pp.Optional(var_name) + rcurly
    variable = pp.Group(
        var_literal
        + var_name("name")
        + lcurly
        + type_discrete_literal
        + lsquare
        + int_
        + rsquare
        + lcurly
        + pp.delimitedList(var_name)("levels")
        + rcurly
        + semicolon
        + rcurly
    )
    cpt = pp.Group(source_cpt ^ child_cpt)

    parser = (
        pp.Suppress(pp.Optional(network))
        + pp.OneOrMore(variable)("variables")
        + pp.OneOrMore(cpt)("cpts")
    )

    parsed = parser.parseFile(bif_path, parseAll=True).asDict()

    dag = baynet.DAG()
    for vertex in parsed["variables"]:
        dag.add_vertex(name=vertex["name"], levels=vertex["levels"])
    for cpt in parsed["cpts"]:
        for parent in cpt.get("parents", []):
            dag.add_edge(parent, cpt["variable"])
    for cpt in parsed["cpts"]:
        node = dag.get_node(cpt["variable"])
        node["CPD"] = ConditionalProbabilityTable(node)
        if "parents" not in cpt:
            node["CPD"].array = np.array(cpt["probabilities"])
            node["CPD"].rescale_probabilities()
            continue
        for row in cpt["cpt"]:
            indexer = [0] * len(cpt["parents"])
            for parent, level in zip(cpt["parents"], row["parent_values"]):
                idx = node["CPD"].parents.index(parent)
                indexer[idx] = dag.get_node(parent)["levels"].index(level)
            node["CPD"].array[tuple(indexer)] = row["probabilities"]
        node["CPD"].rescale_probabilities()
    return dag
