"""Metrics for comparing Graph objects."""
from typing import Set, Tuple
from .structure import DAG


def _check_args(true_graph: DAG, learnt_graph: DAG, skeleton: bool = False) -> bool:
    if not isinstance(true_graph, DAG):
        raise ValueError(f"Expected `true_graph` to be a Graph object, got {type(true_graph)}")
    if not true_graph.is_directed() and not skeleton:
        raise ValueError("`true_graph` is undirected")
    if not isinstance(learnt_graph, DAG):
        raise ValueError(f"Expected `learnt_graph` to be a Graph object, got {type(learnt_graph)}")
    if not learnt_graph.is_directed() and not skeleton:
        raise ValueError("`learnt_graph` is undirected")
    return True


def false_positive_edges(
    true_graph: DAG, learnt_graph: DAG, skeleton: bool
) -> Set[Tuple[str, str]]:
    """Get the set of edges in learnt_graph but not true_graph."""
    _check_args(true_graph, learnt_graph, skeleton)
    if not skeleton:
        return learnt_graph.edges - true_graph.edges
    return learnt_graph.skeleton_edges - true_graph.skeleton_edges


def true_positive_edges(true_graph: DAG, learnt_graph: DAG, skeleton: bool) -> Set[Tuple[str, str]]:
    """Get the set of edges in both learnt_graph and true_graph."""
    _check_args(true_graph, learnt_graph, skeleton)
    if not skeleton:
        return true_graph.edges & learnt_graph.edges
    return true_graph.skeleton_edges & learnt_graph.skeleton_edges


def false_negative_edges(
    true_graph: DAG, learnt_graph: DAG, skeleton: bool
) -> Set[Tuple[str, str]]:
    """Get the set of edges in true_graph but not learnt_graph."""
    _check_args(true_graph, learnt_graph, skeleton)
    if not skeleton:
        return true_graph.edges - learnt_graph.edges
    return true_graph.skeleton_edges - learnt_graph.skeleton_edges


def precision(true_graph: DAG, learnt_graph: DAG, skeleton: bool) -> float:
    """Compute (DAG or skeleton) precision of a learnt graph."""
    _check_args(true_graph, learnt_graph, skeleton)
    true_pos = len(true_positive_edges(true_graph, learnt_graph, skeleton))
    all_learnt = len(learnt_graph.edges) if not skeleton else len(learnt_graph.skeleton_edges)
    return 0.0 if (all_learnt == 0.0) else true_pos / all_learnt


def recall(true_graph: DAG, learnt_graph: DAG, skeleton: bool) -> float:
    """Compute (DAG or skeleton) recall of a learnt graph."""
    args = (true_graph, learnt_graph, skeleton)
    _check_args(*args)
    true_pos = len(true_positive_edges(*args))
    false_neg = len(false_negative_edges(*args))
    return 0.0 if (true_pos + false_neg == 0.0) else true_pos / (true_pos + false_neg)


def f1_score(true_graph: DAG, learnt_graph: DAG, skeleton: bool) -> float:
    """Compute F1 score (DAG or skeleton) of a learnt graph."""
    args = (true_graph, learnt_graph, skeleton)
    _check_args(*args)
    prec = precision(*args)
    rec = recall(*args)
    return 0.0 if (prec + rec == 0.0) else (2 * prec * rec) / (prec + rec)


def shd(true_graph: DAG, learnt_graph: DAG, skeleton: bool) -> int:
    """Compute structural hamming distance (DAG or skeleton) of a learnt graph."""
    _check_args(true_graph, learnt_graph)
    missing = len(false_negative_edges(true_graph, learnt_graph, True)) / 2
    added = len(false_positive_edges(true_graph, learnt_graph, True)) / 2
    backwards = 0 if skeleton else len(true_graph.reversed_edges & learnt_graph.edges)
    return int(missing + added + backwards)


def false_positive_v_structures(true_graph: DAG, learnt_graph: DAG) -> Set[Tuple[str, str, str]]:
    """Get the set of v-structures in learnt_graph but not true_graph."""
    _check_args(true_graph, learnt_graph)
    return learnt_graph.get_v_structures() - true_graph.get_v_structures()


def true_positive_v_structures(true_graph: DAG, learnt_graph: DAG) -> Set[Tuple[str, str, str]]:
    """Get the set of v-structures in both learnt_graph and true_graph."""
    _check_args(true_graph, learnt_graph)
    return learnt_graph.get_v_structures() & true_graph.get_v_structures()


def false_negative_v_structures(true_graph: DAG, learnt_graph: DAG) -> Set[Tuple[str, str, str]]:
    """Get the set of v-structures in true_graph but not learnt_graph."""
    _check_args(true_graph, learnt_graph)
    return true_graph.get_v_structures() - learnt_graph.get_v_structures()


def v_precision(true_graph: DAG, learnt_graph: DAG) -> float:
    """Compute the v-structure precision of a learnt graph."""
    _check_args(true_graph, learnt_graph)
    true_pos = len(true_positive_v_structures(true_graph, learnt_graph))
    all_learnt = len(learnt_graph.get_v_structures())
    return 0.0 if (all_learnt == 0) else true_pos / all_learnt


def v_recall(true_graph: DAG, learnt_graph: DAG) -> float:
    """Compute the v-structure recall of a learnt graph."""
    _check_args(true_graph, learnt_graph)
    true_pos = len(true_positive_v_structures(true_graph, learnt_graph))
    false_neg = len(false_negative_v_structures(true_graph, learnt_graph))
    return 0.0 if (true_pos + false_neg == 0.0) else true_pos / (true_pos + false_neg)


def v_f1(true_graph: DAG, learnt_graph: DAG) -> float:
    """Compute the v-structure F1 score of a learnt graph."""
    _check_args(true_graph, learnt_graph)
    prec = v_precision(true_graph, learnt_graph)
    rec = v_recall(true_graph, learnt_graph)
    return 0.0 if (prec + rec == 0.0) else (2 * prec * rec) / (prec + rec)
