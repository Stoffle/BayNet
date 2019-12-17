from typing import Set, Dict, Tuple
from .structure import Graph


def _check_args(true_graph: Graph, learnt_graph: Graph, skeleton: bool = False) -> None:
    if not isinstance(true_graph, Graph):
        raise ValueError(f"Expected `true_graph` to be a Graph object, got {type(true_graph)}")
    if not true_graph.is_directed() and not skeleton:
        raise ValueError("`true_graph` is undirected")
    if not isinstance(learnt_graph, Graph):
        raise ValueError(f"Expected `learnt_graph` to be a Graph object, got {type(learnt_graph)}")
    if not learnt_graph.is_directed() and not skeleton:
        raise ValueError("`learnt_graph` is undirected")
    return True


def false_positive_edges(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> Set[str]:
    _check_args(true_graph, learnt_graph, skeleton)
    if not skeleton:
        return learnt_graph.edges - true_graph.edges
    else:
        return learnt_graph.skeleton_edges - true_graph.skeleton_edges


def true_positive_edges(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> Set[str]:
    _check_args(true_graph, learnt_graph, skeleton)
    if not skeleton:
        return true_graph.edges & learnt_graph.edges
    else:
        return true_graph.skeleton_edges & learnt_graph.skeleton_edges


def false_negative_edges(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> Set[str]:
    _check_args(true_graph, learnt_graph, skeleton)
    if not skeleton:
        return true_graph.edges - learnt_graph.edges
    else:
        return true_graph.skeleton_edges - learnt_graph.skeleton_edges


def precision(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> float:
    _check_args(true_graph, learnt_graph, skeleton)
    TP = len(true_positive_edges(true_graph, learnt_graph, skeleton))
    all_learnt = len(learnt_graph.edges) if not skeleton else len(learnt_graph.skeleton_edges)
    return 0.0 if (all_learnt == 0) else TP / all_learnt


def recall(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> float:
    args = (true_graph, learnt_graph, skeleton)
    _check_args(*args)
    TP = len(true_positive_edges(*args))
    FN = len(false_negative_edges(*args))
    return 0.0 if (TP + FN == 0.0) else TP / (TP + FN)


def f1(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> float:
    args = (true_graph, learnt_graph, skeleton)
    _check_args(*args)
    prec = precision(*args)
    rec = recall(*args)
    return 0.0 if (prec + rec == 0.0) else (2 * prec * rec) / (prec + rec)


def shd(true_graph: Graph, learnt_graph: Graph, skeleton: bool) -> int:
    _check_args(true_graph, learnt_graph)
    missing = len(false_negative_edges(true_graph, learnt_graph, True)) / 2
    added = len(false_positive_edges(true_graph, learnt_graph, True)) / 2
    backwards = 0 if skeleton else len(true_graph.reversed_edges & learnt_graph.edges)
    return missing + added + backwards


def false_positive_v_structures(
    true_graph: Graph, learnt_graph: Graph
) -> Set[Tuple[str, str, str]]:
    _check_args(true_graph, learnt_graph)
    return learnt_graph.get_v_structures() - true_graph.get_v_structures()


def true_positive_v_structures(true_graph: Graph, learnt_graph: Graph) -> Set[Tuple[str, str, str]]:
    _check_args(true_graph, learnt_graph)
    return learnt_graph.get_v_structures() & true_graph.get_v_structures()


def false_negative_v_structures(
    true_graph: Graph, learnt_graph: Graph
) -> Set[Tuple[str, str, str]]:
    _check_args(true_graph, learnt_graph)
    return true_graph.get_v_structures() - learnt_graph.get_v_structures()


def v_precision(true_graph: Graph, learnt_graph: Graph) -> float:
    _check_args(true_graph, learnt_graph)
    TP = len(true_positive_v_structures(true_graph, learnt_graph))
    all_learnt = len(learnt_graph.get_v_structures())
    return 0.0 if (all_learnt == 0) else TP / all_learnt


def v_recall(true_graph: Graph, learnt_graph: Graph) -> float:
    _check_args(true_graph, learnt_graph)
    TP = len(true_positive_v_structures(true_graph, learnt_graph))
    FN = len(false_negative_v_structures(true_graph, learnt_graph))
    return 0.0 if (TP + FN == 0.0) else TP / (TP + FN)


def v_f1(true_graph: Graph, learnt_graph: Graph) -> float:
    _check_args(true_graph, learnt_graph)
    prec = v_precision(true_graph, learnt_graph)
    rec = v_recall(true_graph, learnt_graph)
    return 0.0 if (prec + rec == 0.0) else (2 * prec * rec) / (prec + rec)
