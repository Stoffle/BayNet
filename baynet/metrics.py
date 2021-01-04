"""Metrics for comparing Graph objects."""
from typing import Set, Tuple
from .structure import DAG


def _check_args(true_dag: DAG, learnt_dag: DAG, skeleton: bool = False) -> bool:
    if not isinstance(true_dag, DAG):
        raise ValueError(f"Expected `true_dag` to be a DAG object, got {type(true_dag)}")
    if not true_dag.is_directed() and not skeleton:
        raise ValueError("`true_dag` is undirected")
    if not isinstance(learnt_dag, DAG):
        raise ValueError(f"Expected `learnt_dag` to be a DAG object, got {type(learnt_dag)}")
    if not learnt_dag.is_directed() and not skeleton:
        raise ValueError("`learnt_dag` is undirected")
    return True


def false_positive_edges(
    true_dag: DAG, learnt_dag: DAG, skeleton: bool = False
) -> Set[Tuple[str, str]]:
    """Get the set of edges in learnt_dag but not true_dag."""
    _check_args(true_dag, learnt_dag, skeleton)
    if not skeleton:
        return learnt_dag.edges - true_dag.edges
    return learnt_dag.skeleton_edges - true_dag.skeleton_edges


def true_positive_edges(
    true_dag: DAG, learnt_dag: DAG, skeleton: bool = False
) -> Set[Tuple[str, str]]:
    """Get the set of edges in both learnt_dag and true_dag."""
    _check_args(true_dag, learnt_dag, skeleton)
    if not skeleton:
        return true_dag.edges & learnt_dag.edges
    return true_dag.skeleton_edges & learnt_dag.skeleton_edges


def false_negative_edges(
    true_dag: DAG, learnt_dag: DAG, skeleton: bool = False
) -> Set[Tuple[str, str]]:
    """Get the set of edges in true_dag but not learnt_dag."""
    _check_args(true_dag, learnt_dag, skeleton)
    if not skeleton:
        return true_dag.edges - learnt_dag.edges
    return true_dag.skeleton_edges - learnt_dag.skeleton_edges


def precision(true_dag: DAG, learnt_dag: DAG, skeleton: bool = False) -> float:
    """Compute (DAG or skeleton) precision of a learnt graph."""
    _check_args(true_dag, learnt_dag, skeleton)
    true_pos = len(true_positive_edges(true_dag, learnt_dag, skeleton))
    all_learnt = len(learnt_dag.edges) if not skeleton else len(learnt_dag.skeleton_edges)
    return 0.0 if (all_learnt == 0.0) else true_pos / all_learnt


def recall(true_dag: DAG, learnt_dag: DAG, skeleton: bool = False) -> float:
    """Compute (DAG or skeleton) recall of a learnt graph."""
    args = (true_dag, learnt_dag, skeleton)
    _check_args(*args)
    true_pos = len(true_positive_edges(*args))
    false_neg = len(false_negative_edges(*args))
    return 0.0 if (true_pos + false_neg == 0.0) else true_pos / (true_pos + false_neg)


def f1_score(true_dag: DAG, learnt_dag: DAG, skeleton: bool = False) -> float:
    """Compute F1 score (DAG or skeleton) of a learnt graph."""
    args = (true_dag, learnt_dag, skeleton)
    _check_args(*args)
    prec = precision(*args)
    rec = recall(*args)
    return 0.0 if (prec + rec == 0.0) else (2 * prec * rec) / (prec + rec)


def shd(true_dag: DAG, learnt_dag: DAG, skeleton: bool = False) -> int:
    """Compute structural hamming distance (DAG or skeleton) of a learnt graph."""
    _check_args(true_dag, learnt_dag)
    missing = len(false_negative_edges(true_dag, learnt_dag, True)) / 2
    added = len(false_positive_edges(true_dag, learnt_dag, True)) / 2
    backwards = 0 if skeleton else len(true_dag.reversed_edges & learnt_dag.edges)
    return int(missing + added + backwards)


def false_positive_v_structures(true_dag: DAG, learnt_dag: DAG) -> Set[Tuple[str, str, str]]:
    """Get the set of v-structures in learnt_dag but not true_dag."""
    _check_args(true_dag, learnt_dag)
    return learnt_dag.get_v_structures() - true_dag.get_v_structures()


def true_positive_v_structures(true_dag: DAG, learnt_dag: DAG) -> Set[Tuple[str, str, str]]:
    """Get the set of v-structures in both learnt_dag and true_dag."""
    _check_args(true_dag, learnt_dag)
    return learnt_dag.get_v_structures() & true_dag.get_v_structures()


def false_negative_v_structures(true_dag: DAG, learnt_dag: DAG) -> Set[Tuple[str, str, str]]:
    """Get the set of v-structures in true_dag but not learnt_dag."""
    _check_args(true_dag, learnt_dag)
    return true_dag.get_v_structures() - learnt_dag.get_v_structures()


def v_precision(true_dag: DAG, learnt_dag: DAG) -> float:
    """Compute the v-structure precision of a learnt graph."""
    _check_args(true_dag, learnt_dag)
    true_pos = len(true_positive_v_structures(true_dag, learnt_dag))
    all_learnt = len(learnt_dag.get_v_structures())
    return 0.0 if (all_learnt == 0) else true_pos / all_learnt


def v_recall(true_dag: DAG, learnt_dag: DAG) -> float:
    """Compute the v-structure recall of a learnt graph."""
    _check_args(true_dag, learnt_dag)
    true_pos = len(true_positive_v_structures(true_dag, learnt_dag))
    false_neg = len(false_negative_v_structures(true_dag, learnt_dag))
    return 0.0 if (true_pos + false_neg == 0.0) else true_pos / (true_pos + false_neg)


def v_f1(true_dag: DAG, learnt_dag: DAG) -> float:
    """Compute the v-structure F1 score of a learnt graph."""
    _check_args(true_dag, learnt_dag)
    prec = v_precision(true_dag, learnt_dag)
    rec = v_recall(true_dag, learnt_dag)
    return 0.0 if (prec + rec == 0.0) else (2 * prec * rec) / (prec + rec)
