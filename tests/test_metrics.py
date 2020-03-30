import pytest
from baynet import metrics
from .utils import test_dag, partial_dag


def test_check_args():
    dag1 = test_dag()
    dag1.to_undirected()
    dag2 = test_dag(reverse=True)
    assert metrics._check_args(dag2, dag2, False)
    assert metrics._check_args(dag1, dag2, True)
    with pytest.raises(ValueError):
        metrics._check_args(dag1, dag2, False)
    with pytest.raises(ValueError):
        metrics._check_args(dag2, dag1, False)
    with pytest.raises(ValueError):
        metrics._check_args(None, dag2, True)
    with pytest.raises(ValueError):
        metrics._check_args(dag2, None, True)


def test_false_positive_edges():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    assert metrics.false_positive_edges(dag1, dag1, True) == set()
    assert metrics.false_positive_edges(dag1, dag1, False) == set()
    assert metrics.false_positive_edges(dag1, dag2, True) == set()
    assert metrics.false_positive_edges(dag1, dag2, False) == dag2.edges


def test_true_positive_edges():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    assert metrics.true_positive_edges(dag1, dag1, True) == dag1.edges | dag1.reversed_edges
    assert metrics.true_positive_edges(dag1, dag1, False) == dag1.edges
    assert metrics.true_positive_edges(dag1, dag2, True) == dag1.edges | dag1.reversed_edges
    assert metrics.true_positive_edges(dag1, dag2, False) == set()


def test_precision():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()
    assert metrics.precision(dag1, dag1, True) == 1.0
    assert metrics.precision(dag1, dag1, False) == 1.0
    assert metrics.precision(dag1, dag3, True) == 1.0
    assert metrics.precision(dag1, dag3, False) == 1.0
    assert metrics.precision(dag2, dag3, True) == 1.0
    assert metrics.precision(dag2, dag3, False) == 0.0


def test_recall():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()
    assert metrics.recall(dag1, dag1, True) == 1.0
    assert metrics.recall(dag1, dag1, False) == 1.0
    assert metrics.recall(dag1, dag2, True) == 1.0
    assert metrics.recall(dag1, dag2, False) == 0.0
    assert metrics.recall(dag1, dag3, True) == 2 / 3
    assert metrics.recall(dag1, dag3, False) == 2 / 3


def test_f1_score():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()
    assert metrics.f1_score(dag1, dag1, True) == 1.0
    assert metrics.f1_score(dag1, dag2, True) == 1.0
    assert metrics.f1_score(dag1, dag1, False) == 1.0
    assert metrics.f1_score(dag1, dag2, False) == 0.0


def test_dag_shd():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.shd(dag1, dag1, False) == 0
    assert metrics.shd(dag1, dag2, False) == 3
    assert metrics.shd(dag1, dag3, False) == 1


def test_skeleton_shd():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.shd(dag1, dag1, True) == 0
    assert metrics.shd(dag1, dag2, True) == 0
    assert metrics.shd(dag1, dag3, True) == 1


def test_false_positive_v_structures():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.false_positive_v_structures(dag1, dag2) == set()
    assert metrics.false_positive_v_structures(dag1, dag3) == {("C", "B", "D")}
    assert metrics.false_positive_v_structures(dag2, dag3) == {("C", "B", "D")}


def test_true_positive_v_structures():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.true_positive_v_structures(dag1, dag2) == set()
    assert metrics.true_positive_v_structures(dag1, dag3) == set()
    assert metrics.true_positive_v_structures(dag1, dag1) == set()
    assert metrics.true_positive_v_structures(dag3, dag3) == {("C", "B", "D")}


def test_false_negative_v_structures():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.false_negative_v_structures(dag1, dag2) == set()
    assert metrics.false_negative_v_structures(dag1, dag3) == set()
    assert metrics.false_negative_v_structures(dag3, dag1) == {("C", "B", "D")}


def test_v_precision():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.v_precision(dag1, dag2) == 0.0
    assert metrics.v_precision(dag1, dag3) == 0.0
    assert metrics.v_precision(dag3, dag3) == 1.0


def test_v_recall():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.v_recall(dag1, dag2) == 0.0
    assert metrics.v_recall(dag3, dag3) == 1.0


def test_v_f1():
    dag1 = test_dag()
    dag2 = test_dag(reverse=True)
    dag3 = partial_dag()

    assert metrics.v_f1(dag1, dag2) == 0.0
    assert metrics.v_f1(dag3, dag3) == 1.0
