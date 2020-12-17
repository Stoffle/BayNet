import pytest
from baynet import metrics


def test_check_args(test_dag, reversed_dag):
    test_dag = test_dag
    test_dag.graph.to_undirected()
    reversed_dag = reversed_dag
    assert metrics._check_args(reversed_dag, reversed_dag, False)
    assert metrics._check_args(test_dag, reversed_dag, True)
    with pytest.raises(ValueError):
        metrics._check_args(test_dag, reversed_dag, False)
    with pytest.raises(ValueError):
        metrics._check_args(reversed_dag, test_dag, False)
    with pytest.raises(ValueError):
        metrics._check_args(None, reversed_dag, True)
    with pytest.raises(ValueError):
        metrics._check_args(reversed_dag, None, True)


def test_false_positive_edges(test_dag, reversed_dag):
    assert metrics.false_positive_edges(test_dag, test_dag, True) == set()
    assert metrics.false_positive_edges(test_dag, test_dag, False) == set()
    assert metrics.false_positive_edges(test_dag, reversed_dag, True) == set()
    assert metrics.false_positive_edges(test_dag, reversed_dag, False) == reversed_dag.edges


def test_true_positive_edges(test_dag, reversed_dag):
    assert (
        metrics.true_positive_edges(test_dag, test_dag, True)
        == test_dag.edges | test_dag.reversed_edges
    )
    assert metrics.true_positive_edges(test_dag, test_dag, False) == test_dag.edges
    assert (
        metrics.true_positive_edges(test_dag, reversed_dag, True)
        == test_dag.edges | test_dag.reversed_edges
    )
    assert metrics.true_positive_edges(test_dag, reversed_dag, False) == set()


def test_precision(test_dag, reversed_dag, partial_dag):
    assert metrics.precision(test_dag, test_dag, True) == 1.0
    assert metrics.precision(test_dag, test_dag, False) == 1.0
    assert metrics.precision(test_dag, partial_dag, True) == 1.0
    assert metrics.precision(test_dag, partial_dag, False) == 1.0
    assert metrics.precision(reversed_dag, partial_dag, True) == 1.0
    assert metrics.precision(reversed_dag, partial_dag, False) == 0.0


def test_recall(test_dag, reversed_dag, partial_dag):
    test_dag = test_dag
    reversed_dag = reversed_dag
    partial_dag = partial_dag
    assert metrics.recall(test_dag, test_dag, True) == 1.0
    assert metrics.recall(test_dag, test_dag, False) == 1.0
    assert metrics.recall(test_dag, reversed_dag, True) == 1.0
    assert metrics.recall(test_dag, reversed_dag, False) == 0.0
    assert metrics.recall(test_dag, partial_dag, True) == 2 / 3
    assert metrics.recall(test_dag, partial_dag, False) == 2 / 3


def test_f1_score(test_dag, reversed_dag, partial_dag):
    assert metrics.f1_score(test_dag, test_dag, True) == 1.0
    assert metrics.f1_score(test_dag, reversed_dag, True) == 1.0
    assert metrics.f1_score(test_dag, test_dag, False) == 1.0
    assert metrics.f1_score(test_dag, reversed_dag, False) == 0.0


def test_dag_shd(test_dag, reversed_dag, partial_dag):
    assert metrics.shd(test_dag, test_dag, False) == 0
    assert metrics.shd(test_dag, reversed_dag, False) == 3
    assert metrics.shd(test_dag, partial_dag, False) == 1


def test_skeleton_shd(test_dag, reversed_dag, partial_dag):
    assert metrics.shd(test_dag, test_dag, True) == 0
    assert metrics.shd(test_dag, reversed_dag, True) == 0
    assert metrics.shd(test_dag, partial_dag, True) == 1


def test_false_positive_v_structures(test_dag, reversed_dag, partial_dag):
    assert metrics.false_positive_v_structures(test_dag, reversed_dag) == set()
    assert metrics.false_positive_v_structures(test_dag, partial_dag) == {("C", "B", "D")}
    assert metrics.false_positive_v_structures(reversed_dag, partial_dag) == {("C", "B", "D")}


def test_true_positive_v_structures(test_dag, reversed_dag, partial_dag):
    assert metrics.true_positive_v_structures(test_dag, reversed_dag) == set()
    assert metrics.true_positive_v_structures(test_dag, partial_dag) == set()
    assert metrics.true_positive_v_structures(test_dag, test_dag) == set()
    assert metrics.true_positive_v_structures(partial_dag, partial_dag) == {("C", "B", "D")}


def test_false_negative_v_structures(test_dag, reversed_dag, partial_dag):
    assert metrics.false_negative_v_structures(test_dag, reversed_dag) == set()
    assert metrics.false_negative_v_structures(test_dag, partial_dag) == set()
    assert metrics.false_negative_v_structures(partial_dag, test_dag) == {("C", "B", "D")}


def test_v_precision(test_dag, reversed_dag, partial_dag):
    assert metrics.v_precision(test_dag, reversed_dag) == 0.0
    assert metrics.v_precision(test_dag, partial_dag) == 0.0
    assert metrics.v_precision(partial_dag, partial_dag) == 1.0


def test_v_recall(test_dag, reversed_dag, partial_dag):
    assert metrics.v_recall(test_dag, reversed_dag) == 0.0
    assert metrics.v_recall(partial_dag, partial_dag) == 1.0


def test_v_f1(test_dag, reversed_dag, partial_dag):
    assert metrics.v_f1(test_dag, reversed_dag) == 0.0
    assert metrics.v_f1(partial_dag, partial_dag) == 1.0
