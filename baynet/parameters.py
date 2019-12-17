"""Parameter tables for Graph objects."""
from typing import List, Tuple, Union
import numpy as np
import igraph


class ConditionalProbabilityTable:
    """Conditional probability table for categorical data."""

    def __init__(self, node: igraph.Vertex) -> None:
        """Initialise a conditional probability table."""
        self._scaled = False
        # sorted_parents = sorted(node.neighbors(mode="in"), key = lambda x: x['name'])
        # print(sorted_parents)
        parent_levels = [v['levels'] for v in node.neighbors(mode="in")]
        self._n_parents = len(parent_levels)
        if any([pl is None for pl in parent_levels]):
            raise ValueError(f"Parent of {node['name']} missing attribute 'levels'")

        node_levels = node['levels']
        if node_levels is None:
            raise ValueError(f"Node {node['name']} missing attribute 'levels'")

        self._array = np.zeros([*parent_levels, node_levels], dtype=float)
        self._levels = node_levels

    def __getitem__(self, indexer: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Return CPT row corresponding to given indexer.

        Wraps the stored array's __getitem___.
        """
        return self._array[indexer]

    def rescale_probabilities(self) -> None:
        """
        Rescale probability table rows.

        Set any variables with no probabilities to be uniform,
        scale CPT rows to sum to 1, then compute cumulative sums
        to make sampling faster.
        """
        # Anywhere with sum(probs) == 0, we set to all 1 prior to scaling
        self._array[self._array.sum(axis=-1) == 0] = 1
        # Rescale probabilities to sum to 1
        self._array /= np.expand_dims(self._array.sum(axis=-1), axis=-1)
        self._array = self._array.cumsum(axis=-1)
        self._scaled = True

    def sample(self, parent_values: np.ndarray) -> np.ndarray:
        """Sample based on parent values."""
        if not self._scaled:
            raise ValueError("CPT use .rescale_probabilities() before sampling")
        if not parent_values.shape[1] == self._n_parents:
            raise ValueError("Parent values shape don't match number of parents")
        random_vector = np.random.uniform(size=parent_values.shape[0])
        parent_values: List[Tuple[int, ...]] = list(map(tuple, parent_values))
        return _sample_cpt(self._array, parent_values, random_vector)


def _sample_cpt(
    cpt: np.ndarray, parent_values: List[Tuple[int, ...]], random_vector: np.ndarray
) -> np.ndarray:
    """Sample given cpt based on rows of parent values and random vector."""
    out_vector = np.zeros(random_vector.shape)
    for row_idx in range(random_vector.shape[0]):
        probs = cpt[parent_values[row_idx]]
        out_vector[row_idx] = np.argmax(random_vector[row_idx] < probs)
    return out_vector
