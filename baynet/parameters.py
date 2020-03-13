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
        self.parents = np.array([parent.index for parent in node.neighbors(mode="in")], dtype=int)
        self.parent_names = [parent['name'] for parent in node.neighbors(mode="in")]
        if any([pl is None for pl in parent_levels]):
            raise ValueError(f"Parent of {node['name']} missing attribute 'levels'")

        node_levels = node['levels']
        if node_levels is None:
            raise ValueError(f"Node {node['name']} missing attribute 'levels'")

        self.array = np.zeros([*parent_levels, node_levels], dtype=float)
        self._levels = node_levels

    def __getitem__(self, indexer: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Return CPT row corresponding to given indexer.

        Wraps the stored array's __getitem___.
        """
        return self.array[indexer]

    def rescale_probabilities(self) -> None:
        """
        Rescale probability table rows.

        Set any variables with no probabilities to be uniform,
        scale CPT rows to sum to 1, then compute cumulative sums
        to make sampling faster.
        """
        # Anywhere with sum(probs) == 0, we set to all 1 prior to scaling
        self.array[self.array.sum(axis=-1) == 0] = 1
        # Rescale probabilities to sum to 1
        self.array /= np.expand_dims(self.array.sum(axis=-1), axis=-1)
        self.array = self.array.cumsum(axis=-1)
        self._scaled = True

    def sample(self, incomplete_data: np.ndarray) -> np.ndarray:
        """Sample based on parent values."""
        if not self._scaled:
            raise ValueError("CPT not scaled; use .rescale_probabilities() before sampling")
        parent_values = incomplete_data[:, self.parents]
        random_vector = np.random.uniform(size=parent_values.shape[0])
        parent_values: List[Tuple[int, ...]] = list(map(tuple, parent_values))
        return _sample_cpt(self.array, parent_values, random_vector)

    def sample_parameters(self) -> None:
        """Sample CPT from dirichlet distribution."""
        raise NotImplementedError


def _sample_cpt(
    cpt: np.ndarray, parent_values: List[Tuple[int, ...]], random_vector: np.ndarray
) -> np.ndarray:
    """Sample given cpt based on rows of parent values and random vector."""
    out_vector = np.zeros(random_vector.shape)
    for row_idx in range(random_vector.shape[0]):
        probs = cpt[parent_values[row_idx]]
        out_vector[row_idx] = np.argmax(random_vector[row_idx] < probs)
    return out_vector


class ConditionalProbabilityDistribution:
    """Conditional probability distribution for continuous data."""

    def __init__(self, node: igraph.Vertex, noise_scale: float = 1.0) -> None:
        """Initialise a conditional probability table."""
        self.noise_scale = noise_scale
        self.parents = np.array([parent.index for parent in node.neighbors(mode="in")], dtype=int)
        self.parent_names = [parent['name'] for parent in node.neighbors(mode="in")]
        self._n_parents = len(self.parents)
        self.array = np.zeros(self._n_parents, dtype=float)

    def sample_parameters(
        self, weights: Union[List[float], Tuple[float, ...]] = (-2.0, -0.5, 0.5, 2.0)
    ) -> None:
        """Sample parent weights uniformly from defined possible values."""
        self.array = np.random.choice(weights, self._n_parents)

    def sample(self, incomplete_data: np.ndarray) -> np.ndarray:
        """Sample column based on parent columns in incomplete data matrix."""
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=incomplete_data.shape[0])
        if self._n_parents == 0:
            return noise
        parent_values = incomplete_data[:, self.parents]
        return parent_values.dot(self.array) + noise
