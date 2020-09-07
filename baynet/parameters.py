"""Parameter tables for Graph objects."""
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import pandas as pd
import igraph


class ConditionalProbabilityTable:
    """Conditional probability table for categorical data."""

    def __init__(self, vertex: Optional[igraph.Vertex] = None) -> None:
        """Initialise a conditional probability table."""
        if vertex is None:
            return
        self.name = vertex["name"]
        self.parents = [str(v["name"]) for v in vertex.neighbors(mode="in")]
        parent_levels = [v["levels"] for v in vertex.neighbors(mode="in")]
        if any([pl is None for pl in parent_levels]):
            raise ValueError(f"Parent of {vertex['name']} missing attribute 'levels'")
        n_parent_levels = [len(v["levels"]) for v in vertex.neighbors(mode="in")]

        node_levels = vertex["levels"]
        if node_levels is None:
            raise ValueError(f"Node {vertex['name']} missing attribute 'levels'")
        self.levels = node_levels

        self.array = np.zeros([*n_parent_levels, len(node_levels)])
        self.cumsum_array = np.zeros([*n_parent_levels, len(node_levels)])

    @classmethod
    def estimate(
        cls,
        vertex: igraph.Vertex,
        data: pd.DataFrame,
        method: str = "mle",
        method_args: Optional[Dict[str, Union[int, float]]] = None,
    ) -> "ConditionalProbabilityTable":
        """Create a CPT, populated with predicted parameters based on supplied data."""
        cpt = cls(vertex)
        if not method_args:
            method_args = {}
        if method == "mle":
            cpt.mle_estimate(data)
        elif method == "dfe":
            cpt.dfe_estimate(data, **method_args)  # type: ignore
        else:
            raise NotImplementedError(f"Parameter Estimation method {method} not implemented.")
        return cpt

    def dfe_estimate(
        self, data: pd.DataFrame, iterations: int = 250, learning_rate: float = 0.01
    ) -> None:
        """Predict parameters using DFE method."""
        self.rescale_probabilities()
        for _, sample in (
            data.apply(lambda x: x.cat.codes).sample(n=iterations, replace=True).iterrows()
        ):
            p_cgp = np.zeros(len(self.levels))
            p_cgp[sample[self.name]] = 1
            loss = p_cgp - self.array[tuple(sample[self.parents])]
            self.array[tuple(sample[self.parents])] += loss * learning_rate
        self.rescale_probabilities()

    def mle_estimate(self, data: pd.DataFrame) -> None:
        """Predict parameters using the MLE method."""
        matches = data.apply(lambda x: x.cat.codes).groupby([*self.parents, self.name]).size()
        for indexer, count in matches.iteritems():
            self.array[indexer] = count
        self.rescale_probabilities()

    def rescale_probabilities(self) -> None:
        """
        Rescale probability table rows.

        Set any variables with no probabilities to be uniform,
        scale CPT rows to sum to 1, then compute cumulative sums
        to make sampling faster.
        """
        # Anywhere with sum(probs) == 0, we set to all 1 prior to scaling
        self.array[self.array.sum(axis=-1) == 0] = 1.0
        self.array = np.nan_to_num(self.array, nan=1e-8, posinf=1.0 - 1e-8)
        # Rescale probabilities to sum to 1
        self.array[self.array.sum(axis=-1) == 0] = 1.0
        self.array /= np.expand_dims(self.array.sum(axis=-1), axis=-1)
        self.array /= np.expand_dims(self.array.sum(axis=-1), axis=-1)
        self.cumsum_array = self.array.cumsum(axis=-1)

    def sample(self, incomplete_data: pd.DataFrame) -> pd.DataFrame:
        """Sample based on parent values."""
        parent_values_array = incomplete_data[self.parents].apply(lambda x: x.cat.codes).values
        random_vector = np.random.uniform(size=parent_values_array.shape[0])
        parent_values: List[Tuple[int, ...]] = list(map(tuple, parent_values_array))
        out_array = _sample_cpt(self.cumsum_array, parent_values, random_vector)
        dtype = pd.CategoricalDtype(self.levels, ordered=True)
        return pd.Categorical.from_codes(codes=out_array, dtype=dtype)

    def sample_parameters(
        self,
        alpha: Optional[float] = None,
        seed: Optional[int] = None,
        normalise_alpha: bool = True,
    ) -> None:
        """Sample CPT from dirichlet distribution."""
        if alpha is None:
            alpha = 20.0
        if seed is not None:
            np.random.seed(seed)
        parent_levels = int(np.prod(self.array.shape[:-1]))
        if normalise_alpha:
            alpha = np.max(np.array([0.01, alpha / (parent_levels * len(self.levels))]))
        self.array = np.random.dirichlet(
            np.array([alpha] * len(self.levels)), parent_levels
        ).reshape(self.array.shape)
        self.rescale_probabilities()

    def marginalise(self, parent: str) -> None:
        """Marginalise out a parent."""
        assert parent in self.parents
        parent_idx = self.parents.index(parent)
        del self.parents[parent_idx]

        self.array = self.array.sum(axis=parent_idx)
        self.rescale_probabilities()

    def intervene(self, value: str) -> None:
        """Fix CPT to only ever return `value`."""
        value_index = self.levels.index(value)
        self.array = np.zeros(len(self.levels))
        self.parents = []

        self.array[value_index] = 1.0
        self.rescale_probabilities()


def _sample_cpt(
    cpt: np.ndarray, parent_values: List[Tuple[int, ...]], random_vector: np.ndarray
) -> np.ndarray:
    """Sample given cpt based on rows of parent values and random vector."""
    out_vector = np.zeros(random_vector.shape).astype(int)
    for row_idx in range(random_vector.shape[0]):
        probs = cpt[parent_values[row_idx]]
        out_vector[row_idx] = np.argmax(random_vector[row_idx] < probs)
    return out_vector


class ConditionalProbabilityDistribution:
    """Conditional probability distribution for continuous data."""

    def __init__(
        self,
        node: Optional[igraph.Vertex] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> None:
        """Initialise a conditional probability table."""
        if mean is None:
            mean = 0.0
        self.mean = mean
        if std is None:
            std = 1.0
        self.std = std
        if node is None:
            return
        self.name = node["name"]
        self.parents = [str(parent["name"]) for parent in node.neighbors(mode="in")]
        self.array = np.zeros(len(self.parents))

    def sample_parameters(
        self, weights: Optional[List[float]] = None, seed: Optional[int] = None
    ) -> None:
        """Sample parent weights uniformly from defined possible values."""
        if seed is not None:
            np.random.seed(seed)
        if weights is None:
            weights = [-2.0, -0.5, 0.5, 2.0]
        self.array = np.random.choice(weights, len(self.parents))

    def sample(self, incomplete_data: pd.DataFrame) -> pd.DataFrame:
        """Sample column based on parent columns in incomplete data matrix."""
        noise = np.random.normal(loc=self.mean, scale=self.std, size=incomplete_data.shape[0])
        if len(self.parents) == 0:
            return noise
        parent_values = incomplete_data[self.parents]
        return parent_values.dot(self.array) + noise
