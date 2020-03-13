"""Python Bayesian Network library."""
from baynet.structure import DAG

class Graph(DAG):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            DeprecationWarning(
                "baynet.Graph has been renamed baynet.DAG, Graph will be removed in a future release."
            )
        )
        super().__init__(*args, **kwargs)
