from typing import List, Optional
from collections import defaultdict
from pathlib import Path

from igraph import Graph
from baynet import DAG
from baynet.utils.visualisation import draw_graph


class GraphEnsemble:
    "Combine a list of BayNet DAG objects into an ensemble, with edge weights for visualising."
    # pylint: disable=unsupported-assignment-operation
    def __init__(self, dags: Optional[List[DAG]] = None):
        """Initialise from list of DAGs."""
        self.nodes = set()
        self.edge_counts = defaultdict(int)
        self.dag_count = 0
        if dags is not None:
            for dag in dags:
                self.add_dag(dag)

    def add_dag(self, dag: DAG):
        """Add a DAG to the ensemble."""
        self.nodes |= dag.nodes
        self.dag_count += 1
        for edge in dag.edges:
            self.edge_counts[edge] += 1

    def generate_graph(self):
        """Generate Graph from GraphEnsemble with edges weighted by their count."""
        graph = Graph(directed=True)
        graph.add_vertices(list(self.nodes))
        for edge, count in self.edge_counts.items():
            graph.add_edge(edge[0], edge[1], penwidth=count/self.dag_count, label=count)
        graph.vs['label'] = graph.vs['name']
        graph.vs['fontsize'] = 30
        graph.vs['fontname'] = "Helvetica"
        graph.es['color'] = "black"
        graph.es['style'] = "solid"
        return graph

    def plot(self, path: Optional[Path] = Path().resolve() / 'DAG.png') -> None:
        """Save plot of GraphEnsemble to specified path."""
        draw_graph(self.generate_graph(), path)


