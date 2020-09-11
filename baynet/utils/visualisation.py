"""Graph visualisation functions."""
from pathlib import Path

import igraph
import graphviz


class GraphComparison(igraph.Graph):
    """Union of graph_a and graph_b, with edges assigned colours for plotting."""

    # pylint: disable=not-an-iterable, unsupported-assignment-operation
    def __init__(
        self,
        graph_a: igraph.Graph,
        graph_b: igraph.Graph,
        nodes: list,
        a_not_b_col: str = "red",
        b_not_a_col: str = "blue",
        reversed_in_b_col: str = "green",
        both_col: str = "black",
        line_width: int = 2,
    ):
        """Create comparison graph."""
        super().__init__(
            directed=True,
            vertex_attrs={'fontsize': None, 'fontname': None, 'label': None},
            edge_attrs={'color': None, 'penwidth': None, 'style': None},
        )
        self.line_width = line_width
        self.add_vertices(nodes)
        self.vs['label'] = nodes
        self.vs['fontsize'] = 30
        self.vs['fontname'] = "Helvetica"

        self.add_edges(graph_a.edges & graph_b.edges)
        self.colour_uncoloured(both_col)

        self.add_edges(graph_a.edges.difference(graph_b.skeleton_edges))
        self.colour_uncoloured(a_not_b_col)

        self.add_edges(graph_b.edges.difference(graph_a.skeleton_edges))
        self.colour_uncoloured(b_not_a_col)

        self.add_edges(
            graph_a.edges.intersection(graph_b.skeleton_edges).difference(
                graph_a.edges.intersection(graph_b.edges)
            )
        )
        self.colour_uncoloured(reversed_in_b_col)

    def colour_uncoloured(self, colour: str) -> None:
        """Colour edges not yet given a colour."""
        for edge in self.es:
            if edge['color'] is None:
                edge['color'] = colour
                if colour == "red":
                    edge['style'] = "dashed"
                else:
                    edge['style'] = "solid"
                edge['penwidth'] = self.line_width

    def plot(self, path: Path = Path().parent / 'comparison.png') -> None:
        """Save a graphviz plot of comparison."""
        draw_graph(self, path)


def draw_graph(
    graph: igraph.Graph,
    save_path: Path = Path().parent / 'graph.png',
) -> None:
    """Save a graphviz plot of a given graph."""
    temp_path = save_path.parent / 'temp.dot'
    with open(temp_path, 'w') as temp_file:
        graph.write_dot(temp_file)
    graphviz_source = graphviz.Source.from_file(temp_path)
    temp_path.unlink()

    with open(save_path, 'wb') as save_file:
        save_file.write(graphviz_source.pipe(format=save_path.suffix.strip('.')))
