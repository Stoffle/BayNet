"""Graph visualisation functions."""
import io
from pathlib import Path
from typing import Any, Dict, Optional

import graphviz
import igraph
import numpy as np
from matplotlib import pylab
from matplotlib.lines import Line2D
from PIL import Image


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
            vertex_attrs={"fontsize": None, "fontname": None, "label": None},
            edge_attrs={"color": None, "penwidth": None, "style": None},
        )
        self.line_width = line_width
        self.add_vertices(nodes)
        self.vs["label"] = nodes
        self.vs["fontsize"] = 30
        self.vs["fontname"] = "Helvetica"

        self._a_not_b_col = a_not_b_col
        self._b_not_a_col = b_not_a_col
        self._reversed_in_b_col = reversed_in_b_col

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
            if edge["color"] is None:
                edge["color"] = colour
                if colour == "red":
                    edge["style"] = "dashed"
                else:
                    edge["style"] = "solid"
                edge["penwidth"] = self.line_width

    def plot(self, path: Path = Path().parent / "comparison.png") -> None:
        """Save a graphviz plot of comparison."""
        legend_kwargs = {
            "a not b": {"ls": "--", "c": self._a_not_b_col},
            "b not a": {"ls": "-", "c": self._b_not_a_col},
            "reversed in b": {"ls": "-", "c": self._reversed_in_b_col},
        }
        draw_graph(self, path, legend_kwargs=legend_kwargs)


def draw_graph(
    graph: igraph.Graph,
    save_path: Path = Path().parent / "graph.png",
    legend_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Save a graphviz plot of a given graph."""
    save_format = save_path.suffix.strip(".")
    if legend_kwargs and save_format and save_format == "svg":
        raise ValueError("File cannot be saved in the SVG format when a legend is specified.")
    temp_path = save_path.parent / "temp.dot"

    with open(temp_path, "w") as temp_file:
        graph.write_dot(temp_file)
    graphviz_source = graphviz.Source.from_file(temp_path)
    temp_path.unlink()

    if legend_kwargs:
        buf = io.BytesIO()
        buf.write(graphviz_source.pipe(format="png"))
        buf.seek(0)
        chart_im = Image.open(buf)

        dpi = chart_im.size[1] // 3
        legend_markers = [Line2D([0], [0], **kwargs, lw=1.3) for kwargs in legend_kwargs.values()]
        chart_legend = pylab.figure(figsize=(1.45, 3), dpi=dpi)

        buf = io.BytesIO()
        chart_legend.legend(legend_markers, legend_kwargs.keys(), loc="center", frameon=False)
        chart_legend.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)

        legend_im = Image.open(buf)
        if legend_im.size[1] != chart_im.size[1]:
            legend_im = legend_im.resize((legend_im.size[0], chart_im.size[1]))

        comp_im = np.concatenate([legend_im, chart_im], axis=1)
        comp_im = Image.fromarray(comp_im)
        comp_im.save(save_path, format="png")
        comp_im.show()

        buf.close()
    else:
        with open(save_path, "wb") as save_file:
            save_file.write(graphviz_source.pipe(format=save_path.suffix.strip(".")))
