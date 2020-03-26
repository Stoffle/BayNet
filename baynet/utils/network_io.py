"""Tools for saving and loading DAG objects."""
from pathlib import Path
from igraph import Vertex
import yaml
from baynet.structure import DAG

def _vertex_to_yaml(vertex: Vertex) -> str:
    return


def load_dag(path: Path) -> DAG:
    with path.open() as dag_file:
        dag_yaml = yaml.load(dag_file, Loader=yaml.SafeLoader)
    return dag_yaml

def save_dag(dag: DAG, path: Path) -> None:
    return

if __name__ == "__main__":
    dag = load_dag(Path(__file__).parent / 'example.yml')
    print(dag)
