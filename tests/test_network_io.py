from baynet.utils.network_io import load_dag, save_dag
from utils import test_dag



def test_load_dag():
    return

def test_save_dag():
    dag = test_dag()
    dag.generate_continuous_parameters(possible_weights=[1])
    return dag.__dict__

if __name__ == "__main__":
    d = test_save_dag()
    print(d)