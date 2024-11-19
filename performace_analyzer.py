from mps_simulator import MpsSimulator
import numpy as np
from db_contraction import duckdb_contraction
from utils import convert_to_einsum

def get_time_average(nr_iter, nr_qbits, gates,db_contraction=None):
    times=np.empty(nr_iter)
    for i in range(nr_iter):
       tmp=MpsSimulator.run(nr_qbits,gates,db_contraction).get_times()
       times[i]=tmp.mean()
    return times


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    get_time_average(10,2,[('h',0),('cnot',0,1)])
    get_time_average(10,2,[('h',0),('cnot',0,1)],duckdb_contraction)
    print(convert_to_einsum(2,[('h',0),('cnot',0,1)]))