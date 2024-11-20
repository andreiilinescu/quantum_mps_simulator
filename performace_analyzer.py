from mps_simulator import MpsSimulator
from one_query_simulator import OneQuerySimulator
import numpy as np
from db_contraction import duckdb_contraction
from utils import convert_to_einsum

def get_time_average(nr_iter, nr_qbits, gates,simulator_method=MpsSimulator.run,db_contraction=None):
    times=np.empty(nr_iter)
    for i in range(nr_iter):
       tmp=simulator_method(nr_qbits,gates,db_contraction).get_times()
       times[i]=tmp.mean()
    return times


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    get_time_average(10,2,[('h',0),('cnot',0,1)])
    t=get_time_average(100,2,[('h',0),('cnot',0,1)],db_contraction=duckdb_contraction)
    t2=get_time_average(100,2,[('h',0),('cnot',0,1)],simulator_method=OneQuerySimulator.run,db_contraction=duckdb_contraction)
    print("mps")
    print(t)
    print("one_query")
    print(t2)