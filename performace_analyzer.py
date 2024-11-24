from mps_simulator import MpsSimulator
from one_query_simulator import OneQuerySimulator
import numpy as np
from db_contraction import duckdb_contraction
from utils import convert_to_einsum
from simulator import Simulator
from plotting import plot_multiple_lines
import tracemalloc

def get_time_means(nr_iter, nr_qbits, gates,simulator:Simulator=MpsSimulator,db_contraction=None):
    times=np.empty(nr_iter)
    for i in range(nr_iter):
       tmp=simulator.run(nr_qbits,gates,db_contraction).get_times()
       times[i]=tmp.sum()
    return np.mean(times)

def get_time_medians(nr_iter, nr_qbits, gates,simulator:Simulator=MpsSimulator,db_contraction=None):
    times=np.empty(nr_iter)
    for i in range(nr_iter):
        tmp=simulator.run(nr_qbits,gates,db_contraction).get_times()
        times[i]=tmp.sum()
    return np.median(times)
def get_memory_medians(nr_iter, nr_qbits, gates,simulator:Simulator=MpsSimulator,db_contraction=None):
    mems=np.empty(nr_iter)
    for i in range(nr_iter):
        tracemalloc.start()
        simulator.run(nr_qbits,gates,db_contraction)
        tmp=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mems[i]=np.median(tmp)
    return mems
def generate_ghz_gates(nr_qbits):
    gates=[('h',0)]
    for x in range(nr_qbits-1):
        gates.append(('cnot',x,x+1))
    return gates

NR_ITERS=100
def run_multiple_ghz(max_qbits,simulator_method:Simulator=MpsSimulator,db_contraction=None, tracker=get_time_medians):
    medians=[]
    for num_qbits in range(2,max_qbits+1):
        gates=generate_ghz_gates(num_qbits)
        times=tracker(NR_ITERS,num_qbits,gates,simulator=simulator_method,db_contraction=db_contraction)
        med=np.mean(times)
        medians.append(med)
    return medians
 
NR_QBITS=10
if __name__ == "__main__":
    print(get_memory_medians(50,10,generate_ghz_gates(10),MpsSimulator))
    # np.set_printoptions(suppress=True)
    # x=np.array(range(2,NR_QBITS+1))
    # y1=run_multiple_ghz(NR_QBITS,OneQuerySimulator,duckdb_contraction)
    # y2=run_multiple_ghz(NR_QBITS,MpsSimulator,duckdb_contraction)
    # label1="OneQuerySimulator, duckDB"
    # label2="HybridMpsSimulator, duckDB"
    # plot_multiple_lines(x,[y1,y2],[label1,label2],"Number of Qbits","Time (s)","GHZ State Time Comparison using DuckDB")
    # t=get_time_medians(100,2,[('h',0),('cnot',0,1)],db_contraction=duckdb_contraction)
    # t2=get_time_medians(100,2,[('h',0),('cnot',0,1)],simulator_method=OneQuerySimulator.run,db_contraction=duckdb_contraction)
    # print("mps")
    # print(t)
    # print("one_query")
    # print(t2)