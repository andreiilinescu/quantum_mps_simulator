from mps_simulator import MpsSimulator
from one_query_simulator import OneQuerySimulator
import numpy as np
from db_contraction import duckDB,abstractDB,sqliteDB
from utils import convert_to_einsum
from simulator import Simulator
from plotting import plot_multiple_lines
import tracemalloc
import json
def get_time_means(nr_iter, nr_qbits, gates,simulator:Simulator=MpsSimulator,db:abstractDB=None):
    times=np.empty(nr_iter)
    for i in range(nr_iter):
       tmp=simulator.run(nr_qbits,gates,db).get_times()
       times[i]=tmp.sum()
    return np.mean(times)

def get_time_medians(nr_iter, nr_qbits, gates,simulator:Simulator=MpsSimulator,db:abstractDB=None):
    times=np.empty(nr_iter)
    for i in range(nr_iter):
        tmp=simulator.run(nr_qbits,gates,db).get_times()
        times[i]=tmp.sum()
    return np.median(times)
def get_memory_medians(nr_iter, nr_qbits, gates,simulator:Simulator=MpsSimulator,db:abstractDB=None):
    mems=np.empty(nr_iter)
    for i in range(nr_iter):
        tracemalloc.start()
        simulator.run(nr_qbits,gates,db)
        tmp=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mems[i]=np.median(tmp)
    return mems

def generate_ghz_gates(nr_qbits):
    gates=[('h',0)]
    for x in range(nr_qbits-1):
        gates.append(('cnot',x,x+1))
    return gates

NR_ITERS=50
def run_multiple_ghz(max_qbits,simulator_method:Simulator=MpsSimulator,db:abstractDB=None, tracker=get_time_medians):
    medians=[]
    for num_qbits in range(2,max_qbits+1):
        gates=generate_ghz_gates(num_qbits)
        times=tracker(NR_ITERS,num_qbits,gates,simulator=simulator_method,db=db)
        med=np.mean(times)
        medians.append(med)
    return medians

def save_data_to_file(data,filename:str):
    if not filename.endswith(".json"):
        filename+=".json"
    with open("./data/"+filename, 'w') as f:
        json.dump(data, f)

NR_QBITS=100
if __name__ == "__main__":
    # print(get_memory_medians(50,10,generate_ghz_gates(10),MpsSimulator))
    np.set_printoptions(suppress=True)
    # y1=run_multiple_ghz(NR_QBITS,OneQuerySimulator,duckDB)
    y2=run_multiple_ghz(NR_QBITS,MpsSimulator,sqliteDB)
    data={"name": "HybridMps SQLITE GHZ Circuit Medians", "max_qubits": NR_QBITS, "num_iterations": NR_ITERS}
    data["times"]={i+2:x for i, x in enumerate(y2)}
    save_data_to_file(data,f"ghz_hybrid_{NR_QBITS}_{NR_ITERS}_median")
    # label1="OneQuerySimulator, duckDB"
    # label2="OneQuerySimulator, sqliteDB"
    # plot_multiple_lines(NR_QBITS,[y2],[label2],"Number of Qbits","Time (s)","GHZ State Time Comparison using SQLITE")
    # t=get_time_medians(100,2,[('h',0),('cnot',0,1)],db_contraction=duckdb_contraction)
    # t2=get_time_medians(100,2,[('h',0),('cnot',0,1)],simulator_method=OneQuerySimulator.run,db_contraction=duckdb_contraction)
    # print("mps")
    # print(t)
    # print("one_query")
    # print(t2)