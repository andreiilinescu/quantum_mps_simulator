from sqlite_mps import SQLITE_MPS
from timeit import default_timer as timer
import numpy as np
import json
from plotting import plot_multiple_lines
from termcolor import colored
import tracemalloc
def generate_ghz_circuit(num_qubits:int) -> dict:
    data={ "number_of_qubits":num_qubits,"gates":[]}
    data["gates"].append({ "qubits": [0],"gate": "H"})
    for i in range(num_qubits-1):
        data['gates'].append({ "qubits": [i, i+1],"gate": "CNOT"})
    return data

def time_ghz_execution(max_qubit:int , num_iterations:int, keep_only_sum=True):
    data={"name":"sqlite_mps GHZ Circuit","max_qubits":max_qubit,"num_iterations":num_iterations,"times":{}}
    for q in range(2,max_qubit+1):
        circ=generate_ghz_circuit(q)
        data["times"][str(q)]=[]
        for i in range(num_iterations):
            s=SQLITE_MPS.run_circuit_json(circ)
            if keep_only_sum:
                data["times"][str(q)].append(sum(s.times))
    return data
def get_medians(data:dict):
    n=data["max_qubits"]
    times=data["times"]
    out={}
    for i in range(2,n+1):
        tmp=times[str(i)]
        out[str(i)]=np.median(tmp)
    return out

def save_data_to_file(data,filename:str):
    if not filename.endswith(".json"):
        filename+=".json"
    with open("./new_data/"+filename, 'w') as f:
        json.dump(data, f)


def get_current_median_time(qbits_num:int,iters:int):
    circ=generate_ghz_circuit(qbits_num)
    times=[]
    for i in range(iters):
        tmp=SQLITE_MPS.run_circuit_json(circ).times
        times.append(sum(tmp))
    return np.median(times)

def get_current_median_memory(qbits_num:int,iters:int):
    circ=generate_ghz_circuit(qbits_num)
    mems=[]
    for i in range(iters):
        tracemalloc.start()
        SQLITE_MPS.run_circuit_json(circ)
        curr,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mems.append(peak)
    return np.median(mems)

def get_current_baseline(num_qubits,iters):
    data={"circuit_name":"GHZ","simulator":"SQLITE NO UDF","num_qbits":num_qubits,"iterations":num_qubits,"time":None,"memory_peak":None}
    data["memory_peak"]=get_current_median_memory(num_qubits,iters)
    data["time"]=get_current_median_time(num_qubits,iters)

    return data

def compare_current_baseline(old:dict):
    curr=get_current_baseline(old["num_qbits"],old["iterations"])
    print(f"Baseline Time:{old["time"]}")
    print(f"Baseline Memory Peak:{old["memory_peak"]}")
    print("")
    print(f"Current Time:{curr["time"]}")
    print(f"Current Memory Peak:{curr["memory_peak"]}")
    print("--------------")
    time_diff=curr["time"]-old["time"]
    mem_diff=curr["memory_peak"]-old["memory_peak"]
    if time_diff<=0:
        print(colored(f"Time:{time_diff}", "green"))
    else:
        print(colored(f"Time:{time_diff}", "red"))
    if mem_diff<=0:
        print(colored(f"Memory Peak:{mem_diff}", "green"))
    else:
        print(colored(f"Memory Peak:{mem_diff}", "red"))

MAX_QBITS=50
ITER=500
SYSTEM="MAC"
def plot_save_data_ghz():
    data= time_ghz_execution(MAX_QBITS,ITER)
    med=get_medians(data)
    data["times"]=med
    plot_multiple_lines(data["max_qubits"],[list(data["times"].values())],["sqlite_mps"],"Number of Qubits","Time (s)","Sqlite MPS ")
    with open(f"./new_data/ghz_{SYSTEM}_{MAX_QBITS}_{ITER}_median.json", "w") as outfile: 
        json.dump(data, outfile)

if __name__ =="__main__":
    # data=get_current_baseline(50,500)
    # save_data_to_file(data,f"baselines/baseline_noudf_{SYSTEM}_{MAX_QBITS}_{ITER}_17_03_2024.json")
    file = open("./new_data/baselines/baseline_noudf_MAC_100_100_17_03_2024.json")
    old_data=json.load(file)
    compare_current_baseline(old_data)
    # data=json.load(file)
    # file=open("./data/ghz_hybrid_20_50_median.json")
    # data2=json.load(file)
    # med=get_medians(data)
    # data["times"]=med
    # plot_multiple_lines(data["max_qubits"],[list(data["times"].values()),list(data2["times"].values())],["sqlite_mps","hybrid_mps"],"Number of Qubits","Time (s)","Sqlite MPS VS Hybrid MPS")