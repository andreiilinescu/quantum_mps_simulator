from sqlite_mps import SQLITE_MPS
from plotting import plot_multiple_lines
from circuit_generator import *

from timeit import default_timer as timer
import numpy as np
import json
from termcolor import colored
import tracemalloc


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


def simulate_times(circuit_creator, num_qubits_list:list, num_iters):
    full_times={}
    for num_qubits in num_qubits_list:
        times=[]
        for _ in range(num_iters):
            circ = circuit_creator(num_qubits)
            meta=SQLITE_MPS.run_circuit_json(circ)
            times.append(np.sum(meta.times))
        full_times[str(num_qubits)]=times
    return full_times

def run_circuit(circ):
    metadata = SQLITE_MPS.run_circuit_json(circ)
    return {"total_exec_time":sum(metadata.times)}

MIN_QUBITS=200
MAX_QUBITS=1000
ITER=100
STEP=100
SYSTEM="PC-2080s"
def plot_save_data_ghz():
    data= time_ghz_execution(MAX_QUBITS,ITER)
    med=get_medians(data)
    data["times"]=med
    plot_multiple_lines(data["max_qubits"],[list(data["times"].values())],["sqlite_mps"],"Number of Qubits","Time (s)","Sqlite MPS ")
    with open(f"./new_data/ghz_{SYSTEM}_{MAX_QUBITS}_{ITER}_median.json", "w") as outfile: 
        json.dump(data, outfile)

if __name__ =="__main__":
    print(run_circuit(generate_qft_circuit(10)))
    # times=simulate_times(generate_ghz_circuit, list(range(MIN_QUBITS,MAX_QUBITS+1,STEP)), ITER)
    # for i in range(MIN_QUBITS,MAX_QUBITS+1,STEP):
    #     times[str(i)]=np.median(times[str(i)])
    # save_data_to_file({"simulator":"sqlite_mps","state":"GHZ","system":SYSTEM,"min_qubits":MIN_QUBITS,"max_qubits":MAX_QUBITS,"step_qubits":STEP,"iter":ITER,"times":times},f"ghz_{SYSTEM}_({MIN_QUBITS}_{MAX_QUBITS}_{STEP})_{ITER}.json")


    # data=time_ghz_execution(MAX_QBITS,ITER)
    # data["times"]=get_medians(data)
    # json.dump(data,open(f"./new_data/ghz_{SYSTEM}_{MAX_QBITS}_{ITER}_median_24_03_2025.json","w"))


    # file=open("./old/data/ghz_10_10_sum.json")
    # data2=json.load(file)
    # data2["times"]=get_medians(data2)
    # plot_multiple_lines(data["max_qubits"],[list(data["times"].values()),list(data2["times"].values())],["sqlite_mps","sqlite_mps(OLD)"],"Number of Qubits","Time (s)","Sqlite MPS VS OLD SQLITE MPS")


    # data=get_current_baseline(MAX_QBITS,ITER)
    # save_data_to_file(data,f"baselines/baseline_densesvd_{SYSTEM}_{MAX_QBITS}_{ITER}_24_03_2024.json")


    # file = open("./new_data/baselines/baseline_noudf_PC-2080s_100_200_24_03_2024.json")
    # old_data=json.load(file)
    # compare_current_baseline(old_data)

    # file=open("./new_data/ghz_PC-2080s_100_250_median_24_03_2025.json")
    # data=json.load(file)
    # file=open("./old/data/ghz_100_50_median.json")
    # data2=json.load(file)
    # med=get_medians(data)
    # data["times"]=med
    # plot_multiple_lines(data["max_qubits"],[list(data["times"].values()),list(data2["times"].values())],["sqlite_mps","old_sqlite_mps"],"Number of Qubits","Time (s)","SQLITE MPS VS OLD SQLITE MPS")


