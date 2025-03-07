from sqlite_mps import SQLITE_MPS
from timeit import default_timer as timer
import numpy as np
import json
from plotting import plot_multiple_lines
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
    with open("./data/"+filename, 'w') as f:
        json.dump(data, f)


    
MAX_QBITS=100
ITER=500
if __name__ =="__main__":
    
    data= time_ghz_execution(MAX_QBITS,ITER)
    med=get_medians(data)
    data["times"]=med
    plot_multiple_lines(data["max_qubits"],[list(data["times"].values())],["sqlite_mps"],"Number of Qubits","Time (s)","Sqlite MPS ")
    with open("./new_data/ghz_{MAX_QBITS}_{ITER}_median.json", "w") as outfile: 
        json.dump(data, outfile)
    # file = open("./data/ghz_20_100_sum.json")
    # data=json.load(file)
    # file=open("./data/ghz_hybrid_20_50_median.json")
    # data2=json.load(file)
    # med=get_medians(data)
    # data["times"]=med
    # plot_multiple_lines(data["max_qubits"],[list(data["times"].values()),list(data2["times"].values())],["sqlite_mps","hybrid_mps"],"Number of Qubits","Time (s)","Sqlite MPS VS Hybrid MPS")