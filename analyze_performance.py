from sqlite_mps import SQLITE_MPS
from timeit import default_timer as timer
import numpy as np
import json
from plotting import plot_multiple_lines
def generate_ghz_circuit(num_qubits:int):
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
        out[i]=np.median(tmp)
    return out

def save_data_to_file(data,filename:str):
    if not filename.endswith(".json"):
        filename+=".json"
    with open("./data/"+filename, 'w') as f:
        json.dump(data, f)


    

if __name__ =="__main__":
    file=open("data/ghz_100_10_sum.json")
    data=json.load(file)
    med=get_medians(data)
    plot_multiple_lines(data["max_qubits"],[list(med.values())],["sqlite_mps"],"Number of Qubits","Time (s)")